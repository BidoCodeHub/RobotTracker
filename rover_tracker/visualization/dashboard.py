"""Streamlit dashboard for viewing completed trial results."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

MM_TO_FT = 1 / 304.8


# ── data helpers ──────────────────────────────────────────────────────────────

def _load_trial(trial_dir: Path) -> tuple[pd.DataFrame, list[dict], dict]:
    import json
    try:
        traj = pd.read_csv(trial_dir / "trajectory.csv") if (trial_dir / "trajectory.csv").exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        traj = pd.DataFrame()
    events_path = trial_dir / "events.json"
    events = json.loads(events_path.read_text()) if events_path.exists() else []
    cfg_path = trial_dir / "config_snapshot.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    return traj, events, cfg


def _compute_stats(traj: pd.DataFrame, events: list[dict]) -> dict:
    if traj.empty:
        return {}
    duration = traj["timestamp_s"].iloc[-1] - traj["timestamp_s"].iloc[0] if len(traj) > 1 else 0
    # Downsample to ~1-second position windows before computing distance.
    # Averaging ~30 frames reduces per-frame centroid jitter (σ~40mm) by √30 ≈ 5×,
    # so accumulated noise over a 1-min trial drops from ~200 ft to ~1 ft.
    # Median filter removes glitch spikes; wide window handles slow rotation cycles.
    x_filt = traj["x_mm"].rolling(15, min_periods=1, center=True).median()
    y_filt = traj["y_mm"].rolling(15, min_periods=1, center=True).median()

    # Downsample to ~2-second chunks so a full rotation cycle averages back to ~zero.
    dt_med = traj["timestamp_s"].diff().median()
    fps_est = max(1, round(1.0 / dt_med)) if (dt_med and dt_med > 0) else 30
    chunk_size = max(fps_est * 2, 1)   # 2-second windows
    groups = traj.index // chunk_size
    x_c = x_filt.groupby(groups).mean()
    y_c = y_filt.groupby(groups).mean()

    # Dead-band: rotation-in-place drifts < 40 mm per 2-s chunk; real translation exceeds it.
    # Max-cap: if a chunk-to-chunk step is implausibly large (> 400 mm in 2 s) it is a
    # tracker glitch / teleport — skip it entirely so it doesn't inflate distance.
    MIN_STEP_MM = 40.0
    MAX_STEP_MM = 400.0
    dist_mm = sum(
        d for i in range(1, len(x_c))
        if MIN_STEP_MM < (d := math.hypot(x_c.iloc[i] - x_c.iloc[i - 1],
                                           y_c.iloc[i] - y_c.iloc[i - 1])) <= MAX_STEP_MM
    )
    # Smooth velocity before computing stats to remove tracking noise spikes
    vel_smooth = traj["velocity_mms"].rolling(10, min_periods=1).mean()
    return {
        "duration_s":      duration,
        "distance_ft":     dist_mm * MM_TO_FT,
        "max_speed_fts":   vel_smooth.max() * MM_TO_FT,
        "avg_speed_fts":   vel_smooth.mean() * MM_TO_FT,
        "frames":          len(traj),
        "collisions":      sum(1 for e in events if e.get("event_type") == "wall_collision"),
        "interventions":   sum(1 for e in events if e.get("event_type") == "manual_intervention"),
    }


def _achievements(stats: dict) -> list[tuple[str, str, str]]:
    out = []
    if stats.get("max_speed_fts", 0) > 3:
        out.append(("🚀", "Speed Demon",      f"Hit {stats['max_speed_fts']:.1f} ft/s!"))
    if stats.get("collisions", 99) == 0:
        out.append(("🎯", "Smooth Operator",  "Zero wall collisions!"))
    elif stats.get("collisions", 99) <= 2:
        out.append(("✨", "Careful Driver",   f"Only {stats['collisions']} collision(s)"))
    if stats.get("interventions", 99) == 0:
        out.append(("🤖", "Fully Autonomous", "No human interventions!"))
    if stats.get("duration_s", 999) < 60:
        out.append(("⚡", "Speed Run",        f"Done in {stats['duration_s']:.0f} s!"))
    if stats.get("distance_ft", 0) > 15:
        out.append(("🏃", "Marathon",         f"{stats['distance_ft']:.1f} ft travelled"))
    if not out:
        out.append(("🎮", "Keep Going!",      "Run more trials to unlock badges"))
    return out


def _maze_shapes(fig: go.Figure, maze_x_max: float, maze_y_max: float,
                 maze_obstacles: list) -> None:
    fig.add_shape(type="rect", x0=0, y0=0, x1=maze_x_max, y1=maze_y_max,
                  line=dict(color="black", width=4))
    for obs in maze_obstacles:
        x0, y0, x1_o, y1_o = obs
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1_o, y1=y1_o,
                      fillcolor="black", line=dict(color="black", width=1))


def _maze_layout(fig: go.Figure, height: int = 650) -> None:
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, title="x (ft)"),
        yaxis=dict(showgrid=False, zeroline=False, title="y (ft)", autorange="reversed"),
    )


# ── main dashboard ─────────────────────────────────────────────────────────────

def run_dashboard():
    st.set_page_config(page_title="RobotTracker", layout="wide", page_icon="🏁")

    st.markdown(
        "<h1 style='text-align:center'>🏁 RobotTracker</h1>"
        "<p style='text-align:center;color:#888;font-size:1.05em'>"
        "Real-time analytics · Race replay · Leaderboard</p>",
        unsafe_allow_html=True,
    )

    # ── trial discovery ──
    trials_root = Path("trials")
    if not trials_root.exists():
        st.warning("No trials directory found. Run `python main.py` first.")
        return

    trial_dirs = sorted(
        [d for d in trials_root.iterdir()
         if d.is_dir() and (
             # Active trial (recording in progress) — include even before data arrives
             (d / "recording.lock").exists()
             # Completed trial with trajectory data
             or ((d / "trajectory.csv").exists()
                 and (d / "trajectory.csv").stat().st_size > 0)
         )],
        reverse=True,
    )
    if not trial_dirs:
        st.info("No trials yet. Start a trial in the Operator UI.")
        import time; time.sleep(2); st.rerun()
        return

    trial_names = [d.name for d in trial_dirs]

    # Detect live trial: recording.lock exists AND trajectory is actively being written.
    # The 30-second mtime check guards against stale locks left by a crashed process.
    def _is_live(d: Path) -> bool:
        import time as _time
        lock = d / "recording.lock"
        if not lock.exists():
            return False
        traj = d / "trajectory.csv"
        ref  = traj if traj.exists() else lock
        return (_time.time() - ref.stat().st_mtime) < 30

    live_dir = next((d for d in trial_dirs if _is_live(d)), None)
    is_live  = live_dir is not None

    # ── sidebar ──
    st.sidebar.header("Controls")

    # Force the selectbox to follow the live trial by writing to session_state
    # before the widget renders — Streamlit honours pre-set session_state values.
    SELECT_KEY = "_trial_select"
    if live_dir:
        st.session_state[SELECT_KEY] = live_dir.name
    elif SELECT_KEY not in st.session_state:
        st.session_state[SELECT_KEY] = trial_names[0]
    elif st.session_state[SELECT_KEY] not in trial_names:
        st.session_state[SELECT_KEY] = trial_names[0]

    selected = st.sidebar.selectbox("🏎️ Trial", trial_names, key=SELECT_KEY)

    # Auto-refresh is always on during a live trial
    auto_refresh = st.sidebar.toggle("🔄 Auto-refresh", value=is_live)
    refresh_s    = st.sidebar.number_input("Interval (s)", min_value=1, value=2,
                                           disabled=not auto_refresh)
    path_smooth  = st.sidebar.slider("Path smoothing", 1, 50, 10)
    st.sidebar.caption(f"`{selected}`")

    if is_live and selected == live_dir.name:
        st.sidebar.markdown(
            "<div style='background:#1a3a1a;border:1px solid #3fb950;"
            "border-radius:6px;padding:8px;text-align:center;"
            "color:#3fb950;font-weight:bold'>🔴 LIVE — trial in progress</div>",
            unsafe_allow_html=True,
        )

    trial_dir = trials_root / selected
    traj, events, cfg = _load_trial(trial_dir)

    if traj.empty:
        st.info("⏳ Waiting for first frames…")
        import time; time.sleep(2); st.rerun()
        return

    # ── live / rover name banner ──
    rover_name = cfg.get("trial", {}).get("rover_name", "")
    is_this_live = is_live and selected == live_dir.name

    if is_this_live:
        st.markdown(
            "<div style='background:linear-gradient(90deg,#0d2b0d,#1a3a1a);"
            "border:1px solid #3fb950;border-radius:8px;padding:10px;"
            "text-align:center;margin-bottom:8px'>"
            "<span style='color:#3fb950;font-size:1.3em;font-weight:bold'>"
            "🔴 LIVE — Recording in progress</span></div>",
            unsafe_allow_html=True,
        )
    if rover_name:
        st.markdown(
            f"<h2 style='text-align:center;color:#f0c040;margin-top:-4px'>"
            f"🤖 {rover_name}</h2>",
            unsafe_allow_html=True,
        )

    # ── config values ──
    vcfg  = cfg.get("visualization", {})
    event_colors = vcfg.get("event_colors", {
        "wall_collision":      "#d62728",
        "manual_intervention": "#9467bd",
    })
    maze_x_max   = cfg.get("maze", {}).get("length_mm", 2438.4) * MM_TO_FT
    maze_y_max   = cfg.get("maze", {}).get("width_mm",  1219.2) * MM_TO_FT
    maze_obs_ft  = [[v * MM_TO_FT for v in o] for o in cfg.get("maze", {}).get("obstacles", [])]
    smooth_win   = vcfg.get("speed_smoothing_window", 15)

    events = [e for e in events if e.get("event_type") != "stop"]
    stats  = _compute_stats(traj, events)

    # ── pre-compute display series ──
    traj["speed_smooth"] = traj["velocity_mms"].rolling(smooth_win, min_periods=1).mean() * MM_TO_FT
    x_ft = traj["x_mm"].rolling(path_smooth, min_periods=1).mean() * MM_TO_FT
    y_ft = traj["y_mm"].rolling(path_smooth, min_periods=1).mean() * MM_TO_FT
    ev_df = pd.DataFrame(events) if events else pd.DataFrame(columns=["event_type", "x_mm", "y_mm"])

    # ── summary metrics row ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("⏱️ Duration",       f"{stats['duration_s']:.1f} s")
    c2.metric("📏 Distance",       f"{stats['distance_ft']:.2f} ft")
    c3.metric("🚀 Top Speed",      f"{stats['max_speed_fts']:.2f} ft/s")
    c4.metric("💨 Avg Speed",      f"{stats['avg_speed_fts']:.2f} ft/s")
    c5.metric("💥 Collisions",     str(stats["collisions"]))
    c6.metric("🖐️ Interventions",  str(stats["interventions"]))

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════
    tab_path, tab_replay, tab_heat, tab_lb = st.tabs(
        ["🗺️ Path", "📽️ Replay", "🔥 Heatmap", "🏆 Leaderboard"]
    )

    # ─────────────────────────────────────────────────────
    # TAB 1 — F1-style path + speedometer
    # ─────────────────────────────────────────────────────
    with tab_path:
        # Full-width map
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_ft, y=y_ft,
            mode="markers",
            name="Path (speed)",
            marker=dict(
                size=5,
                color=traj["speed_smooth"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Speed<br>(ft/s)", thickness=16, len=0.6),
            ),
            hovertemplate="x: %{x:.2f} ft<br>y: %{y:.2f} ft<br>speed: %{marker.color:.2f} ft/s<extra></extra>",
        ))
        for etype, color in event_colors.items():
            sub = ev_df[ev_df["event_type"] == etype] if len(ev_df) else ev_df
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["x_mm"] * MM_TO_FT, y=sub["y_mm"] * MM_TO_FT,
                    mode="markers", name=etype.replace("_", " ").title(),
                    marker=dict(color=color, size=12, symbol="x"),
                ))
        _maze_shapes(fig, maze_x_max, maze_y_max, maze_obs_ft)
        _maze_layout(fig, height=650)
        st.plotly_chart(fig, use_container_width=True)

        # Speedometer + speed timeline on the same row
        max_spd = max(stats["max_speed_fts"], 0.01)
        gauge_col, timeline_col = st.columns([1, 3])

        with gauge_col:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stats["avg_speed_fts"],
                title={"text": "Avg Speed<br><span style='font-size:0.8em;color:#888'>ft/s</span>"},
                number={"suffix": " ft/s", "font": {"size": 26}},
                gauge={
                    "axis": {"range": [0, max_spd * 1.2]},
                    "bar":  {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0,              max_spd * 0.33], "color": "#ffcccc"},
                        {"range": [max_spd * 0.33, max_spd * 0.66], "color": "#ffe0aa"},
                        {"range": [max_spd * 0.66, max_spd * 1.2],  "color": "#ccffcc"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75, "value": max_spd,
                    },
                },
            ))
            gauge.update_layout(height=260, margin=dict(l=10, r=10, t=60, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge, use_container_width=True)
            st.metric("🏎️ Top Speed", f"{max_spd:.2f} ft/s")

        with timeline_col:
            st.subheader("Speed Over Time")
            fig2 = px.line(
                traj, x="timestamp_s", y="speed_smooth",
                labels={"timestamp_s": "Time (s)", "speed_smooth": "Speed (ft/s)"},
                color_discrete_sequence=["#1f77b4"],
            )
            fig2.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=20),
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # TAB 2 — Animated race replay
    # ─────────────────────────────────────────────────────
    with tab_replay:
        st.subheader("📽️ Race Replay")
        st.caption("Press ▶ Play to watch the full run, or drag the slider.")

        n    = len(traj)
        step = max(1, n // 120)           # cap at ~120 animation frames
        idxs = list(range(step, n + 1, step))
        if idxs[-1] != n:
            idxs.append(n)

        spd_max = traj["speed_smooth"].max() or 1.0

        frames = [
            go.Frame(
                data=[go.Scatter(
                    x=x_ft.iloc[:fi], y=y_ft.iloc[:fi],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=traj["speed_smooth"].iloc[:fi],
                        colorscale="RdYlGn",
                        cmin=0, cmax=spd_max,
                        showscale=False,
                    ),
                )],
                name=str(fi),
            )
            for fi in idxs
        ]

        fig_anim = go.Figure(
            data=[go.Scatter(
                x=x_ft.iloc[:1], y=y_ft.iloc[:1],
                mode="markers",
                marker=dict(size=4, color=traj["speed_smooth"].iloc[:1],
                            colorscale="RdYlGn", cmin=0, cmax=spd_max,
                            showscale=True,
                            colorbar=dict(title="ft/s", thickness=12, len=0.5)),
            )],
            frames=frames,
        )
        _maze_shapes(fig_anim, maze_x_max, maze_y_max, maze_obs_ft)
        _maze_layout(fig_anim, height=650)
        fig_anim.update_layout(
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=1.12, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 40, "redraw": True},
                                      "fromcurrent": True,
                                      "transition": {"duration": 0}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}]),
                ],
            )],
            sliders=[dict(
                steps=[
                    dict(method="animate",
                         args=[[str(fi)], {"frame": {"duration": 0, "redraw": True},
                                           "mode": "immediate"}],
                         label=f"{traj['timestamp_s'].iloc[min(fi - 1, n - 1)]:.1f}s")
                    for fi in idxs
                ],
                x=0.05, len=0.9, y=0,
                currentvalue=dict(prefix="Time: ", visible=True, font={"size": 13}),
            )],
        )
        st.plotly_chart(fig_anim, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # TAB 3 — Heatmap
    # ─────────────────────────────────────────────────────
    with tab_heat:
        st.subheader("🔥 Position Heatmap")
        st.caption("Brighter areas = rover spent more time there")

        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram2dContour(
            x=traj["x_mm"] * MM_TO_FT,
            y=traj["y_mm"] * MM_TO_FT,
            colorscale="Hot",
            reversescale=True,
            showscale=True,
            ncontours=25,
            colorbar=dict(title="Dwell"),
        ))
        _maze_shapes(fig_h, maze_x_max, maze_y_max, maze_obs_ft)
        _maze_layout(fig_h, height=650)
        st.plotly_chart(fig_h, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # TAB 4 — Leaderboard
    # ─────────────────────────────────────────────────────
    with tab_lb:
        st.subheader("🏆 Trial Leaderboard")
        rows = []
        for td in trial_dirs:
            try:
                t, ev, c = _load_trial(td)
                if t.empty:
                    continue
                ev = [e for e in ev if e.get("event_type") != "stop"]
                s  = _compute_stats(t, ev)
                name = c.get("trial", {}).get("rover_name", "") or td.name
                rows.append({
                    "Rover":             name,
                    "Trial":             td.name,
                    "Duration (s)":      round(s["duration_s"], 1),
                    "Distance (ft)":     round(s["distance_ft"], 2),
                    "Top Speed (ft/s)":  round(s["max_speed_fts"], 2),
                    "Avg Speed (ft/s)":  round(s["avg_speed_fts"], 2),
                    "Collisions":        s["collisions"],
                    "Interventions":     s["interventions"],
                })
            except Exception:
                pass

        if rows:
            lb = (pd.DataFrame(rows)
                  .sort_values("Avg Speed (ft/s)", ascending=False)
                  .reset_index(drop=True))
            lb.index += 1

            def _hl(row):
                bg = "background-color:#1a3a5c;font-weight:bold" if row["Trial"] == selected else ""
                return [bg] * len(row)

            st.dataframe(lb.style.apply(_hl, axis=1), use_container_width=True)

        # Multi-trial path overlay
        if len(trial_dirs) > 1:
            st.subheader("Path Comparison")
            fig_cmp = go.Figure()
            palette = px.colors.qualitative.Plotly
            for i, td in enumerate(trial_dirs[:8]):
                try:
                    t, _, _ = _load_trial(td)
                    if t.empty:
                        continue
                    xc = t["x_mm"].rolling(10, min_periods=1).mean() * MM_TO_FT
                    yc = t["y_mm"].rolling(10, min_periods=1).mean() * MM_TO_FT
                    fig_cmp.add_trace(go.Scatter(
                        x=xc, y=yc,
                        mode="lines",
                        name=td.name,
                        line=dict(color=palette[i % len(palette)], width=2),
                        opacity=0.9 if td.name == selected else 0.45,
                    ))
                except Exception:
                    pass
            _maze_shapes(fig_cmp, maze_x_max, maze_y_max, maze_obs_ft)
            _maze_layout(fig_cmp, height=440)
            st.plotly_chart(fig_cmp, use_container_width=True)

    # ── event log (collapsed) ──
    if events:
        with st.expander("📋 Event Log", expanded=False):
            st.dataframe(pd.DataFrame(events), use_container_width=True)

    # ── auto-refresh — always on during live trial ──
    if auto_refresh or is_live:
        import time
        time.sleep(refresh_s if auto_refresh else 2)
        st.rerun()
