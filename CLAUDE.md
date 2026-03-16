# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **greenfield Python project** building a computer vision-based rover tracking and analytics system for MIE444 (Mechatronics Engineering Design). The system automatically tracks autonomous rovers navigating an 8ft × 4ft maze using overhead cameras, generating performance analytics for student evaluation.

There is currently **no implementation** — only sample videos in `data_/`. Ask the user before making significant architectural decisions.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the tracking pipeline (uses config.yaml defaults)
python3 main.py

# Run with a specific video
python3 main.py --video data_/WIN_20260225_15_07_32_Pro.mp4

# Show live debug window while processing
python3 main.py --display

# Interactive homography calibration (click 4 maze corners)
python3 main.py --calibrate

# Run the visualization dashboard
streamlit run dashboard.py

# Run all tests
python3 -m pytest tests/ -v

# Run a single test
python3 -m pytest tests/test_event_detector.py::test_wall_collision_west -v
```

## Architecture

The system is designed as **six loosely coupled layers**. Each layer should be an independent module with a clearly defined interface:

| Layer | Responsibility | Key Output |
|---|---|---|
| **Perception** | Rover detection, tracking, position/orientation estimation | Rover state per frame |
| **Event Detection** | Wall collisions, stops, manual intervention, off-track behavior | Timestamped event log |
| **State Representation** | `[x, y, velocity, heading, timestamp, event_flags]` vector | Rover state vector |
| **Data Management** | Log/store trajectory, events, trial metadata | JSON/CSV/DB files |
| **Visualization** | Dashboard showing path, distance, events, speed | Streamlit or Dash UI |
| **Sensor/Input** | Video file or live USB overhead camera feed | Raw frames |

Perception is the core — use OpenCV (background subtraction, feature tracking) or an object detection model for rover detection.

## Configuration-Driven Design

All parameters must live in config files, not hardcoded. Key values to configure:
- Maze dimensions: **Length = 8ft, Width = 4ft**
- Camera intrinsics / homography matrix
- Detection thresholds (collision distance, stop duration, etc.)

## Reproducibility

Every trial run must generate:
1. A config snapshot (copy of the config used)
2. A trajectory log (per-frame rover state)
3. An event log (timestamped events)

## Data

`data_/` contains two reference videos (`WIN_20260225_15_*.mp4`) recorded 2026-02-25. Use these for algorithm development and validation.

## Key Design Rules

- **Modular:** Each module (tracking, collision detection, data logging, visualization) must be independently testable and importable.
- **No tight coupling:** Modules communicate via well-defined interfaces, not shared global state.
- **Ask when unclear** — the spec has open choices (e.g., OpenCV vs. detection model, Streamlit vs. Dash, local DB vs. CSV). Confirm with the user before committing to an approach.
