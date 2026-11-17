# CARLA closed-loop driving with the neural relight sensor

An ego vehicle laps the Thunderhill raceway in CARLA 0.9.16 while a YOLO26 detector on the
800x600 front camera looks for traffic cones and feeds them to the controller (MPC_DLC shifts its
reference path around detected cones). With relighting enabled, the front camera is relit by
FLUX.2-klein-4B + the Delight LoRA (the `relight` package at the repository root) before
detection, so the car keeps seeing cones at sunset and at night.

## Contents

| File | Purpose |
|------|---------|
| `run_sim.py` | The closed-loop client: builds the track, spawns cones, runs perception, control and metrics. |
| `controller2d_fixed.py` | Lateral/longitudinal controllers (PID throttle; MPC, MPC_DLC, Stanley, Pure Pursuit, NMPC steering). |
| `cutils.py` | Small helper used by the controller. |
| `live_plotter.py` | Tk/matplotlib live plots; falls back to headless (figures are still saved) when there is no display. |
| `options.cfg` | Live-plotting options read by `run_sim.py`. |
| `Thunderhilll.xodr` | OpenDRIVE description of the Thunderhill raceway (the world is generated from it at startup). |
| `stanford_waypoints_full.txt` | Reference racing line `x, y, v` (m, m, m/s). |
| `results/` | Default output location (created on first run). |

## Prerequisites

- Linux with an NVIDIA GPU. CARLA, the relighting model (FLUX.2-klein-4B in bf16) and YOLO
  share the GPU.
- **CARLA 0.9.16 server**: download `CARLA_0.9.16.tar.gz` from the
  [CARLA releases](https://github.com/carla-simulator/carla/releases/tag/0.9.16) page and extract it.
  No additional maps are needed; the track is generated from `Thunderhilll.xodr`.
- **Python client**: the `carla` package must match the server version:
  ```bash
  pip install carla==0.9.16
  ```
- Python packages: `torch` (CUDA), a `diffusers` version that provides `Flux2KleinPipeline`,
  `transformers`, `peft`, `ultralytics>=8.4` (YOLO26), `pygame`, `opencv-python`, `matplotlib`,
  `scipy`, `numpy`, `pillow`; `tkinter` (e.g. `python3-tk`) for the live plots.
  Tested with Python 3.11, torch 2.10 + CUDA 12.8, diffusers 0.42.0.dev0, ultralytics 8.4.19, carla 0.9.16.
- Weights: `weights/yolo26_carla.pt` (YOLO26 fine-tuned on CARLA cones) and `weights/delight_lora/`
  ship with this repository. `black-forest-labs/FLUX.2-klein-4b` is downloaded from the Hugging Face
  Hub on first use (set `HF_HOME` to choose the cache location).

## Start the CARLA server

```bash
cd /path/to/CARLA_0.9.16
./CarlaUE4.sh -quality-level=Low                    # with a spectator window
./CarlaUE4.sh -RenderOffScreen -quality-level=Low   # headless (no display needed)
```

The client connects to `127.0.0.1:2000` by default (`--host`, `--port`). It regenerates the
world from the OpenDRIVE file on every run and switches the server to synchronous mode with
`--fixed-delta-seconds` (default 0.1 s), so slow perception never changes what the car sees.
At startup `run_sim.py` kills any leftover `python run_sim.py` client so that a crashed run
cannot keep the server locked.

## Display

- `--disable-visualization` turns off the pygame window that shows the (relit) front camera with
  YOLO boxes. To keep it on a machine without a display, set `SDL_VIDEODRIVER=dummy`.
- `--disable-live-plotting` hides the Tk trajectory/controls plots. Without a display the plotter
  runs headless automatically; the plots are still written to the output folder.
- `--screenshot-sim-window` needs an X11 session plus `xdotool`/ImageMagick (best effort).

## Demo runs

Run from this folder (any working directory works; all paths resolve relative to `run_sim.py`).
`--time-of-day` sets the sun altitude: 12 is noon (90 deg), 18 is sunset (0 deg), 22 is night (-60 deg).
`--obstacle-probability 0.02` places a cone next to ~2% of the waypoints; `--random-seed` makes the
placement reproducible.

```bash
COMMON="--control-method MPC_DLC --fixed-delta-seconds 0.1 --weather clear --obstacle-probability 0.02 --random-seed 4100"

# Day, no relighting (upper bound)
python run_sim.py $COMMON --time-of-day 12 --condition-id Demo_Day

# Sunset, without and with relighting
python run_sim.py $COMMON --time-of-day 18 --condition-id Demo_Sunset
python run_sim.py $COMMON --time-of-day 18 --enable-relighting --async-relighting --condition-id Demo_Sunset_Relit

# Night, without relighting, with continuous relighting, with event-driven relighting
python run_sim.py $COMMON --time-of-day 22 --condition-id Demo_Night
python run_sim.py $COMMON --time-of-day 22 --enable-relighting --async-relighting --condition-id Demo_Night_Relit
python run_sim.py $COMMON --time-of-day 22 --enable-relighting --event-driven-relighting --condition-id Demo_Night_EventDriven

# Pixel-space baseline instead of relighting
python run_sim.py $COMMON --time-of-day 22 --enable-clahe --condition-id Demo_Night_CLAHE
```

Other useful flags: `--control-method {MPC,MPC_DLC,Stanley,PurePursuit,NMPC}`,
`--weather {clear,rainy,heavy_rain,foggy,foggy_rainy,cloudy}` (plus `--precipitation`,
`--fog-density`, `--fog-distance`, `--cloudiness`), `--disable-collision-avoidance`,
`--simulation-timeout SECONDS` (default 400 s of simulated time), `--max-speed-kph`,
`--enable-profiling` (per-stage timings every 100 frames), and
`--screenshot-progress 20 --screenshot-kind both` (saves the raw and the relit front image with YOLO
boxes when the car reaches 20% of the lap, then ends the run; add `--no-exit-after-screenshots` to
keep driving). Run `python run_sim.py --help` for the full list.

## Relighting modes

All modes relight the raw front-camera frame with one inference step at a long side of
1024 / `--relight-size-factor` pixels (default 4, i.e. 256x192 for the 800x600 camera).
`--base-model` and `--lora` select the model (defaults: `black-forest-labs/FLUX.2-klein-4b` and
`../weights/delight_lora`; both also accept Hugging Face Hub ids).

| Flags | Behaviour | `torch.compile` |
|-------|-----------|-----------------|
| `--enable-relighting --async-relighting` | Continuous, non-blocking: every frame is queued for a background thread (queue of 2, oldest dropped) and YOLO runs on the most recent relit frame. | off |
| `--enable-relighting` | Continuous, synchronous: every frame (or every N-th with `--relight-skip-frames N`) is relit before YOLO. | on |
| `--enable-relighting --event-driven-relighting` | YOLO first runs on the raw frame and gives `c_max`, its highest confidence. Relighting switches on when `c_max < --tau-low` (0.3) and off when `c_max > --tau-high` (0.5); frames with no detection keep the current state. While on, every frame is relit synchronously and YOLO runs on the relit frame. | on |

In the synchronous modes the first relit frame includes the `torch.compile` warm-up; the server
waits for the client, so this does not affect the drive.

## Outputs

Each run writes to `results/controller_output/` (overwritten every run) or to `--output-dir`:

- `lap_time.txt`, `collision_stats.txt` (cones spawned, unique cones hit), `path_tracking_error.txt`,
  `speed_tracking_error.txt`, `trajectory.txt`
- `trajectory.png`, `forward_speed.png`, `throttle_output.png`, `brake_output.png`, `steer_output.png`
- with `--mode record`: `tracking_data_with_cmax.csv` (pose, speed, controls, `c_max` per frame) and
  `yolo_cmax_data.csv` (per-frame `c_max` and detection count; event-driven runs also log
  `relighting_triggered` and `is_relighting_active`)
- with `--screenshot-progress`: `screenshots/`

With `--async-relighting` the run ends by printing how many frames the background thread relit
and dropped, and its mean relighting time over the last 100 frames.
