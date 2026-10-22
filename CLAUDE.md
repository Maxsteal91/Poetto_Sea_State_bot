# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Telegram bot that classifies the sea state at Poetto Beach (Cagliari) as `mare_calmo` / `mare_mosso` by running a fine-tuned 3D ResNet-18 (`torchvision` `r3d_18`, 2-class head) on a short clip grabbed from a public HLS webcam stream. User-facing text is in Italian/English mix; code comments are mostly English.

## Running

- Requires `TELEGRAM_TOKEN` in the environment or in a local `.env` (loaded via `python-dotenv` only if the file exists). The script raises at import time if it is missing.
- Requires `ffmpeg` on PATH for `.ts` → `.mp4` conversion (falls back to OpenCV `VideoWriter` if ffmpeg fails).
- Run locally: `uv run python telegram_poetto_bot.py` (no `pyproject.toml` yet; dependencies live in `requirements.txt`). Use a separate test bot token: two processes polling with the same token conflict.
- Docker: `docker build -t poetto-bot .` then `docker run -e TELEGRAM_TOKEN=... poetto-bot`. `Procfile` declares it as a `worker` process (Heroku-style deploy).
- `get_wind_info.py` and `sea_conditions.py` (Open-Meteo wind, waves, water temperature, forecast) are imported by the bot and can also run standalone (`uv run python sea_conditions.py`).

There are no tests, linter config, or CI in this repo.

## Dependency gotchas

- `torch` / `torchvision` are **not** in `requirements.txt`: the Dockerfile installs `torchvision` from the PyTorch CPU wheel index before the rest. Install them separately for local runs.
- `requirements.txt` is saved as UTF-16 (LE, with BOM). Keep that in mind when editing or parsing it with tools that assume UTF-8.
- The model weights `model/r3d_18_poetto.pth` (too big for GitHub, >100 MB) are gitignored. On startup the bot downloads them from Google Drive via `gdown` (`MODEL_GDRIVE_ID`) if absent, and raises if the download fails.

## Architecture (`telegram_poetto_bot.py`, plus Open-Meteo helpers in `get_wind_info.py` and `sea_conditions.py`)

Module-level side effects happen at import: env/token loading, `./temp`, `./data` and `./model` creation, model download, model load onto `DEVICE`. Importing the module therefore needs the token and the weights.

Analysis pipeline (`analyze_sea`, blocking, run via `asyncio.to_thread` under `analysis_lock` from `run_inference`):
1. `download_segments` fetches the `M3U8_URL` playlist and downloads the `N_SEGMENTS` most recent `.ts` segments (most recent first) into a per-request `tempfile.TemporaryDirectory` under `./temp`.
2. `load_video` samples `FRAME_COUNT` (16) evenly spaced frames, crops to `ROI` (x1, y1, x2, y2 in source-pixel coords), resizes to 112×112 → tensor shaped `(1, C, T, H, W)`.
3. `predict_probs` gives softmax probabilities for one segment; `predict` averages them over all segments and returns `(label, confidence, top_class)`, where `label` is `UNCERTAIN_LABEL` when the averaged confidence is below `CONFIDENCE_THRESHOLD`.
4. The most recent segment becomes a frame (`extract_frame`) or MP4 (`convert_ts_to_mp4`) per chat `output_format`; wind comes from Open-Meteo (`get_wind`, `None` on failure), waves and water temperature from Open-Meteo Marine (`sea_conditions.get_marine_now`). `marine_mismatch_note` warns when the webcam label and the wave estimate clearly disagree.
5. `build_caption` + `send_analysis` send it with a random message (`MESSAGES_CALM` / `MESSAGES_ROUGH` / `MESSAGES_UNCERTAIN`) and an "Analizza di nuovo" button.

`ROI` is tied to the current webcam framing; when the camera moves, the crop must be re-tuned (an alternative ROI is kept commented out next to it). `class_names` order must match the training label order of the `.pth` weights.

Bot state:
- Per-chat settings live in `context.chat_data` (`notify_time` "HH:MM" or `None`, `output_format`), saved by `PicklePersistence` to `data/bot_state.pickle`. In Docker, mount `data/` as a volume or settings are lost on redeploy.
- PTB does not persist jobs: `post_init` re-creates the daily jobs from persisted `chat_data` and registers the command menu (`BOT_COMMANDS`).
- Daily notifications use `job_queue.run_daily` with `TIMEZONE` (`zoneinfo` Europe/Rome, DST-safe), one job per chat named `daily_inference_{chat_id}`. Notifications are off until the user picks a time. If a user blocks the bot (`Forbidden`), their notification is disabled.
- `concurrent_updates(True)`: handlers run concurrently; `chats_in_progress` ignores repeated requests from the same chat.
- Commands: `/start`, `/analizza` (alias `/inference`), `/previsioni` (wave/wind forecast by time slot, `sea_conditions.get_forecast`), `/notifica` (alias `/settings`), `/stop`, `/output`. Callback data is `action:value` (`output:frame|video`, `notify:HH:MM|off`, `inference_now`). After `/notifica`, free text is parsed by `parse_notify_time` (flag `user_data['awaiting_notify_time']`).
- User-facing texts are in Italian.
