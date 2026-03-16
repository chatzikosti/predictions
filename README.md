# Stock Predictions

Minimal Python project scaffold for experimenting with stock prediction models.

## Project structure

- `app.py`: entry point
- `data/`: datasets (raw/processed), downloads, cached files
- `models/`: model definitions, training code, saved artifacts
- `utils/`: shared utilities (IO, features, evaluation, etc.)

## Run

```bash
python3 app.py
```

## Schedule daily runs

### Option A: APScheduler (recommended)

Install:

```bash
pip install apscheduler
```

Run a persistent scheduler (runs once at startup, then daily):

```bash
python3 scheduler.py --hour 9 --minute 0
```

Logs append to `data/predictions.log` by default.

### Option B: cron (macOS/Linux)

Example: run every day at 9:00am and append output to a log.

```bash
crontab -e
```

Add:

```bash
0 9 * * * cd "/Users/user/Applications/predictions" && /usr/bin/python3 app.py >> data/predictions.log 2>&1
```

## Optional UI (later)

If you want a quick interface later, a common next step is a small web app (e.g. Streamlit) that reads the latest results/log and displays recommendations.
