from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_app(log_path: Path | None = None) -> None:
    """
    Run app.py in a subprocess so the scheduler stays isolated from app deps.
    """
    cmd = [sys.executable, str(Path(__file__).with_name("app.py"))]
    ts = datetime.now().isoformat(timespec="seconds")

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n\n=== run {ts} ===\n")
            f.flush()
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    else:
        subprocess.run(cmd, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily scheduler for Stock Predictions")
    parser.add_argument("--hour", type=int, default=9, help="Hour to run daily (0-23)")
    parser.add_argument("--minute", type=int, default=0, help="Minute to run daily (0-59)")
    parser.add_argument(
        "--log",
        type=str,
        default="data/predictions.log",
        help="Path to append logs (set empty to disable)",
    )
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else None

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ModuleNotFoundError as e:
        print("[error] Missing dependency: APScheduler")
        print("Install: pip install apscheduler")
        raise

    sched = BlockingScheduler()
    trigger = CronTrigger(hour=args.hour, minute=args.minute)
    sched.add_job(lambda: run_app(log_path), trigger=trigger, id="daily_predictions")

    print("Scheduler started.")
    print(f"- Runs daily at {args.hour:02d}:{args.minute:02d}")
    if log_path:
        print(f"- Logs appended to: {log_path}")
    else:
        print("- Logging disabled")
    print("Press Ctrl+C to stop.")

    # Optional: run once immediately at startup
    run_app(log_path)
    sched.start()


if __name__ == "__main__":
    main()
