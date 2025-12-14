#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor training progress for all Nahawi models.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import re

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

TASK_FILES = {
    'GeneralGEC': '/tmp/claude/tasks/ba88e76.output',
    'HamzaFixer': '/tmp/claude/tasks/b5179d5.output',
    'SpaceFixer': '/tmp/claude/tasks/be39a5c.output',
    'SpellingFixer': '/tmp/claude/tasks/be4de29.output',
    'DeletedWordFixer': '/tmp/claude/tasks/b504d42.output',
    'MorphologyFixer': '/tmp/claude/tasks/b423d50.output',
}


def parse_progress(line):
    """Parse progress from tqdm output."""
    # Pattern: 3%|...| 1024/37500 [56:33<63:01:56, 6.22s/it]
    match = re.search(r'(\d+)%\|.*\|\s*(\d+)/(\d+)\s*\[[\d:]+<([\d:]+)', line)
    if match:
        pct = int(match.group(1))
        current = int(match.group(2))
        total = int(match.group(3))
        remaining = match.group(4)
        return pct, current, total, remaining
    return None


def get_last_line(filepath):
    """Get the last non-empty line from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if line.strip():
                    return line.strip()
    except:
        pass
    return None


def check_status():
    """Check training status for all models."""
    print(f"\n{'='*60}")
    print(f"NAHAWI TRAINING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for name, filepath in TASK_FILES.items():
        line = get_last_line(filepath)
        if line:
            progress = parse_progress(line)
            if progress:
                pct, current, total, remaining = progress
                bar = '#' * (pct // 5) + '-' * (20 - pct // 5)
                print(f"[{bar}] {pct:3}% {name:20} ({current:>6}/{total}) ETA: {remaining}")
            else:
                print(f"[{'?'*20}]  ?% {name:20} (parsing...)")
        else:
            print(f"[{'.'*20}]  0% {name:20} (not started)")

    print()


def main():
    """Run once or loop."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', action='store_true', help='Continuously monitor')
    parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    args = parser.parse_args()

    if args.loop:
        print("Monitoring training progress (Ctrl+C to stop)...")
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                check_status()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped monitoring.")
    else:
        check_status()


if __name__ == '__main__':
    main()
