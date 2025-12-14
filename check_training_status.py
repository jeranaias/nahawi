#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick check on training status for all models.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import json
from datetime import datetime

OUTPUT_DIR = Path('C:/nahawi/nahawi_ensemble/checkpoints')

def check_model(name):
    model_dir = OUTPUT_DIR / name

    if not model_dir.exists():
        return None

    # Check for final model
    final_dir = model_dir / 'final'
    if final_dir.exists():
        info_file = model_dir / 'training_info.json'
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
                return {
                    'status': 'COMPLETE',
                    'duration': info.get('duration_minutes', 0),
                    'train_size': info.get('train_size', 0),
                    'timestamp': info.get('timestamp', ''),
                }
        return {'status': 'COMPLETE', 'duration': 0}

    # Check for checkpoint
    checkpoints = list(model_dir.glob('checkpoint-*'))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return {
            'status': 'TRAINING',
            'latest_checkpoint': latest.name,
        }

    return {'status': 'NOT_STARTED'}


def main():
    print("=" * 60)
    print("NAHAWI ENSEMBLE TRAINING STATUS")
    print(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    models = [
        'general_gec',
        'hamza_fixer',
        'space_fixer',
        'spelling_fixer',
    ]

    for name in models:
        info = check_model(name)
        if info is None:
            print(f"[  ----  ] {name}: Not started")
        elif info['status'] == 'COMPLETE':
            mins = info.get('duration', 0)
            hrs = mins / 60
            print(f"[COMPLETE] {name}: Done in {hrs:.1f}h, {info.get('train_size', '?')} pairs")
        elif info['status'] == 'TRAINING':
            print(f"[TRAINING] {name}: {info.get('latest_checkpoint', 'in progress')}")
        else:
            print(f"[  ----  ] {name}: Not started")

    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
