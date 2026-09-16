"""48-turn event comparison; uses the existing event engine and metrics."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from .event_format_experiment import EventDesign, PROJECT_ROOT, run_study

DESIGN=EventDesign(json.loads((PROJECT_ROOT/'poc/config/event_formats_v2.json').read_text(encoding='utf-8')))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed-start',type=int,default=101)
    p.add_argument('--seed-end',type=int,default=1100)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    try:
        run_study(a.output or PROJECT_ROOT/'results'/datetime.now().strftime('event-formats-48-%Y%m%d-%H%M%S-%f'),a.seed_start,a.seed_end,a.workers,DESIGN)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        p.exit(1,f'Experiment failed: {exc}; missing manifest means incomplete.\n')

if __name__=='__main__': main()
