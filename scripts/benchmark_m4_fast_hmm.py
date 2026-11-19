"""Bench actual M3->M4 inference, optionally checking parity with original M2.

Examples (from repository root):
  python scripts/benchmark_m4_fast_hmm.py --trip 00000052 --compare
  python scripts/benchmark_m4_fast_hmm.py --trip 00000056

No reference .route files or candidate labels are accessed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline import load_trip
from backend.real_inference import load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trip', default='00000052')
    parser.add_argument('--trip-root', type=Path, default=Path('dataset/student_test'))
    parser.add_argument('--model', type=Path,
                        default=Path('artifacts/m3/random_forest_tuned_20261118T123057Z.joblib'))
    parser.add_argument('--config', type=Path,
                        default=Path('artifacts/m3/random_forest_tuned_20261118T123057Z.json'))
    parser.add_argument('--compare', action='store_true',
                        help='Compare original vs accelerated on a small trip (<=500 observations).')
    args = parser.parse_args()
    trip = load_trip(args.trip_root / args.trip, args.trip)
    if args.compare and len(trip.track) > 500:
        parser.error('--compare is restricted to <=500 observations to avoid original decoder memory/time spikes')
    os.environ['PATHSNAP_MODEL_PATH'] = str(args.model.resolve())
    os.environ['PATHSNAP_MODEL_CONFIG'] = str(args.config.resolve())
    os.environ['PATHSNAP_HMM_DECODER'] = 'backend.m2_hmm_adapter:decode_trip_for_m4'
    os.environ['PATHSNAP_PREDICTION_CACHE_SIZE'] = '0'

    def run(fast):
        os.environ['PATHSNAP_HMM_FAST'] = '1' if fast else '0'
        pipeline = load_pipeline()
        start = time.perf_counter()
        pred = pipeline(trip)
        seconds = time.perf_counter() - start
        print(f'{"FAST" if fast else "ORIGINAL"}: {seconds:.3f}s, '
              f'observations={len(pred.observation_arc_ids)}, route_arcs={len(pred.route_arc_ids)}',
              flush=True)
        return pred

    accelerated = run(True)
    if args.compare:
        baseline = run(False)
        print('Same observation arc IDs:', accelerated.observation_arc_ids == baseline.observation_arc_ids)
        print('Same route arc IDs:', accelerated.route_arc_ids == baseline.route_arc_ids)
        print('Same break indices:', accelerated.route_break_indices == baseline.route_break_indices)
        if accelerated != baseline:
            raise SystemExit('FAIL: accelerated result differs from original on this trip')
        print('PASS: outputs are identical')


if __name__ == '__main__':
    main()
