#!/usr/bin/env python3
"""Read-only acceptance audit for experimental R2 full-resume checkpoints."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--world-size', type=int, default=2)
    args = p.parse_args()
    d = args.directory
    marker = d / 'resume_manifest.json'
    worker = d / 'worker_resume_complete.json'
    snapshot = d / 'snapshot_manifest.json'
    driver = d / 'driver_state.pt'
    for file in [marker, worker, snapshot, driver]:
        if not file.is_file() or file.stat().st_size == 0:
            raise RuntimeError(f'Missing or empty: {file}')
    meta = json.loads(marker.read_text())
    wmeta = json.loads(worker.read_text())
    modelmeta = json.loads(snapshot.read_text())
    logical_step = meta['logical_step']
    if wmeta['logical_step'] != logical_step or modelmeta['logical_step'] != logical_step:
        raise RuntimeError('Step numbers disagree across manifests')
    for rank in range(args.world_size):
        for part in ('model', 'optim', 'extra_state'):
            f = d / f'{part}_world_size_{args.world_size}_rank_{rank}.pt'
            if not f.is_file() or f.stat().st_size < 1024:
                raise RuntimeError(f'Missing/truncated: {f}')
            print(f'FOUND {f.name}: {f.stat().st_size / 2**30:.3f} GiB')
    if list(d.glob('*.incomplete')):
        raise RuntimeError('Incomplete files exist')
    import torch
    state = torch.load(driver, map_location='cpu', weights_only=False)
    if state['logical_step'] != logical_step:
        raise RuntimeError('Driver step mismatch')
    for rank in range(args.world_size):
        extra = torch.load(d / f'extra_state_world_size_{args.world_size}_rank_{rank}.pt',
                           map_location='cpu', weights_only=False)
        if extra['logical_step'] != logical_step:
            raise RuntimeError('Worker step mismatch')
        if not ('rng' in extra and 'scheduler' in extra):
            raise RuntimeError('Missing RNG or scheduler state')
    print('FULL_RESUME_FILES_AND_STATE_AUDIT_OK')
    print('LOGICAL_STEP:', logical_step)
    print('MODEL_ORIG:', meta['signature'].get('model'))
    print('NOTE: does not validate full optimizer deserialization or successful training resume')


if __name__ == '__main__':
    main()
