#!/usr/bin/env python3
"""Audit two non-destructive R2-Searcher FSDP model-only snapshots.

Run only AFTER Ray workers exit. Loads one rank's two shards into CPU RAM
at a time (~14 GiB for Qwen2.5-3B FP32 shards). Inputs are trusted local files.
"""
import argparse
import gc
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor-dir', type=Path, required=True,
                        help='.../outputs/<run_id>/actor')
    parser.add_argument('--steps', nargs=2, type=int, default=(1, 2))
    parser.add_argument('--world-size', type=int, default=2)
    args = parser.parse_args()

    snapshots = [args.actor_dir / f'snapshot_step_{step}' for step in args.steps]
    for step, path in zip(args.steps, snapshots):
        manifest_path = path / 'snapshot_manifest.json'
        assert manifest_path.is_file(), f'Missing completed manifest: {manifest_path}'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['logical_step'] == step, manifest
        assert manifest['world_size'] == args.world_size, manifest
        assert manifest['optimizer_preserved_in_running_process'] is True, manifest
        assert manifest['optimizer_state_saved'] is False, manifest
        for rank in range(args.world_size):
            shard = path / f'model_world_size_{args.world_size}_rank_{rank}.pt'
            assert shard.is_file() and shard.stat().st_size > 100_000_000, f'Missing/short shard: {shard}'
            print('SNAPSHOT_FILE_OK', 'step=', step, 'rank=', rank,
                  'bytes=', shard.stat().st_size, flush=True)
        assert not list(path.glob('*.incomplete')), f'Incomplete files: {path}'
    print('SNAPSHOT_FILE_AUDIT_OK', flush=True)

    import torch
    changed_total = 0
    tested_total = 0
    for rank in range(args.world_size):
        p1 = snapshots[0] / f'model_world_size_{args.world_size}_rank_{rank}.pt'
        p2 = snapshots[1] / f'model_world_size_{args.world_size}_rank_{rank}.pt'
        print(f'LOADING_RANK_{rank}_SNAPSHOT_1', flush=True)
        first = torch.load(p1, map_location='cpu', weights_only=False)
        print(f'LOADING_RANK_{rank}_SNAPSHOT_2', flush=True)
        second = torch.load(p2, map_location='cpu', weights_only=False)
        assert set(first) == set(second), f'Parameter key mismatch on rank {rank}'
        changed = []
        tested = 0
        for key, x in first.items():
            y = second[key]
            if hasattr(x, '_local_tensor'):
                x = x._local_tensor
                y = y._local_tensor
            if not torch.is_tensor(x):
                continue
            assert torch.is_tensor(y) and x.shape == y.shape and x.dtype == y.dtype, key
            tested += 1
            if not torch.equal(x, y):
                changed.append(key)
        print('RANK_AUDIT', rank, 'compared_keys=', tested,
              'changed_keys=', len(changed), 'example_changes=', changed[:6], flush=True)
        changed_total += len(changed)
        tested_total += tested
        del first, second
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass

    print('TOTAL_COMPARED_KEYS', tested_total)
    print('TOTAL_CHANGED_KEYS', changed_total)
    if changed_total == 0:
        raise SystemExit('SNAPSHOT_AUDIT_FAIL: no parameter changes detected')
    print('SNAPSHOT_WEIGHT_CHANGE_CONFIRMED')
    print('NOTE: optimizer preserved in running process, NOT stored in snapshots')


if __name__ == '__main__':
    main()
