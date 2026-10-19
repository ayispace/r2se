#!/usr/bin/env python3
"""Deterministic, leakage-audited R2-Searcher RL train/validation split.

Inputs must be RLHFDataset-compatible parquet. This is a train-only split:
DO NOT pass a test or benchmark held-out file here. A prompt group never
straddles train and validation. Does not mutate input files.
"""
import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def canonical(x):
    def normalize(t):
        if isinstance(t, str):
            return ' '.join(t.split())
        if isinstance(t, dict):
            return {str(k): normalize(v) for k, v in sorted(t.items())}
        if isinstance(t, (tuple, list)):
            return [normalize(v) for v in t]
        if hasattr(t, 'tolist'):
            return normalize(t.tolist())
        if t is None:
            return None
        return t if isinstance(t, (int, float, bool)) else str(t)
    return json.dumps(normalize(x), ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(x):
    return hashlib.sha256(x.encode('utf-8')).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def source_name(v):
    return 'unknown' if pd.isna(v) else str(v)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='training parquet, NOT a test/benchmark holdout')
    p.add_argument('--output-dir', default='data/formal_v1', type=Path)
    p.add_argument('--val-ratio', type=float, default=0.05)
    p.add_argument('--val-max', type=int, default=256)
    p.add_argument('--seed', type=int, default=20261012)
    p.add_argument('--prompt-key', default='prompt')
    p.add_argument('--check-against', nargs='*', default=[], type=Path,
                   help='Additional parquet files, e.g. test; abort if any prompt overlaps validation')
    a = p.parse_args()
    if not (0 < a.val_ratio < 0.5) or a.val_max < 1:
        p.error('Require 0 < --val-ratio < 0.5 and --val-max >= 1')
    if not a.input.is_file():
        p.error(f'File missing: {a.input}')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error(f'Output directory is nonempty: {a.output_dir} (refuse overwrite)')
    df = pd.read_parquet(a.input).reset_index(drop=True)
    needed = {a.prompt_key, 'reward_model', 'data_source'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Input is not RLHFDataset-compatible; missing {sorted(needed - set(df.columns))}')
    if len(df) < 4:
        raise ValueError('Too few records for meaningful train/validation split')
    # Group by the entire normalized prompt. A group remains exclusively on one side.
    # This is an exact-normalized deduplication check, not a semantic paraphrase detector.
    prompt_hashes = [sha(canonical(x)) for x in df[a.prompt_key]]
    groups = defaultdict(list)
    for i, h in enumerate(prompt_hashes):
        groups[h].append(i)
    source_groups = defaultdict(list)
    for h, indices in groups.items():
        srcs = [source_name(df.iloc[i]['data_source']) for i in indices]
        src = Counter(srcs).most_common(1)[0][0]
        source_groups[src].append(h)

    # Deterministic split, no unseeded random operations.
    # Allocate a capped validation budget proportional to group counts.
    total_groups = len(groups)
    target = min(a.val_max, max(1, round(total_groups * a.val_ratio)))
    eligible = {src: hs for src, hs in source_groups.items() if len(hs) >= 2}
    if not eligible:
        raise ValueError('Every source has fewer than 2 unique prompt groups')
    if target < len(eligible):
        raise ValueError(f'val-max/ratio produces {target} groups, but {len(eligible)} data sources need coverage; increase --val-max or ratio')
    assignments = {src: 1 for src in eligible}
    remaining = target - len(eligible)
    # Largest-remainder-style apportionment, with one group left in training for each source.
    while remaining:
        candidates = [src for src, hs in eligible.items() if assignments[src] < len(hs) - 1]
        if not candidates:
            break
        chosen = max(candidates, key=lambda s: (
            len(eligible[s]) / max(1, assignments[s]),
            sha(f'{a.seed}:{s}')
        ))
        assignments[chosen] += 1
        remaining -= 1
    selected = set()
    for src, hs in eligible.items():
        ordered = sorted(hs, key=lambda h: sha(f'{a.seed}:{h}'))
        selected.update(ordered[:assignments[src]])
    train_indices = [i for i, h in enumerate(prompt_hashes) if h not in selected]
    val_indices = [i for i, h in enumerate(prompt_hashes) if h in selected]
    if not train_indices or not val_indices:
        raise RuntimeError('Split is empty')
    train_hash = set(prompt_hashes[i] for i in train_indices)
    val_hash = set(prompt_hashes[i] for i in val_indices)
    assert train_hash.isdisjoint(val_hash)

    # A test/held-out overlap veto. Never use external test rows to create validation.
    overlaps = {}
    for path in a.check_against:
        other = pd.read_parquet(path)
        if a.prompt_key not in other.columns:
            raise ValueError(f'Check file {path} has no {a.prompt_key} column')
        check = set(sha(canonical(x)) for x in other[a.prompt_key])
        count = len(val_hash & check)
        overlaps[str(path)] = count
        if count:
            raise RuntimeError(f'LEAKAGE DETECTED: {count} validation prompts overlap {path}')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    train_path = a.output_dir / 'train.parquet'
    val_path = a.output_dir / 'val.parquet'
    df.iloc[train_indices].reset_index(drop=True).to_parquet(train_path, index=False)
    df.iloc[val_indices].reset_index(drop=True).to_parquet(val_path, index=False)
    # Readback verifies nested prompt/reward objects survived parquet serialization.
    tr = pd.read_parquet(train_path)
    va = pd.read_parquet(val_path)
    if len(tr) != len(train_indices) or len(va) != len(val_indices):
        raise RuntimeError('Parquet roundtrip row counts differ')
    check_tr = {sha(canonical(x)) for x in tr[a.prompt_key]}
    check_va = {sha(canonical(x)) for x in va[a.prompt_key]}
    if check_tr & check_va or check_tr != train_hash or check_va != val_hash:
        raise RuntimeError('Roundtrip validation disjointness failure')
    manifest = {
        'purpose': 'model_selection_not_test',
        'source_train': str(a.input.resolve()),
        'source_train_sha256': hash_file(a.input),
        'seed': a.seed, 'group_key': f'normalized_{a.prompt_key}_sha256',
        'val_ratio_requested': a.val_ratio,
        'val_max_group_count': a.val_max,
        'original_rows': len(df),
        'train_rows': len(tr), 'val_rows': len(va),
        'unique_train_prompt_groups': len(check_tr),
        'unique_val_prompt_groups': len(check_va),
        'train_source_counts': dict(Counter(source_name(x) for x in tr.data_source)),
        'val_source_counts': dict(Counter(source_name(x) for x in va.data_source)),
        'external_overlap_checks': overlaps,
        'train_file_sha256': hash_file(train_path),
        'val_file_sha256': hash_file(val_path),
        'limitations': [
            'Only exact normalized prompt overlap is excluded; paraphrase leakage is not detected.',
            'This check does not establish disjointness from prior SFT training examples.',
            'Models trained on the original full training parquet have already seen the newly held-out RL-val prompts only if used for RL; keep their data provenance separate.',
            'Best selection on validation is not an unbiased final test result.',
        ],
    }
    (a.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('FORMAL_SPLIT_OK')
    print(json.dumps({k: manifest[k] for k in ['original_rows','train_rows','val_rows','train_source_counts','val_source_counts']}, ensure_ascii=False, indent=2))
    print(f"TRAIN_FILE={train_path.resolve()}")
    print(f"VAL_FILE={val_path.resolve()}")
    print(f"MANIFEST={a.output_dir.resolve() / 'manifest.json'}")


if __name__ == '__main__':
    main()
