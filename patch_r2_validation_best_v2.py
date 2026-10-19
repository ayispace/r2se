#!/usr/bin/env python3
"""R²-Searcher V100: install a SMOKE-only best-snapshot pointer.

For the user branch where sparse-source _validate aggregation is ALREADY fixed.
Never re-writes the aggregation or the FSDP worker. Never changes AdamW.
Idempotent when both trainer and generated shell script already contain V3.
The smoke metric is chosen from actual validation output, with provenance.
"""
from pathlib import Path
import os
import re
import shutil
import subprocess

trainer_path = Path('verl/trainer/ppo/ray_trainer_ts.py')
worker_path = Path('verl/workers/fsdp_workers.py')
source_script = Path('experiments/r2_tree_rl_sft_2updates_snapshot.sh')
target_script = Path('experiments/r2_tree_rl_sft_val_best_smoke.sh')
marker = 'R2_SNAPSHOT_BEST_SELECTION_V3'

for path in (trainer_path, worker_path, source_script):
    if not path.is_file():
        raise FileNotFoundError(f'Required file missing: {path}')
trainer = trainer_path.read_text(encoding='utf-8')
worker = worker_path.read_text(encoding='utf-8')
script = source_script.read_text(encoding='utf-8')

if marker in trainer:
    if target_script.is_file() and '+trainer.smoke_best_snapshot=true' in target_script.read_text():
        print('VALIDATION_BEST_V3_ALREADY_INSTALLED')
        raise SystemExit(0)
    raise RuntimeError('V3 marker exists, but smoke script is absent/incomplete: inspect manually')
if 'R2_SNAPSHOT_BEST_SELECTION_V1' in trainer or 'R2_SNAPSHOT_BEST_SELECTION_V2' in trainer:
    raise RuntimeError('Another best-selection patch already installed; refusing duplicate installation')
if target_script.exists():
    raise RuntimeError(f'Refusing to overwrite existing smoke script: {target_script}')
if 'SMOKE_NONDESTRUCTIVE_SNAPSHOT_DONE' not in trainer:
    raise RuntimeError('Non-destructive periodic snapshot trainer hook is missing')
if 'SNAPSHOT_OPTIMIZER_PRESERVED' not in worker:
    raise RuntimeError('Non-destructive snapshot worker hook is missing')
if 'smoke_skip_saving' not in trainer:
    raise RuntimeError('Missing earlier smoke_skip_saving guard')

# Confirm that sparse-dataset aggregation has already been made safe. The V3
# patch does not try to rewrite this custom function again.
val_start = re.search(r'(?m)^    def _validate\(self\):\s*$', trainer)
val_end = re.search(r'(?m)^    def init_workers\(self\):\s*$', trainer)
if not val_start or not val_end or val_start.end() >= val_end.start():
    raise RuntimeError('Could not identify _validate() bounds')
validate_body = trainer[val_start.end():val_end.start()]

sparse_safe = (
    ("if f'{root}/{ds}' in metric_dict" in validate_body and "'_observed'" in validate_body)
    or ("if f'{root}/{ds}' in metric_dict" in validate_body and "group + '_observed'" in validate_body)
    or ('present_all' in validate_body and 'if f\'{root}/{ds}\' in metric_dict' in validate_body)
    or ('R2_VAL_PRESENT_DATA_SOURCES_V2' in validate_body)
)
if not sparse_safe:
    print('=== _validate() final 48 lines (diagnostics only) ===')
    print('\n'.join(validate_body.splitlines()[-48:]))
    raise RuntimeError('Sparse-source validation aggregation not verified. V3 will not overwrite it.')
print('VALIDATION_AGGREGATION_ALREADY_SPARSE_SAFE')

# Insert AFTER snapshot logic and BEFORE the unique per-step logger call.
anchor = '                logger.log(data=metrics, step=self.global_steps)\n'
if trainer.count(anchor) != 1:
    raise RuntimeError(f'Expected one per-step logger anchor, found {trainer.count(anchor)}')

selection = '''                # R2_SNAPSHOT_BEST_SELECTION_V3
                # SMOKE-ONLY: use actual metrics and record metric provenance.
                # Never use training reward as the model-selection metric.
                if self.config.trainer.get('smoke_best_snapshot', False):
                    import json
                    import math
                    snapshot_every = int(self.config.trainer.get('smoke_snapshot_every', 0))
                    if snapshot_every <= 0:
                        raise RuntimeError('Best-selection requires smoke_snapshot_every > 0')
                    if self.global_steps % snapshot_every == 0:
                        requested = str(self.config.trainer.get('smoke_best_metric', 'auto'))
                        if requested == 'auto':
                            # This project labels incomplete benchmark aggregates
                            # as /mean_observed, not /mean. Keep the distinction.
                            preference = (
                                'val/test_score/mean',
                                'val/test_score/mean_observed',
                            )
                            chosen = next((name for name in preference if name in metrics), None)
                            if chosen is None:
                                source_candidates = sorted(
                                    name for name in metrics
                                    if name.startswith('val/test_score/')
                                    and name.split('/')[-1] not in {
                                        'mean', 'mean_observed', 'single',
                                        'single_observed', 'multi', 'multi_observed',
                                    }
                                )
                                if len(source_candidates) != 1:
                                    raise RuntimeError(
                                        'Cannot choose unique validation F1 metric; '
                                        f'candidates={source_candidates}; '
                                        f'available={[k for k in metrics if k.startswith("val/test_score/")]}'
                                    )
                                chosen = source_candidates[0]
                            metric_name = chosen
                        else:
                            metric_name = requested
                            if metric_name not in metrics:
                                raise RuntimeError(
                                    f'Requested validation metric {metric_name!r} missing; '
                                    f'available={[k for k in metrics if k.startswith("val/test_score/")]}'
                                )
                        score = float(metrics[metric_name])
                        if not math.isfinite(score):
                            raise RuntimeError(f'Non-finite validation metric {metric_name}: {score}')
                        actor_dir = os.path.join(self.config.trainer.default_local_dir, 'actor')
                        candidate = os.path.join(actor_dir, f'snapshot_step_{self.global_steps}')
                        manifest = os.path.join(candidate, 'snapshot_manifest.json')
                        if not os.path.isfile(manifest):
                            raise RuntimeError(f'Snapshot missing or incomplete: {manifest}')
                        best_file = os.path.join(actor_dir, 'best_snapshot.json')
                        previous = None
                        if os.path.isfile(best_file):
                            with open(best_file, 'r', encoding='utf-8') as f:
                                previous = json.load(f)
                        improved = previous is None or score > float(previous['score'])
                        if improved:
                            record = {
                                'logical_step': int(self.global_steps),
                                'metric': metric_name,
                                'score': score,
                                'snapshot_dir': os.path.abspath(candidate),
                                'val_files': str(self.config.data.val_files),
                                'val_data_num': self.config.data.val_data_num,
                                'model_only': True,
                                'selection': 'smoke_not_paper_result',
                                'metric_scope': (
                                    'incomplete_dataset_observed' if metric_name.endswith('_observed')
                                    else 'source_or_full_dataset'
                                ),
                            }
                            tmp = best_file + '.incomplete'
                            with open(tmp, 'w', encoding='utf-8') as f:
                                json.dump(record, f, indent=2, ensure_ascii=False)
                            os.replace(tmp, best_file)
                        print(
                            f'SMOKE_BEST_SELECTION step={self.global_steps} '
                            f'metric={metric_name} score={score:.6f} '
                            f'improved={improved} best_file={best_file}',
                            flush=True,
                        )

'''
new_trainer = trainer.replace(anchor, selection + anchor, 1)

# Generate the smoke command from the known-good two-snapshot script.
if script.count('trainer.test_freq=-1') != 1:
    raise RuntimeError('Expected exactly one trainer.test_freq=-1 in source script')
if script.count('+trainer.smoke_snapshot_every=1') != 1:
    raise RuntimeError('Expected snapshot_every=1 in source script')
if 'trainer.total_training_steps=3' not in script:
    raise RuntimeError('Two-update smoke script is not the expected training configuration')
if '+trainer.smoke_skip_saving=true' not in script:
    raise RuntimeError('Do not enable unconditional original checkpoint saving')
if '+trainer.smoke_best_snapshot=' in script:
    raise RuntimeError('Best flag already set in source script')
new_script = script.replace('trainer.test_freq=-1', 'trainer.test_freq=1', 1)
lines = new_script.splitlines(keepends=True)
matches = [i for i, line in enumerate(lines) if 'trainer.test_freq=1' in line]
if len(matches) != 1:
    raise RuntimeError('Ambiguous trainer.test_freq argument after replacement')
idx = matches[0]
line = lines[idx]
if not line.rstrip().endswith('\\'):
    raise RuntimeError(f'Expected bash continuation at test_freq line: {line!r}')
indent = line[:len(line) - len(line.lstrip())]
lines.insert(idx, indent + "'+trainer.smoke_best_snapshot=true' " + chr(92) + '\n')
lines.insert(idx + 1, indent + "'+trainer.smoke_best_metric=auto' " + chr(92) + '\n')
new_script = ''.join(lines)

# Preflight: compile Python and syntax-check shell before changing source files.
compile(new_trainer, str(trainer_path), 'exec')
tmp_script = target_script.with_suffix('.sh.check')
try:
    tmp_script.write_text(new_script, encoding='utf-8')
    subprocess.run(['bash', '-n', str(tmp_script)], check=True)
finally:
    tmp_script.unlink(missing_ok=True)

backup = trainer_path.with_name(trainer_path.name + '.before_validation_best_v3.bak')
if backup.exists():
    raise RuntimeError(f'Backup already exists; refusing overwrite: {backup}')
shutil.copy2(trainer_path, backup)
tmp_trainer = trainer_path.with_name(trainer_path.name + '.v3.incomplete')
try:
    tmp_trainer.write_text(new_trainer, encoding='utf-8')
    os.replace(tmp_trainer, trainer_path)
    target_script.write_text(new_script, encoding='utf-8')
    target_script.chmod(source_script.stat().st_mode)
finally:
    tmp_trainer.unlink(missing_ok=True)

print('VALIDATION_BEST_V3_OK')
print('New smoke script:', target_script)
print('NOTE: one-sample val_files can test the mechanics, not scientifically select Best')
