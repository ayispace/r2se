#!/usr/bin/env python3
"""R2-Searcher: experimental complete FSDP/AdamW/cursor checkpoint extension.

Assumes prior NONDESTRUCTIVE snapshot + V3 best selection patches are installed.
Performs strict preflight before modifying any file. NEVER clears optimizer state.
Requires two-GPU V100 environment with optimizer_offload=true, same world size.
NOT PROVEN until save->exit->restart->load->continue test passes.
"""
from pathlib import Path
import os
import shutil
import subprocess

wp = Path('verl/workers/fsdp_workers.py')
tp = Path('verl/trainer/ppo/ray_trainer_ts.py')
source_script = Path('experiments/r2_tree_rl_sft_val_best_smoke.sh')
a_script = Path('experiments/r2_full_resume_phaseA.sh')
b_script = Path('experiments/r2_full_resume_phaseB.sh')
for p in (wp, tp, source_script):
    if not p.is_file():
        raise FileNotFoundError(p)
w = wp.read_text()
t = tp.read_text()
s = source_script.read_text()
marker = 'R2_FULL_RESUME_EXPERIMENT_V1'
if marker in w or marker in t:
    raise RuntimeError('Resume patch already partly installed. Inspect manually; refuse duplicate.')
if a_script.exists() or b_script.exists():
    raise RuntimeError('Resume test scripts already exist; refuse overwrite')
required_worker = [
    'def snapshot_model_only_shards(',
    'SNAPSHOT_OPTIMIZER_PRESERVED',
    'self.checkpoint_manager',
    '    @register(dispatch_mode=Dispatch.ONE_TO_ALL)\n    def load_checkpoint(',
]
for pat in required_worker:
    if pat not in w:
        raise RuntimeError(f'Worker incompatible: missing {pat}')
required_trainer = [
    'R2_SNAPSHOT_BEST_SELECTION_V3',
    'SMOKE_NONDESTRUCTIVE_SNAPSHOT_DONE',
    '                logger.log(data=metrics, step=self.global_steps)\n',
    '        self.global_steps = 0\n',
    '        # we start from step 1\n        self.global_steps += 1',
    '        steps_to_skip = self.global_steps\n',
    '            for batch_dict in self.train_dataloader:',
    '            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps',
    'shuffle=self.config.data.shuffle_train_dataloader,',
]
for pat in required_trainer:
    if t.count(pat) != 1:
        raise RuntimeError(f'Trainer structure mismatch: {pat!r} count={t.count(pat)}')
for pat in ['+trainer.smoke_snapshot_every=1', '+trainer.smoke_best_snapshot=true',
            '+trainer.smoke_skip_saving=true', 'trainer.test_freq=1',
            'trainer.total_training_steps=3', 'trainer.save_freq=-1']:
    if s.count(pat) != 1:
        raise RuntimeError(f'Test shell mismatch {pat!r}; count={s.count(pat)}')
if '+trainer.smoke_export_model_only=true' in s:
    raise RuntimeError('Terminal model export must not run in resume training')

# Save just optimizer/scheduler/RNG: model shards already saved by the proven
# snapshot_model_only_shards() method. Both exports represent the SAME step.
worker_methods = '''    # R2_FULL_RESUME_EXPERIMENT_V1
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_full_resume_optimizer(self, local_path, logical_step):
        """Save CPU-offloaded AdamW, LR scheduler, per-rank RNG.

        NEVER clears optimizer. Serial per-rank disk writes limit peak RAM.
        Model state should already be saved by snapshot_model_only_shards().
        """
        import os
        import json
        import torch
        import torch.distributed as dist
        assert self._is_actor
        assert self._is_offload_optimizer, 'Only tested with optimizer_offload=true'
        assert len(self.actor_optimizer.state) > 0, 'AdamW state uninitialized'
        assert os.path.isfile(os.path.join(local_path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt'))
        optim_before = len(self.actor_optimizer.state)
        os.makedirs(local_path, exist_ok=True)
        dist.barrier()
        for rank in range(self.world_size):
            if self.rank == rank:
                filename = os.path.join(local_path, f'optim_world_size_{self.world_size}_rank_{rank}.pt')
                tmp = filename + '.incomplete'
                print(f'[rank-{rank}] FULL_RESUME_OPTIM_SAVING {filename}', flush=True)
                torch.save(self.actor_optimizer.state_dict(), tmp)
                os.replace(tmp, filename)
                del tmp
                print(f'[rank-{rank}] FULL_RESUME_OPTIM_SAVED {filename}', flush=True)
            dist.barrier()
        for rank in range(self.world_size):
            if self.rank == rank:
                extra = {
                    'logical_step': int(logical_step),
                    'scheduler': self.actor_lr_scheduler.state_dict(),
                    'rng': self.checkpoint_manager.get_rng_state(),
                    'world_size': self.world_size,
                }
                filename = os.path.join(local_path, f'extra_state_world_size_{self.world_size}_rank_{rank}.pt')
                tmp = filename + '.incomplete'
                torch.save(extra, tmp)
                os.replace(tmp, filename)
                print(f'[rank-{rank}] FULL_RESUME_EXTRA_SAVED {filename}', flush=True)
            dist.barrier()
        assert len(self.actor_optimizer.state) == optim_before
        if self.rank == 0:
            marker_file = os.path.join(local_path, 'worker_resume_complete.json')
            temp_file = marker_file + '.incomplete'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump({'logical_step': int(logical_step), 'world_size': self.world_size,
                           'format': 'model_snapshot_plus_rank_optimizer_scheduler_rng'}, f, indent=2)
            os.replace(temp_file, marker_file)
        dist.barrier()
        print(f'[rank-{self.rank}] FULL_RESUME_OPTIMIZER_PRESERVED step={logical_step}', flush=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_full_resume_checkpoint(self, path):
        """Load model, AdamW, LR scheduler, per-rank RNG; fixed world_size."""
        import os
        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
            StateDictType, ShardedStateDictConfig)
        assert self._is_actor and not self._is_offload_param
        assert self._is_offload_optimizer, 'Only tested with optimizer_offload=true'
        assert os.path.isfile(os.path.join(path, 'resume_manifest.json'))
        # Serialize disk reads, then ALL ranks participate in FSDP.load_state_dict.
        model_sd = None
        for rank in range(self.world_size):
            if self.rank == rank:
                f = os.path.join(path, f'model_world_size_{self.world_size}_rank_{rank}.pt')
                model_sd = torch.load(f, map_location='cpu', weights_only=False)
            dist.barrier()
        assert model_sd is not None
        with FSDP.state_dict_type(self.actor_module_fsdp, StateDictType.SHARDED_STATE_DICT,
                                  ShardedStateDictConfig(offload_to_cpu=True)):
            self.actor_module_fsdp.load_state_dict(model_sd, strict=True)
        del model_sd
        dist.barrier()
        opt_sd = None
        extra = None
        for rank in range(self.world_size):
            if self.rank == rank:
                opt_sd = torch.load(os.path.join(path, f'optim_world_size_{self.world_size}_rank_{rank}.pt'),
                                    map_location='cpu', weights_only=False)
                extra = torch.load(os.path.join(path, f'extra_state_world_size_{self.world_size}_rank_{rank}.pt'),
                                   map_location='cpu', weights_only=False)
            dist.barrier()
        # Restore AdamW state WITHOUT optimizer.load_state_dict(), which can
        # opportunistically move 1st/2nd moments onto CUDA and cause V100 OOM.
        # Here tensors remain on CPU until the existing update_actor() loads them.
        assert isinstance(self.actor_optimizer, torch.optim.AdamW)
        saved_groups = opt_sd['param_groups']
        current_groups = self.actor_optimizer.param_groups
        assert len(saved_groups) == len(current_groups)
        index_to_param = {}
        for old_group, cur_group in zip(saved_groups, current_groups):
            old_ids, cur_params = old_group['params'], cur_group['params']
            if len(old_ids) != len(cur_params):
                raise RuntimeError('RESUME_OPTIM_PARAM_LAYOUT_MISMATCH')
            for param_id, param in zip(old_ids, cur_params):
                index_to_param[param_id] = param
            for k, v in old_group.items():
                if k != 'params':
                    cur_group[k] = v
        self.actor_optimizer.state.clear()
        for param_id, state in opt_sd['state'].items():
            if param_id not in index_to_param:
                raise RuntimeError(f'Unknown saved optimizer param id {param_id}')
            self.actor_optimizer.state[index_to_param[param_id]] = state
        del opt_sd
        if not self.actor_optimizer.state:
            raise RuntimeError('RESTORED_OPTIMIZER_HAS_NO_STATE')
        self.actor_lr_scheduler.load_state_dict(extra['scheduler'])
        self.checkpoint_manager.load_rng_state(extra['rng'])
        dist.barrier()
        print(f'[rank-{self.rank}] FULL_RESUME_WORKER_LOADED step={extra["logical_step"]}', flush=True)

'''
# The worker file also contains CriticWorker.load_checkpoint(); only modify
# ActorRolloutRefWorker to avoid relying on a global uniqueness assumption.
worker_anchor = '    @register(dispatch_mode=Dispatch.ONE_TO_ALL)\n    def load_checkpoint('
actor_start = w.index('class ActorRolloutRefWorker(')
critic_start = w.index('\nclass CriticWorker(', actor_start)
actor_part = w[actor_start:critic_start]
if actor_part.count(worker_anchor) != 1:
    raise RuntimeError(
        f'ActorRolloutRefWorker load_checkpoint anchor count={actor_part.count(worker_anchor)}, expected 1'
    )
w_new = (w[:actor_start] + actor_part.replace(worker_anchor, worker_methods + worker_anchor, 1)
         + w[critic_start:])

# Deterministic data order with a dedicated sampler generator. Leave original
# dataloader unchanged whenever opt-in full_resume_enabled=false.
train_loader_start = '        self.train_dataloader = DataLoader(dataset=self.train_dataset,'
assert t.count(train_loader_start) == 1
t_new = t.replace(train_loader_start, '''        # R2_FULL_RESUME_EXPERIMENT_V1: replayable train sampler.
        resume_seed = int(self.config.trainer.get('full_resume_data_seed', 20261012))
        resume_generator = (torch.Generator().manual_seed(resume_seed)
                            if self.config.trainer.get('full_resume_enabled', False) else None)
''' + train_loader_start, 1)
t_new = t_new.replace('shuffle=self.config.data.shuffle_train_dataloader,',
    'shuffle=self.config.data.shuffle_train_dataloader,\n                                           generator=resume_generator,', 1)

# Load worker and controller states, but restore DRIVER RNG only AFTER replayed
# old batches have been skipped (DataLoader can advance RNG on iter creation).
head = '        self.global_steps = 0\n'
assert t_new.count(head) == 1
resume_init = '''        # R2_FULL_RESUME_EXPERIMENT_V1: opt-in resume preflight.
        resume_completed = 0
        resume_driver_rng = None
        resume_prior_scores = (None, None)
        resume_path = self.config.trainer.get('full_resume_from', None)
        if resume_path is not None and str(resume_path).lower() not in ('null', 'none', ''):
            import json as _rjson
            import torch as _rtorch
            import os as _ros
            path = _ros.path.abspath(str(resume_path))
            manifest_file = _ros.path.join(path, 'resume_manifest.json')
            with open(manifest_file, 'r', encoding='utf-8') as f:
                resume_manifest = _rjson.load(f)
            if resume_manifest['signature'] != self._r2_full_resume_signature():
                raise RuntimeError('RESUME_CONFIG_MISMATCH: same model/datasets/batch/world_size required')
            driver = _rtorch.load(_ros.path.join(path, 'driver_state.pt'),
                                  map_location='cpu', weights_only=False)
            resume_completed = int(driver['logical_step'])
            assert resume_completed == resume_manifest['logical_step']
            self.actor_rollout_wg.load_full_resume_checkpoint(path)
            resume_driver_rng = driver['rng']
            if 'kl_controller_value' in driver and hasattr(self.kl_ctrl, 'value'):
                self.kl_ctrl.value = float(driver['kl_controller_value'])
            resume_prior_scores = (driver.get('best_reward'), driver.get('best_val'))
            # Carry the previous Best pointer into the new experiment directory.
            previous_best = _ros.path.join(_ros.path.dirname(path), 'best_snapshot.json')
            if _ros.path.isfile(previous_best):
                current_best = _ros.path.join(self.config.trainer.default_local_dir,
                                             'actor', 'best_snapshot.json')
                _ros.makedirs(_ros.path.dirname(current_best), exist_ok=True)
                import shutil as _rshutil
                _rshutil.copy2(previous_best, current_best)
            print(f'FULL_RESUME_DRIVER_LOADED completed={resume_completed}', flush=True)
'''
t_new = t_new.replace(head, head + resume_init, 1)

# Preserve trainer-side best trackers across process restart, too.
best_anchor = "        self.best_reward = float('-inf')\n        self.best_val = 0.0"
if t_new.count(best_anchor) != 1:
    raise RuntimeError('Expected best_reward/best_val init anchor exactly once')
t_new = t_new.replace(best_anchor, best_anchor + '''
        if resume_completed:
            if resume_prior_scores[0] is not None:
                self.best_reward = float(resume_prior_scores[0])
            if resume_prior_scores[1] is not None:
                self.best_val = float(resume_prior_scores[1])
''', 1)

next_step = '        # we start from step 1\n        self.global_steps += 1'
assert t_new.count(next_step) == 1
t_new = t_new.replace(next_step, next_step + '''
        if resume_completed:
            self.global_steps = resume_completed + 1
''', 1)

# Loop: skip exactly completed training batches. The replayable generator
# recreates order; skip must happen before any expensive RL operations.
loop_anchor = '        steps_to_skip = self.global_steps\n'
assert t_new.count(loop_anchor) == 1
t_new = t_new.replace(loop_anchor, loop_anchor + '        full_resume_seen_batches = 0\n', 1)
batch_anchor = '            for batch_dict in self.train_dataloader:\n'
assert t_new.count(batch_anchor) == 1
t_new = t_new.replace(batch_anchor, batch_anchor + '''                if full_resume_seen_batches < resume_completed:
                    full_resume_seen_batches += 1
                    continue
                if resume_driver_rng is not None:
                    _rtorch.set_rng_state(resume_driver_rng['torch'])
                    import numpy as _rnp
                    import random as _rrandom
                    _rnp.random.set_state(resume_driver_rng['numpy'])
                    _rrandom.setstate(resume_driver_rng['python'])
                    resume_driver_rng = None
                assert self.global_steps == full_resume_seen_batches + 1, (
                    self.global_steps, full_resume_seen_batches)
                full_resume_seen_batches += 1
''', 1)

# Checkpoint occurs AFTER best selection and snapshot, BEFORE logger.log.
logger_anchor = '                logger.log(data=metrics, step=self.global_steps)\n'
assert t_new.count(logger_anchor) == 1
ckpt_hook = '''                # R2_FULL_RESUME_EXPERIMENT_V1: full resume files after snapshot.
                if self.config.trainer.get('full_resume_enabled', False):
                    period = int(self.config.trainer.get('full_resume_every', 0))
                    snapshot_period = int(self.config.trainer.get('smoke_snapshot_every', 0))
                    if period < 1 or snapshot_period < 1 or period % snapshot_period:
                        raise RuntimeError('full_resume_every must be multiple of snapshot_every')
                    if self.global_steps % period == 0:
                        import json as _rjson
                        import os as _ros
                        import torch as _rtorch
                        import numpy as _rnp
                        import random as _rrandom
                        directory = _ros.path.join(self.config.trainer.default_local_dir,
                            'actor', f'snapshot_step_{self.global_steps}')
                        assert _ros.path.isfile(_ros.path.join(directory, 'snapshot_manifest.json'))
                        self.actor_rollout_wg.save_full_resume_optimizer(directory, self.global_steps)
                        driver = {
                            'logical_step': int(self.global_steps),
                            'best_reward': float(self.best_reward),
                            'best_val': float(self.best_val),
                            'kl_controller_value': (float(self.kl_ctrl.value)
                                                    if hasattr(self.kl_ctrl, 'value') else None),
                            'rng': {'torch': _rtorch.get_rng_state(),
                                    'numpy': _rnp.random.get_state(),
                                    'python': _rrandom.getstate()},
                        }
                        driver_final = _ros.path.join(directory, 'driver_state.pt')
                        driver_tmp = driver_final + '.incomplete'
                        _rtorch.save(driver, driver_tmp)
                        _ros.replace(driver_tmp, driver_final)
                        manifest = {
                            'logical_step': int(self.global_steps),
                            'signature': self._r2_full_resume_signature(),
                            'format_version': 'r2_full_resume_experimental_v1',
                            'model_only_snapshot_dir': directory,
                        }
                        manifest_file = _ros.path.join(directory, 'resume_manifest.json')
                        manifest_tmp = manifest_file + '.incomplete'
                        with open(manifest_tmp, 'w', encoding='utf-8') as f:
                            _rjson.dump(manifest, f, indent=2, ensure_ascii=False)
                        _ros.replace(manifest_tmp, manifest_file)
                        print(f'FULL_RESUME_CHECKPOINT_READY: {directory}', flush=True)
'''
t_new = t_new.replace(logger_anchor, ckpt_hook + logger_anchor, 1)

# A fingerprint catches most unintended changes to model/data/optimizer/sampler.
# It is NOT a substitute for sha256-verified formal dataset manifests.
signature_method = '''    # R2_FULL_RESUME_EXPERIMENT_V1
    @staticmethod
    def _r2_file_stat(path):
        import os
        # Only local path support in this experimental resume implementation.
        stat = os.stat(path)
        return {'size': int(stat.st_size), 'mtime_ns': int(stat.st_mtime_ns)}

    def _r2_full_resume_signature(self):
        import os
        cfg = self.config
        return {
            'model': os.path.abspath(str(cfg.actor_rollout_ref.model.path)),
            'train_files': str(cfg.data.train_files),
            'val_files': str(cfg.data.val_files),
            'train_file_stat': self._r2_file_stat(str(cfg.data.train_files)),
            'val_file_stat': self._r2_file_stat(str(cfg.data.val_files)),
            'val_data_num': str(cfg.data.val_data_num),
            'max_prompt_length': int(cfg.data.max_prompt_length),
            'total_epochs': int(cfg.trainer.total_epochs),
            'train_batch_size': int(cfg.data.train_batch_size),
            'train_data_num': str(cfg.data.train_data_num),
            'shuffle': bool(cfg.data.shuffle_train_dataloader),
            'data_seed': int(cfg.trainer.get('full_resume_data_seed', 20261012)),
            'world_size': int(cfg.trainer.n_gpus_per_node * cfg.trainer.nnodes),
            'learning_rate': float(cfg.actor_rollout_ref.actor.optim.lr),
            'total_training_steps': int(self.total_training_steps),
            'ts_m': int(cfg.actor_rollout_ref.rollout.ts_m),
            'ts_n': int(cfg.actor_rollout_ref.rollout.ts_n),
            'ts_l': int(cfg.actor_rollout_ref.rollout.ts_l),
            'ts_k': int(cfg.actor_rollout_ref.rollout.ts_k),
        }

'''
init_workers_anchor = '    def init_workers(self):\n'
assert t_new.count(init_workers_anchor) == 1
t_new = t_new.replace(init_workers_anchor, signature_method + init_workers_anchor, 1)

# No mutation until all Python and Shell preflight checks pass.
script_a = s
lines = script_a.splitlines(keepends=True)
matches = [i for i,x in enumerate(lines) if 'trainer.save_freq=-1' in x]
assert len(matches) == 1
idx = matches[0]
indent = lines[idx][:len(lines[idx]) - len(lines[idx].lstrip())]
flags = [
    '+trainer.full_resume_enabled=true',
    '+trainer.full_resume_every=1',
    '+trainer.full_resume_data_seed=20261012',
    '+trainer.full_resume_from=null',
]
for j, flag in enumerate(flags):
    lines.insert(idx+j, indent + "'" + flag + "' " + chr(92) + '\n')
script_a = ''.join(lines)
script_b = script_a.replace("'+trainer.full_resume_from=null'", '+trainer.full_resume_from="$RESUME_FROM"', 1)
assert script_b != script_a
script_b = script_b.replace('set -euo pipefail',
   'set -euo pipefail\n: "${RESUME_FROM:?Set RESUME_FROM to a complete snapshot_step_1 directory}"', 1)
for data, path in [(script_a,a_script),(script_b,b_script)]:
    compile(data, str(path), 'exec') if False else None
    check_path = path.with_suffix('.sh.syntax_check')
    try:
        check_path.write_text(data)
        subprocess.run(['bash', '-n', str(check_path)], check=True)
    finally:
        check_path.unlink(missing_ok=True)
compile(w_new, str(wp), 'exec')
compile(t_new, str(tp), 'exec')

for path in (wp,tp):
    backup = path.with_name(path.name+'.before_full_resume_experimental_v1.bak')
    if backup.exists():
        raise RuntimeError(f'Backup exists, refusing overwrite {backup}')
for path in (wp,tp):
    shutil.copy2(path, str(path)+'.before_full_resume_experimental_v1.bak')
    temp = path.with_name(path.name+'.full_resume.incomplete')
    temp.write_text(w_new if path == wp else t_new)
    os.replace(temp, path)
for path,contents in [(a_script,script_a),(b_script,script_b)]:
    path.write_text(contents)
    path.chmod(source_script.stat().st_mode)
print('R2_FULL_RESUME_EXPERIMENT_PATCH_OK')
print('PhaseA:',a_script)
print('PhaseB:',b_script)
print('WARNING: experimental; must verify OOM, reload, and continuous-vs-resume results')
