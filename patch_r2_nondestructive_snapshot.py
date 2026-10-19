#!/usr/bin/env python3
"""R2-Searcher V100: opt-in, NON-DESTRUCTIVE model-only FSDP snapshots.

Use only on the locally patched branch that already has export_model_only_shards
and the smoke_skip_saving guard. The previously successful terminal export is left alone.

Smoke experiment: two parameter updates, save snapshot_step_1 and snapshot_step_2,
without clearing AdamW optimizer states. NOT a resumable checkpoint.
"""
from pathlib import Path
import shutil

worker_path = Path('verl/workers/fsdp_workers.py')
trainer_path = Path('verl/trainer/ppo/ray_trainer_ts.py')
source_script = Path('experiments/r2_tree_rl_sft_3steps.sh')
output_script = Path('experiments/r2_tree_rl_sft_2updates_snapshot.sh')

for p in (worker_path, trainer_path, source_script):
    if not p.is_file():
        raise FileNotFoundError(f'Missing file: {p}')

worker = worker_path.read_text()
trainer = trainer_path.read_text()
script = source_script.read_text()

worker_anchor = (
    '    @register(dispatch_mode=Dispatch.ONE_TO_ALL)\n'
    '    def export_model_only_shards(self, local_path):'
)
trainer_anchor = '                logger.log(data=metrics, step=self.global_steps)\n'

if 'def snapshot_model_only_shards(' in worker or 'SMOKE_NONDESTRUCTIVE_SNAPSHOT_DONE' in trainer:
    raise RuntimeError('Periodic snapshot patch appears already installed; do not apply twice.')
assert worker.count(worker_anchor) == 1, 'Worker export method anchor absent or duplicated'
assert trainer.count(trainer_anchor) == 1, 'Trainer per-step logging anchor absent or duplicated'
assert 'smoke_skip_saving' in trainer, 'Existing smoke_skip_saving protection missing'
assert 'SMOKE_MODEL_ONLY_EXPORT_DONE' in trainer, 'Existing terminal model-only export patch missing'
assert script.count('trainer.total_training_steps=3') == 1, 'Expected 3 training_steps in known-good script'
assert script.count('trainer.total_epochs=3') == 1, 'Expected 3 epochs in known-good script'
assert script.count('+trainer.smoke_skip_saving=true') == 1, 'Existing smoke skip-saving flag missing'
assert script.count('trainer.save_freq=-1') == 1, 'save_freq argument mismatch'
assert not output_script.exists(), f'Target script already exists: {output_script}'

worker_method = '''    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def snapshot_model_only_shards(self, local_path, step):
        """Non-destructive, experimental periodic model-only FSDP snapshot.

        Unlike export_model_only_shards, this method NEVER clears or offloads
        optimizer states. Both ranks enter the state_dict collective before any
        serialization, then save sequentially to reduce peak host RAM.
        Model only; NOT an exact optimizer/RNG/scheduler-resumable checkpoint.
        """
        assert self._is_actor
        assert not self._is_offload_param, 'Tested only with actor param_offload=false'
        import os
        import gc
        import ctypes
        import json
        import torch
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            ShardedStateDictConfig,
        )

        def trim_allocator():
            gc.collect()
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except (OSError, AttributeError):
                pass

        # Model-only saving must not alter optimizer state.
        optim_before = (
            len(self.actor_optimizer.state),
            sum(len(v) for v in self.actor_optimizer.state.values()),
        )
        print(
            f'[rank-{self.rank}] SNAPSHOT_OPTIMIZER_BEFORE '
            f'step={step} states={optim_before}',
            flush=True,
        )
        trim_allocator()
        torch.cuda.empty_cache()
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(
                f'[rank-{self.rank}] SNAPSHOT_HOST_RAM_BEFORE '
                f'used_gib={mem.used / 1024**3:.1f} '
                f'available_gib={mem.available / 1024**3:.1f}',
                flush=True,
            )
        except ImportError:
            pass

        config = ShardedStateDictConfig(offload_to_cpu=False)
        with FSDP.state_dict_type(
            self.actor_module_fsdp,
            StateDictType.SHARDED_STATE_DICT,
            config,
        ):
            state = self.actor_module_fsdp.state_dict()

        os.makedirs(local_path, exist_ok=True)
        final_path = os.path.join(
            local_path,
            f'model_world_size_{self.world_size}_rank_{self.rank}.pt',
        )

        # All ranks have completed the FSDP state_dict collective.
        torch.distributed.barrier()
        for saving_rank in range(self.world_size):
            if self.rank == saving_rank:
                tmp_path = final_path + '.incomplete'
                print(f'[rank-{self.rank}] SNAPSHOT_SAVING {tmp_path}', flush=True)
                torch.save(state, tmp_path)
                os.replace(tmp_path, final_path)
                del state
                trim_allocator()
                print(f'[rank-{self.rank}] SNAPSHOT_SHARD_SAVED {final_path}', flush=True)
            torch.distributed.barrier()

        if self.rank == 0:
            hf_config_dir = os.path.join(local_path, 'huggingface')
            os.makedirs(hf_config_dir, exist_ok=True)
            self.actor_model_config.save_pretrained(hf_config_dir)
            self.tokenizer.save_pretrained(hf_config_dir)
            if self.generation_config is not None:
                self.generation_config.save_pretrained(hf_config_dir)
        torch.distributed.barrier()

        optim_after = (
            len(self.actor_optimizer.state),
            sum(len(v) for v in self.actor_optimizer.state.values()),
        )
        if optim_before != optim_after:
            raise RuntimeError(
                f'Optimizer state changed during snapshot: {optim_before} -> {optim_after}'
            )
        print(
            f'[rank-{self.rank}] SNAPSHOT_OPTIMIZER_PRESERVED '
            f'step={step} states={optim_after}',
            flush=True,
        )
        # Rank 0 writes the success manifest only after BOTH ranks saved.
        if self.rank == 0:
            manifest = {
                'logical_step': int(step),
                'world_size': int(self.world_size),
                'format': 'fsdp_sharded_state_dict_model_only',
                'optimizer_preserved_in_running_process': True,
                'optimizer_state_saved': False,
                'scheduler_and_rng_saved': False,
            }
            tmp_manifest = os.path.join(local_path, 'snapshot_manifest.json.incomplete')
            final_manifest = os.path.join(local_path, 'snapshot_manifest.json')
            with open(tmp_manifest, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp_manifest, final_manifest)
        torch.distributed.barrier()
        print(f'[rank-{self.rank}] SNAPSHOT_RANK_DONE step={step}', flush=True)
        torch.cuda.empty_cache()

'''

trainer_snippet = '''                # Smoke-only periodic model snapshots. Unlike the final export,
                # this RPC MUST preserve AdamW state for the next actor update.
                snapshot_every = int(self.config.trainer.get('smoke_snapshot_every', 0))
                if snapshot_every > 0 and self.global_steps % snapshot_every == 0:
                    snapshot_dir = os.path.join(
                        self.config.trainer.default_local_dir,
                        'actor',
                        f'snapshot_step_{self.global_steps}',
                    )
                    self.actor_rollout_wg.snapshot_model_only_shards(
                        snapshot_dir, self.global_steps,
                    )
                    print(
                        f'SMOKE_NONDESTRUCTIVE_SNAPSHOT_DONE: '
                        f'step={self.global_steps} path={snapshot_dir}',
                        flush=True,
                    )

'''

worker_new = worker.replace(worker_anchor, worker_method + worker_anchor, 1)
trainer_new = trainer.replace(trainer_anchor, trainer_snippet + trainer_anchor, 1)

# Three-step known-good source has total_epochs=3 and total_training_steps=3.
# In this trainer, global_steps STARTS AT ONE and the stop check happens after
# increment, so total_training_steps=3 => two updates: steps 1 and 2.
script_new = script
if '+trainer.smoke_export_model_only=true' in script_new:
    script_new = script_new.replace('+trainer.smoke_export_model_only=true', '+trainer.smoke_export_model_only=false')
if '+trainer.smoke_snapshot_every=' in script_new:
    raise RuntimeError('Smoke snapshot flag already present in source script')
script_new = script_new.replace(
    '  trainer.save_freq=-1 \\\n',
    "  '+trainer.smoke_snapshot_every=1' \\\n  trainer.save_freq=-1 \\\n",
    1,
)
assert script_new != script, 'Could not inject snapshot flag into shell command'
assert script_new.count('+trainer.smoke_snapshot_every=1') == 1
assert 'Qwen2.5-3B-R2-SFT-E5' in script_new, 'Script is not initialized from SFT model'

compile(worker_new, str(worker_path), 'exec')
compile(trainer_new, str(trainer_path), 'exec')

for path in (worker_path, trainer_path):
    backup = path.with_name(path.name + '.before_nondestructive_snapshot.bak')
    if backup.exists():
        raise RuntimeError(f'Backup already exists: {backup}')
for path in (worker_path, trainer_path):
    backup = path.with_name(path.name + '.before_nondestructive_snapshot.bak')
    shutil.copy2(path, backup)

worker_path.write_text(worker_new)
trainer_path.write_text(trainer_new)
output_script.write_text(script_new)
output_script.chmod(source_script.stat().st_mode)

print('NONDESTRUCTIVE_SNAPSHOT_PATCH_OK')
print('Smoke test:', output_script)
print('Expected TWO RL updates and TWO snapshot directories (steps 1 and 2).')
