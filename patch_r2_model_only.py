from pathlib import Path
import shutil
import re

worker_path = Path('verl/workers/fsdp_workers.py')
trainer_path = Path('verl/trainer/ppo/ray_trainer_ts.py')
source_script = Path('experiments/r2_tree_rl_sft_smoke.sh')
output_script = Path('experiments/r2_tree_rl_sft_export_smoke.sh')

for p in (worker_path, trainer_path, source_script):
    if not p.is_file():
        raise FileNotFoundError(f'Required file is missing: {p}')

w = worker_path.read_text()
t = trainer_path.read_text()
sh = source_script.read_text()

worker_anchor = (
    '    @register(dispatch_mode=Dispatch.ONE_TO_ALL)\n'
    '    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):'
)
assert w[:w.index('class CriticWorker(')].count(worker_anchor) == 1, 'Actor worker anchor mismatch or already patched'
assert 'def export_model_only_shards(' not in w, 'Worker already patched'

method = '''    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_model_only_shards(self, local_path):
        """One-off smoke export: FSDP model shards only, NOT a resumable checkpoint.

        This is called exclusively after the last optimizer update. We deliberately
        release AdamW states to lower host RAM usage on the 128-GiB V100 node.
        """
        assert self._is_actor
        assert self._is_offload_optimizer, 'Expected optimizer_offload=true'
        import os
        import gc
        import ctypes
        import torch
        from torch.distributed.fsdp import (
            FSDP, StateDictType, ShardedStateDictConfig,
        )

        # Destructive by design. Never use this RPC mid-training.
        self.actor_optimizer.state.clear()
        gc.collect()
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass
        print(f'[rank-{self.rank}] MODEL_ONLY_OPTIMIZER_RELEASED', flush=True)

        if self._is_offload_param:
            load_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                device_id=torch.cuda.current_device(),
                load_grad=self._is_offload_grad,
            )

        state_config = ShardedStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(
            self.actor_module_fsdp,
            StateDictType.SHARDED_STATE_DICT,
            state_config,
        ):
            state = self.actor_module_fsdp.state_dict()

        os.makedirs(local_path, exist_ok=True)
        path = os.path.join(
            local_path,
            f'model_world_size_{self.world_size}_rank_{self.rank}.pt',
        )
        print(f'[rank-{self.rank}] MODEL_ONLY_SAVING {path}', flush=True)
        torch.save(state, path)
        del state
        gc.collect()
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass
        print(f'[rank-{self.rank}] MODEL_ONLY_SHARD_SAVED {path}', flush=True)

        torch.distributed.barrier()
        if self.rank == 0:
            hf_config_dir = os.path.join(local_path, 'huggingface')
            os.makedirs(hf_config_dir, exist_ok=True)
            self.actor_model_config.save_pretrained(hf_config_dir)
            self.tokenizer.save_pretrained(hf_config_dir)
            if self.generation_config is not None:
                self.generation_config.save_pretrained(hf_config_dir)
        torch.distributed.barrier()

        if self._is_offload_param:
            offload_fsdp_param_and_grad(
                module=self.actor_module_fsdp,
                offload_grad=self._is_offload_grad,
            )
        torch.cuda.empty_cache()

'''
w_new = w.replace(worker_anchor, method + worker_anchor, 1)

marker = "print('SMOKE_STEP_FINISHED_NO_CHECKPOINT', flush=True)"
assert t.count(marker) == 1, 'Smoke stop marker missing/duplicated'
assert 'SMOKE_MODEL_ONLY_EXPORT_DONE' not in t, 'Trainer already patched'
lines = t.splitlines(keepends=True)
new_lines = []
for line in lines:
    if marker in line:
        indentation = re.match(r'^\s*', line).group()
        new_lines += [
            indentation + "if self.config.trainer.get('smoke_export_model_only', False):\n",
            indentation + "    export_dir = os.path.join(self.config.trainer.default_local_dir, 'actor',\n",
            indentation + "                              f'model_only_step_{self.global_steps}')\n",
            indentation + "    self.actor_rollout_wg.export_model_only_shards(export_dir)\n",
            indentation + "    print(f'SMOKE_MODEL_ONLY_EXPORT_DONE: {export_dir}', flush=True)\n",
        ]
    new_lines.append(line)
t_new = ''.join(new_lines)

flag = '+trainer.smoke_skip_saving=true'
assert sh.count(flag) == 1, 'Expected one smoke_skip_saving flag'
assert '+trainer.smoke_export_model_only=true' not in sh, 'Export flag already present'
assert not output_script.exists(), f'Output script already exists: {output_script}'
sh_new = sh.replace(flag, flag + ' +trainer.smoke_export_model_only=true', 1)

compile(w_new, str(worker_path), 'exec')
compile(t_new, str(trainer_path), 'exec')

for p in (worker_path, trainer_path):
    backup = p.with_name(p.name + '.before_model_only_export.bak')
    if backup.exists():
        raise RuntimeError(f'Backup already exists, refusing overwrite: {backup}')

for p in (worker_path, trainer_path):
    backup = p.with_name(p.name + '.before_model_only_export.bak')
    shutil.copy2(p, backup)

worker_path.write_text(w_new)
trainer_path.write_text(t_new)
output_script.write_text(sh_new)
output_script.chmod(source_script.stat().st_mode)

print('MODEL_ONLY_EXPORT_PATCH_OK')
print('Run:', output_script)
