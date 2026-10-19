#!/usr/bin/env python3
"""One-time, auditable change to the already installed model-only FSDP export.

Keep state dict tensors on GPUs instead of eagerly offloading to CPU;
serialize each rank one after another, with atomic output filenames.
This remains a non-resumable, model-only export.
"""
from pathlib import Path
import shutil

p = Path('verl/workers/fsdp_workers.py')
s = p.read_text()
start = s.index('    def export_model_only_shards(')
end = s.index('    @register(dispatch_mode=Dispatch.ONE_TO_ALL)\n    def save_checkpoint(', start)
block = s[start:end]

old_cfg = 'state_config = ShardedStateDictConfig(offload_to_cpu=True)'
new_cfg = 'state_config = ShardedStateDictConfig(offload_to_cpu=False)'
old_io = '''        print(f'[rank-{self.rank}] MODEL_ONLY_SAVING {path}', flush=True)
        torch.save(state, path)
        del state
        gc.collect()
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass
        print(f'[rank-{self.rank}] MODEL_ONLY_SHARD_SAVED {path}', flush=True)

        torch.distributed.barrier()
'''
new_io = '''        # All ranks must finish the collective state_dict() call before any rank saves.
        torch.distributed.barrier()
        print(f'[rank-{self.rank}] MODEL_ONLY_SERIAL_SAVE_READY', flush=True)
        for save_rank in range(self.world_size):
            if self.rank == save_rank:
                incomplete = path + '.incomplete'
                print(f'[rank-{self.rank}] MODEL_ONLY_SAVING {incomplete}', flush=True)
                torch.save(state, incomplete)
                os.replace(incomplete, path)
                del state
                gc.collect()
                try:
                    ctypes.CDLL('libc.so.6').malloc_trim(0)
                except (OSError, AttributeError):
                    pass
                print(f'[rank-{self.rank}] MODEL_ONLY_SHARD_SAVED {path}', flush=True)
            # At most one worker serializes at a time.
            torch.distributed.barrier()
'''

assert block.count(old_cfg) == 1, 'Unexpected FSDP offload setting; stop and inspect.'
assert block.count(old_io) == 1, 'Unexpected export code layout; stop and inspect.'
assert 'MODEL_ONLY_SERIAL_SAVE_READY' not in block, 'Already modified; do not patch twice.'
new_block = block.replace(old_cfg, new_cfg, 1).replace(old_io, new_io, 1)
new_source = s[:start] + new_block + s[end:]
compile(new_source, str(p), 'exec')
backup = p.with_name(p.name + '.before_gpu_serial_export.bak')
assert not backup.exists(), f'Backup already exists: {backup}'
shutil.copy2(p, backup)
p.write_text(new_source)
print('GPU_SERIAL_EXPORT_PATCH_OK')
print('Backup:', backup)
