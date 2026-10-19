import os
from datetime import timedelta
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)

print(
    f"[rank {rank}] GPU={torch.cuda.get_device_name(local_rank)} "
    f"device={local_rank}",
    flush=True,
)

dist.init_process_group(
    backend="cpu:gloo,cuda:nccl",
    timeout=timedelta(seconds=45),
)

x = torch.tensor(
    [float(rank + 1)],
    device=f"cuda:{local_rank}",
)

print(f"[rank {rank}] BEFORE all_reduce", flush=True)

dist.all_reduce(x)

torch.cuda.synchronize()

value = x.item()
print(f"[rank {rank}] AFTER all_reduce: {value}", flush=True)
assert abs(value - 3.0) < 1e-6, (
    f"Rank {rank}: incorrect AllReduce result {value}"
)

dist.barrier(device_ids=[local_rank])

print(f"[rank {rank}] NCCL PASS", flush=True)

dist.destroy_process_group()
