"""Minimal 2-GPU NCCL all-reduce check (torchrun --nproc_per_node 2 scripts/nccl_test.py)."""
import os
import torch
import torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(4, device="cuda") * (r + 1)
dist.all_reduce(x)
print(f"rank {r} all_reduce -> {x.tolist()}", flush=True)
dist.destroy_process_group()
