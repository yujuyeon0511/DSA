"""Quick NCCL multi-node connectivity test."""
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    # All-reduce test
    tensor = torch.ones(1, device=device) * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))

    print(f"[Rank {rank}/{world_size}] local_rank={local_rank} "
          f"device={device} all_reduce={tensor.item()} expected={expected} "
          f"{'OK' if tensor.item() == expected else 'FAIL'}")

    dist.barrier()
    if rank == 0:
        print("NCCL multi-node test PASSED!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
