# repurposed from: https://github.com/jxmorris12/vec2text/blob/master/vec2text/utils/utils.py

import multiprocessing
import os
import shutil
from typing import Callable

import datasets
import torch

datasets.disable_caching()


def get_world_size() -> int:
    """get number of processes in distributed setup"""
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def get_num_proc() -> int:
    """get number of cpu cores for each process"""
    world_size: int = get_world_size()
    try:
        return len(os.sched_getaffinity(0)) // world_size  # type: ignore[attr-defined]
    except AttributeError:
        return multiprocessing.cpu_count() // world_size


def torch_main_worker_finish_first(func: Callable):
    def wrapper(*args, **kwargs):
        try:
            local_rank = torch.distributed.get_rank()
            # main worker rank is 0
            ddp_enabled = True
        except (RuntimeError, ValueError):
            local_rank = -1
            ddp_enabled = False
        is_main_worker = local_rank <= 0
        # Run on main worker first.
        if is_main_worker:
            result = func(*args, **kwargs)
        # Barrier: wait till every other process in dist system reaches
        if ddp_enabled:
            torch.distributed.barrier()
        # Once everyone has reached, run for them
        if not is_main_worker:
            result = func(*args, **kwargs)
        # Barrier again: wait again until function executes
        if ddp_enabled:
            torch.distributed.barrier()
        return result

    return wrapper


def dataset_map_multi_worker(
    dataset: datasets.Dataset, map_fn: Callable, *args, **kwargs
) -> datasets.Dataset:
    """parallely apply a map_fn on dataset"""
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        kwargs["num_proc"] = kwargs.get("num_proc", get_num_proc())
    except (RuntimeError, ValueError):
        # if not running in distributed mode
        kwargs["num_proc"] = kwargs.get("num_proc", get_num_proc())
        return dataset.map(map_fn, *args, **kwargs)
    datasets.disable_caching()

    cache_path = os.environ.get(
        "MNIST_CACHE", os.path.expanduser("~/.cache/mnist")
    )  # if env var not set, default to .cache/mnist

    ds_shard_filepaths = [
        os.path.join(cache_path, f"{dataset._fingerprint}_subshard_{w}.cache")
        for w in range(0, world_size)
    ]  # one shard for each world

    print(f"\tworker {rank} saving sub-shard to {ds_shard_filepaths[rank]}")
    ds_shard = dataset.shard(num_shards=world_size, index=rank, contiguous=True)
    ds_shard = ds_shard.map(map_fn, *args, **kwargs)
    ds_shard.save_to_disk(ds_shard_filepaths[rank])
    print("rank", rank, "saving:", ds_shard_filepaths[rank])
    # why save?: free up active space for computations
    torch.distributed.barrier()  # wait till all ranks done
    full_dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(p) for p in ds_shard_filepaths]
    )
    torch.distributed.barrier()
    print("rank", rank, "deleting:", ds_shard_filepaths[rank])
    shutil.rmtree(ds_shard_filepaths[rank])
    return full_dataset
