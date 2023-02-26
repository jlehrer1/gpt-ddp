import os

from torch.distributed import (barrier, destroy_process_group,
                               init_process_group)


class DDPManager:
    def __enter__(self):
        print("Setting up process groups for DDP training.")
        self.init_distributed()

    def __exit__():
        print("Tearing down process groups for DDP training.")
        destroy_process_group()

    def init_distributed(self):
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        dist_url = "env://"  # default

        # only works with torch.distributed.launch // torch.run
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)

        # synchronizes all the threads to reach this point before moving on
        barrier()
