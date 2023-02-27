import os
import traceback
from types import TracebackType
from typing import Optional, Type

from torch.distributed import (barrier, destroy_process_group,
                               init_process_group)


class DDPManager:
    def __init__(self, ddp: bool = True) -> None:
        self.ddp = ddp

    def __enter__(self):
        if self.ddp:
            print("Setting up process groups for DDP training.")
            dist_url = "env://"  # default
            # only works with torch.distributed.launch // torch.run
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)

            # synchronizes all the threads to reach this point before moving on
            barrier()

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> bool:
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_val, exc_tb)
            return False

        if self.ddp:
            print("Tearing down process groups for DDP training.")
            destroy_process_group()

        return True
