import os
import traceback
from types import TracebackType
from typing import Optional, Type

from torch.distributed import destroy_process_group, init_process_group


class DDPManager:
    """
    Context manager for setting up and tearing down DDP process groups.
    """
    def __init__(self, ddp: bool = True) -> None:
        self.ddp = ddp

    def __enter__(self):
        if self.ddp:
            dist_url = "env://"  # default
            # only works with torch.distributed.launch // torch.run
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])

            print(f"Setting up process groups for DDP training on rank {self.rank}.")
            init_process_group(backend="nccl", init_method=dist_url, world_size=self.world_size, rank=self.rank)

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> bool:
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_val, exc_tb)
            return False

        if self.ddp:
            print(f"Tearing down process groups for DDP training on rank {self.rank}.")
            destroy_process_group()

        return True
