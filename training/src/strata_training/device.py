import time

import torch

from strata_training.torch_api import api


def configure_gpu(reserve_gib: float = 12) -> None:
    torch.set_num_threads(8)
    while True:
        free, total = torch.cuda.mem_get_info()
        budget = min(12 * 2**30, free - int(reserve_gib * 2**30) - 128 * 2**20)
        if budget > 128 * 2**20:
            api.cuda.set_per_process_memory_fraction(budget / total)
            return
        time.sleep(10)
