import time

import torch


def main() -> None:
    device = torch.device("cuda:0")
    torch.cuda.init()

    for _ in range(100):
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        _ = free + reserved - allocated

    n = 10_000
    start = time.perf_counter()
    for _ in range(n):
        free = torch.cuda.get_device_properties(device)
        torch.cuda.get_per_process_memory_fraction(device)
        # allocated = torch.cuda.memory_allocated(device)
        # reserved = torch.cuda.memory_reserved(device)
        # _ = free + reserved - allocated
    elapsed = time.perf_counter() - start

    print(f"{elapsed / n * 1e6:.2f} us/call")


if __name__ == "__main__":
    main()
