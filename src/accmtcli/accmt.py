#!/usr/bin/env python
import os
from argparse import ArgumentParser, REMAINDER
from .utils import configs, modify_config_file, get_free_gpus

def main():
    parser = ArgumentParser(description="AcceleratorModule CLI to run train processes on top of 🤗 Accelerate.")

    parser.add_argument(
        "--gpus",
        "-n",
        default="all",
        type=str,
        required=False,
        help="Number or GPU indices to use (e.g. -n=0,1,4,5 | -n=all | -n=available)."
    )
    parser.add_argument(
        "-N",
        default="0",
        type=str,
        required=False,
        help="Number of GPUs to use. This does not consider GPU indices by default, although you can represent "
             "a Python slice. (e.g. '2:', which means from index 2 to the last GPU index, or "
             "'3:8', which means from index 3 to index 7, or lastly ':4', which means indices 0 to 3 or a total of 4 gpus)."
    )
    parser.add_argument(
        "--strat",
        type=str,
        required=False,
        default="ddp",
        help="Parallelism strategy to apply. See 'accmth --strategies'."
    )
    parser.add_argument(
        "file",
        type=str,
        help="File to run training."
    )
    parser.add_argument(
        "extra_args",
        nargs=REMAINDER
    )

    args = parser.parse_args()
    gpus = args.gpus.lower()
    strat = args.strat
    file = args.file
    extra_args = " ".join(args.extra_args)

    accelerate_config_file = configs[strat]

    import torch
    if not torch.cuda.is_available():
        raise ImportError("Could not run CLI: CUDA is not available on your PyTorch installation.")

    NUM_DEVICES = torch.cuda.device_count()

    gpu_indices = ""
    if gpus == "available":
        gpu_indices = ",".join(get_free_gpus(NUM_DEVICES))
    elif gpus == "all":
        gpu_indices = ",".join(str(i) for i in range(NUM_DEVICES))
    else:
        gpu_indices = gpus.removeprefix(",").removesuffix(",")

    if gpu_indices == "":
        raise RuntimeError("Could not get GPU indices. If you're using 'available' in 'gpus' "
                           "parameter, make sure there is at least one GPU free of memory.")

    if args.N != "0":
        if ":" in args.N:
            _slice = slice(*map(lambda x: int(x.strip()) if x.strip() else None, args.N.split(':')))
            gpu_indices = ",".join([str(i) for i in range(NUM_DEVICES)][_slice])
        else:
            gpu_indices = ",".join(str(i) for i in range(int(args.N)))

    num_processes = len(gpu_indices.split(","))
    modify_config_file(accelerate_config_file, num_processes)
    
    cmd = (f"CUDA_VISIBLE_DEVICES={gpu_indices} "
            f"accelerate launch --config_file={accelerate_config_file} "
            f"{file} {extra_args}")
    
    os.system(cmd)


if __name__ == "__main__":
    main()
