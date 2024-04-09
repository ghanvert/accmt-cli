import os

from argparse import ArgumentParser, REMAINDER
from utils import configs, modify_config_file, get_free_gpus

def main():
    import torch

    if not torch.cuda.is_available():
        raise ImportError("Could not run CLI: CUDA is not available on your PyTorch installation.")

    NUM_DEVICES = torch.cuda.device_count()

    parser = ArgumentParser(description="AcceleratorModule CLI to run train processes on top of 🤗 Accelerate.")

    parser.add_argument(
        "--gpus",
        "-n",
        default="available",
        type=str,
        required=False,
        help="Number or GPU indices to use (e.g. -n=0,1,4,5 | -n=all | -n=available)."
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        required=False,
        help="Applies standard Distributed Data Parallel (DDP) strategy."
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        required=False,
        help="Applies standard Fully Sharded Data Parallel (FSDP) strategy."
    )
    parser.add_argument(
        "--deepspeed",
        "--ds",
        required=False,
        help="Applies standard DeepSpeed strategy."
    )
    parser.add_argument(
        "file",
        type=str,
        required=True,
        help="File to run training."
    )
    parser.add_argument(
        "extra_args",
        nargs=REMAINDER
    )

    args = parser.parse_args()
    gpus = args.gpus.lower()
    ddp = args.ddp
    fsdp = args.fsdp
    deepspeed = args.deepspeed
    file = args.file
    extra_args = " ".join(args.extra_args)

    accelerate_config_file = None
    if all([ddp, fsdp, deepspeed]):
        raise RuntimeError("Cannot execute multiple strategies. Please, choose only one.")
    elif ddp:
        accelerate_config_file = configs["ddp"]
    elif fsdp:
        accelerate_config_file = configs["fsdp"]
    elif deepspeed:
        accelerate_config_file = configs["deepspeed"]
    else:
        accelerate_config_file = configs["ddp"]

    gpu_indices = ""
    if gpus == "available":
        gpu_indices = ",".join(get_free_gpus(NUM_DEVICES))
    elif gpus == "all":
        gpu_indices = ",".join([str(i) for i in range(NUM_DEVICES)])
    else:
        gpu_indices = gpus.removeprefix(",").removesuffix(",")

    if gpu_indices == "":
        raise RuntimeError("Could not get GPU indices. If you're using 'available' in 'gpus' "
                           "parameter, make sure there is at least one GPU free of memory.")

    modify_config_file(accelerate_config_file, len(gpu_indices))

    cmd = (f"CUDA_AVAILABLE_DEVICES={gpu_indices} "
               f"accelerate launch --config_file={accelerate_config_file} "
               f"{file} {extra_args}")
    
    os.system(cmd)


if __name__ == "__main__":
    main()
