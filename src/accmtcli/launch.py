import os
import torch
from .utils import get_free_gpus, configs, modify_config_file

def launch(args):
    if args.command == "debug":
        os.environ["ACCMT_DEBUG_MODE"] = str(args.level)

    gpus = args.gpus.lower()
    strat = args.strat
    file = args.file
    extra_args = " ".join(args.extra_args)

    if "." in strat:
        accelerate_config_file = strat
    else:
        accelerate_config_file = configs[strat]

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

    # TODO: For now, we need to find a way to collect processes that are running on certain GPUs to verify if they're free to use.
    #if not args.ignore_warnings:
    #    gpu_indices_list = [int(idx) for idx in gpu_indices.split(",")]
    #    device_indices_in_use = []
    #    for idx in gpu_indices_list:
    #        if cuda_device_in_use(idx):
    #            device_indices_in_use.append(idx)
    #
    #    if len(device_indices_in_use) > 0:
    #        raise RuntimeError(
    #            f"The following CUDA devices are in use: {device_indices_in_use}."
    #             "You can ignore this warning via '--ignore-warnings'."
    #        )
    
    num_processes = len(gpu_indices.split(","))
    modify_config_file(accelerate_config_file, num_processes)
    
    optimization1 = f"OMP_NUM_THREADS={os.cpu_count() // num_processes}" if args.O1 else ""

    cmd = (f"{optimization1} CUDA_VISIBLE_DEVICES={gpu_indices} "
            f"accelerate launch --config_file={accelerate_config_file} "
            f"{file} {extra_args}")
    
    os.system(cmd)
