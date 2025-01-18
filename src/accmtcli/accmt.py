#!/usr/bin/env python
import os
import shutil
from .utils import configs, modify_config_file, get_free_gpus, get_python_cmd, remove_compiled_prefix, generate_hps, show_strategies, cuda_device_in_use
from .parser import get_parser

def main():
    parser, args = get_parser()

    if args.command is None:
        parser.print_help()
        exit(0)

    import torch

    if args.command in ["launch", "debug"]:
        if args.command == "debug":
            os.environ["ACCMT_DEBUG_MODE"] = args.level

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

        if not args.ignore_warnings:
            gpu_indices_list = [int(idx) for idx in gpu_indices.split(",")]
            device_indices_in_use = []
            for idx in gpu_indices_list:
                if cuda_device_in_use(idx):
                    device_indices_in_use.append(idx)

            if len(device_indices_in_use) > 0:
                raise RuntimeError(
                    f"The following CUDA devices are in use: {device_indices_in_use}."
                     "You can ignore this warning via '--ignore-warnings'."
                )
        
        num_processes = len(gpu_indices.split(","))
        modify_config_file(accelerate_config_file, num_processes)
        
        optimization1 = f"OMP_NUM_THREADS={os.cpu_count() // num_processes}" if args.O1 else ""

        cmd = (f"{optimization1} CUDA_VISIBLE_DEVICES={gpu_indices} "
                f"accelerate launch --config_file={accelerate_config_file} "
                f"{file} {extra_args}")
        
        os.system(cmd)
    elif args.command == "get":
        assert args.out is not None, "You must specify an output directory ('--out')."
        assert hasattr(torch, args.dtype), f"'{args.dtype}' not supported in PyTorch."
        CHKPT_BASE_DIRECTORY = f"{args.checkpoint}/checkpoint"
        checkpoint_dir = CHKPT_BASE_DIRECTORY if os.path.exists(CHKPT_BASE_DIRECTORY) else args.get
        files = os.listdir(checkpoint_dir)

        python_cmd = get_python_cmd()
        os.makedirs(args.out, exist_ok=True)
        if "status.json" in os.listdir(args.checkpoint):
            shutil.copy(f"{args.checkpoint}/status.json", args.out)
        
        state_dict_file = f"{args.out}/pytorch_model.pt"

        if "zero_to_fp32.py" in files: # check for DeepSpeed
            print("Converting Zero to float32 parameters...")
            exit_code = os.system(f"{python_cmd} {checkpoint_dir}/zero_to_fp32.py {checkpoint_dir} {state_dict_file}")
            if exit_code != 0:
                raise RuntimeError("Something went wrong when converting Zero to float32.")
        elif "pytorch_model_fsdp_0" in files: # check for FSDP
            # using Accelerate's approach for now, and only checking for one node
            exit_code = os.system(f"accelerate merge-weights {checkpoint_dir}/pytorch_model_fsdp_0 {args.out}")
            if exit_code != 0:
                raise RuntimeError("Something went wrong when merging weights from FSDP.")
            
            shutil.copy(f"{checkpoint_dir}/pytorch_model.bin", f"{args.out}/pytorch_model.pt")
        else: # check for DDP
            shutil.copy(f"{checkpoint_dir}/pytorch_model.bin", f"{args.out}/pytorch_model.pt")
            
        state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        state_dict = remove_compiled_prefix(state_dict)

        _dtype_str = f" and converting to dtype {args.dtype}" if args.dtype is not None else ""
        print(f"Setting 'requires_grad' to False{_dtype_str}...")
        for key in state_dict.keys():
            state_dict[key].requires_grad = False
            if args.dtype is not None:
                state_dict[key] = state_dict[key].to(getattr(torch, args.dtype))

        torch.save(state_dict, state_dict_file)
        print(f"Model directory saved to '{args.out}'.")
    elif args.command == "strats":
        if args.ddp:
            show_strategies(filter="ddp")
        elif args.fsdp:
            show_strategies(filter="fsdp")
        elif args.deepspeed:
            show_strategies(filter="deepspeed")
        else:
            show_strategies()
    elif args.command == "example":
        generate_hps()
        print("'hps_example.yaml' generated.")

if __name__ == "__main__":
    main()
