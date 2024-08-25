import yaml
import os
import socket
import shutil

configs = {}
_directory = os.path.dirname(__file__)
for file in os.listdir(f"{_directory}/config"):
    key = file.split(".")[0]
    configs[key] = f"{_directory}/config/{file}"

def get_free_gpus(num_devices: int) -> list[str]:
    import torch

    GB = 1024**3

    devices = []
    for i in range(num_devices):
        mem_alloc = torch.cuda.memory_allocated(i)
        mem_resvd = torch.cuda.memory_reserved(i)
        mem_total = mem_alloc + mem_resvd
        if mem_total > 0:
            device_name = torch.cuda.get_device_name(i)
            print("------------------------------------------------")
            print("GPU in use:")
            print(f"Name: {device_name}")
            print(f"Memory allocated: {mem_alloc/GB} GB")
            print(f"Memory reserved: {mem_resvd/GB} GB")
            print("ACCMT will not use this GPU during training.")
            print("------------------------------------------------")

            continue

        devices.append(str(i))

    return devices

def check_port_available(port: int, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result != 0

def modify_config_file(path: str, num_gpus: int, port: int = 29500):
    data = yaml.safe_load(open(path))

    _port = port
    port = port if check_port_available(port) else 0
    if port == 0:
        for current_port in range(_port+1, 65536):
            if check_port_available(current_port):
                port = current_port
                break
        
        if port == 0: # if 29500 to 65535 is not available
            for current_port in range(1, _port):
                if check_port_available(current_port):
                    port = current_port
                    break

        if port == 0:
            raise RuntimeError("There are no ports available in your system.")
    
    prev_main_process_port = data["main_process_port"] if "main_process_port" in data else -1
    prev_num_processes = data["num_processes"]

    if prev_main_process_port == port and prev_num_processes == num_gpus:
        return # skip write process

    data["main_process_port"] = port
    data["num_processes"] = num_gpus

    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def get_python_cmd():
    if shutil.which("python") is not None:
        return "python"
    else:
        return "python3"

def remove_compiled_prefix(state_dict):
    compiled = "_orig_mod" in list(state_dict.keys())[0]
    if not compiled: return state_dict

    t = type(state_dict)
    return t({k.removeprefix("_orig_mod."):v for k, v in state_dict.items()})

def show_strategies(filter: str = None):
    if filter is None: filter = ""
    for strat in configs.keys():
        if filter in strat:
            print(f"\t{strat}")

    exit(1)

def generate_hps():
    directory = os.path.dirname(__file__)
    shutil.copy(f"{directory}/example/hps_example.yaml", ".")
