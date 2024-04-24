import yaml
import os

configs = {}
_directory = os.path.dirname(__file__)
for file in os.listdir("config"):
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


def modify_config_file(path: str, num_gpus: int):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    data["num_processes"] = num_gpus

    with open(path, "w") as f:
        yaml.safe_dump(data, f)
