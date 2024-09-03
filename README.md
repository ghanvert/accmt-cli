# AcceleratorModule CLI
CLI for AcceleratorModule library ([accmt](https://github.com/ghanvert/AcceleratorModule)).

This is a command-line tool wrapper around Accelerate's command-line utilities ('accelerate').

## Installation
**accmt-cli** is automatically installed when installing **accmt** library. You also install it via pip:
```bash
pip install accmt-cli
```

## Launch
You can launch any distributed training process with the following command:
```bash
accmt launch [-N][-n, --gpus][-O1][--strat] <your_python_script> [...]
```

Where:
- **-N** (*optional*): Corresponds to the number of processes, or a Python-like slice to take GPUs from a certain index (e.g. '-N=2:', to take GPUs from index 2).
- **-n** or **--gpus** (*optional*): Corresponds to a list of CUDA devices (e.g. '-n=1,3,5,6', to take GPUs indices 1, 3, 5 and 6).
- **-O1** (*optional*): Corresponds to the optimization of type 1, which calculates the efficient number for 'OMP_NUM_THREADS', depending on how many processes you will run you training script.
- **--strat** (*optional*): Corresponds to the specific strategy to implement, or a configuration file path from Accelerate ('accelerate config --config_file=your-config.yaml'). See 'accmt strats' for specific strategies.
**...** (*optional*): You can add here any additional arguments that your Python script might have.

## Get model from checkpoint
You can get a model from any checkpoint using the following command:
```bash
accmt get <checkpoint> --out=<output-model-directory> [--dtype]
```

Where:
- **--out** or **-O** (*REQUIRED*): Output model directory name where to save the model.
- **--dtype** (*Optional*): PyTorch data type of model parameters. Default is 'float32'.

## Strats
You can check the specific strats included with the following command:
```bash
accmt strats [--ddp][--fsdp][--deepspeed]
```

Where:
- **--ddp**: To only filter for DDP strategies.
- **--fsdp**: To only filter for FSDP strategies.
- **--deepspeed**: To only filter for DeepSpeed strategies.

## Example
Generate an example HPS file config with the following command:
```bash
accmt example
```

This will generate a file on your current directory called 'hps_example.yaml'.
