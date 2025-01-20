from argparse import ArgumentParser, REMAINDER, Namespace


def add_launch_arguments(parser: ArgumentParser):
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
        help="Parallelism strategy to apply or config file path. See 'accmt strats'."
    )
    parser.add_argument("-O1", action="store_true", help="Apply optimization type 1: efficient OMP_NUM_THREADS.")

    # TODO: For now, we need to find a way to collect processes that are running on certain GPUs to verify if they're free to use.
    #parser.add_argument("--ignore-warnings", action="store_true", help="Ignore warnings (launch independent if GPUs are being used).")

    parser.add_argument("file", type=str, help="File to run training.")
    parser.add_argument("extra_args", nargs=REMAINDER)


def get_parser() -> tuple[ArgumentParser, Namespace]:
    parser = ArgumentParser(description="AcceleratorModule CLI to run train processes on top of 🤗 Accelerate.")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Run distributed training
    launch_parser = subparsers.add_parser("launch", help="Launch distributed training processes.")
    add_launch_arguments(launch_parser)
    debug_parser = subparsers.add_parser("debug", help="Launch distributed training processes in debug mode.")
    add_launch_arguments(debug_parser)
    debug_parser.add_argument(
        "--level",
        "-L",
        "-l",
        type=int,
        default=4,
        required=False,
        help="Debug mode level. See more details using 'accmt debug-levels'."
    )

    # Get model from checkpoint
    get_parser = subparsers.add_parser("get", help="Get model from a checkpoint directory.")
    get_parser.add_argument("checkpoint", type=str, help="Checkpoint directory.")
    get_parser.add_argument("--out", "-O", "-o", required=True, type=str, help="Output directory path name.")
    get_parser.add_argument("--dtype", type=str, default="float32", help=(
        "Data type of model parameters. Available options are all "
        "those from PyTorch ('float32', 'float16', etc)."
    ))

    # Strats
    strats_parser = subparsers.add_parser("strats", help="Available strategies.")
    strats_parser.add_argument("--ddp", action="store_true", help="Only show DistributedDataParallel (DDP) strategies.")
    strats_parser.add_argument("--fsdp", action="store_true", help="Only show FullyShardedDataParallel (FSDP) strategies.")
    strats_parser.add_argument("--deepspeed", action="store_true", help="Only show DeepSpeed strategies.")

    # Debug levels
    debug_levels_parser = subparsers.add_parser("debug-levels", help="Available debug levels.")
    debug_levels_parser.add_argument(
        "--level",
        "-L",
        "-l",
        type=int,
        required=False,
        help="See details about a specific debug mode level."
    )

    # Generate example
    subparsers.add_parser("example", help="Generate example file.")

    return parser, parser.parse_args()
