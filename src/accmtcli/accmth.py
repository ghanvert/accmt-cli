from argparse import ArgumentParser

def show_strategies():
    from utils import configs

    for strat in configs.keys():
        print(f"\t{strat}")

    exit(1)

def generate_hps():
    import os
    import shutil

    directory = os.path.dirname(__file__)
    shutil.copy(f"{directory}/example/hps_example.yaml", ".")

    exit(1)

if __name__ == "__main__":
    parser = ArgumentParser(description="ACCMT Helper CLI.")
    parser.add_argument(
        "--strategies",
        action="store_true",
        required=False,
        help="Show available strategies."
    )
    parser.add_argument(
        "--hps",
        action="store_true",
        required=False,
        help="Generate a HPS (Hyperparameters) YAML file example."
    )
    args = parser.parse_args()

    strategies = args.strategies
    hps = args.hps

    if strategies:
        show_strategies()
    elif hps:
        generate_hps()
    else:
        print("ERROR: You must provide one argument ('--strategies' or '--hps').")
