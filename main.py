import argparse

from plot_results import main as plot_main
from test_env import main as test_env_main
from train_ppo_mac import main as train_ppo_main
from train_sac_mac import main as train_sac_main


def build_parser():
    parser = argparse.ArgumentParser(description="Unified entry point for RL tasks.")
    parser.add_argument(
        "command",
        choices=["test", "ppo", "sac", "plot"],
        help="Action to run.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "test":
        test_env_main()
    elif args.command == "ppo":
        train_ppo_main()
    elif args.command == "sac":
        train_sac_main()
    elif args.command == "plot":
        plot_main()


if __name__ == "__main__":
    main()
