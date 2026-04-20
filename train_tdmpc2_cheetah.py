from tdmpc2.train_tdmpc2 import main as train_main


def main(total_steps: int = 10_000, max_wall_clock_seconds: float = 0.0) -> None:
    train_main(
        env_name="cheetah",
        run_name="tdmpc2_cheetah_mlp",
        total_steps=total_steps,
        max_wall_clock_seconds=max_wall_clock_seconds,
    )


if __name__ == "__main__":
    main()
