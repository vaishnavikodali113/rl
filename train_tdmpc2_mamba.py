from tdmpc2.train_tdmpc2 import main as train_main


def main(
    env_name: str = "walker",
    task: str | None = None,
    run_name: str | None = None,
    total_steps: int = 10_000,
    max_wall_clock_seconds: float = 0.0,
) -> None:
    train_main(
        env_name=env_name,
        task=task,
        dynamics_type="mamba",
        run_name=run_name or f"tdmpc2_{env_name}_mamba",
        total_steps=total_steps,
        max_wall_clock_seconds=max_wall_clock_seconds,
    )


if __name__ == "__main__":
    main()
