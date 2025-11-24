import argparse

from .prepare_dataset_from_info import prepare_dataset_from_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Adult dataset tensors.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio if val split absent.")
    parser.add_argument("--random-state", type=int, default=777, help="Random seed for splitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset_from_info(
        "adult",
        val_ratio=args.val_ratio,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
