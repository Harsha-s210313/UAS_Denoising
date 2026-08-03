import sys
from generator import generate_dataset, print_summary


def main():
    print("Dataset Generation Started...")
    try:
        stats = generate_dataset()
        print_summary(stats)
    except Exception as e:
        print(f"\nExecution failed with error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
