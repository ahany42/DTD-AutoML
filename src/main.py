import sys


def main():
    mode = "dynamic"  

    if mode == "deterministic":
        from run_deterministic import run
    elif mode == "dynamic":
        from run_dynamic import run
    else:
        print("Invalid mode")
        return

    run()


if __name__ == "__main__":
    main()