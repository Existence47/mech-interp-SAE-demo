import sys

import torch


def main() -> None:
    print("Python:", sys.version)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("TransformerLens import: OK")


if __name__ == "__main__":
    main()
