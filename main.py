import torch


def main():
    print("Hello from attention-recreation!")

    print("Checking torch installation...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"CUDA device properties: {torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'N/A'}")
    print("Torch installation check complete.")


if __name__ == "__main__":
    main()
