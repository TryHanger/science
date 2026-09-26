import sys
import time

try:
    import torch
    import torchvision
    from torchvision import models
except ImportError as e:
    print(f"Error importing PyTorch/torchvision: {e}")
    sys.exit(1)


def main():
    py_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    tv_ver = torchvision.__version__
    cuda_ver = torch.version.cuda
    cudnn_ver = torch.backends.cudnn.version()
    cuda_avail = torch.cuda.is_available()

    print("==================================================")
    print("CUDA & PyTorch Environment Diagnostic")
    print("==================================================")
    print(f"Python version:      {py_ver}")
    print(f"PyTorch version:     {torch_ver}")
    print(f"Torchvision version: {tv_ver}")
    print(f"PyTorch CUDA build:  {cuda_ver}")
    print(f"cuDNN version:       {cudnn_ver}")
    print(f"CUDA available:      {cuda_avail}")

    if not cuda_avail:
        print("\n[ERROR] CUDA is not available in current PyTorch installation.")
        print("To install PyTorch with CUDA support, run:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    print(f"GPU device count:    {device_count}")
    print(f"GPU device 0:        {device_name}")
    print(f"Total VRAM:          {total_gb:.2f} GB")
    print(f"Free VRAM:           {free_gb:.2f} GB")
    print("--------------------------------------------------")

    print("Running GPU benchmark test...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    device = torch.device("cuda:0")
    start_time = time.perf_counter()

    with torch.no_grad():
        # 1. Matmul 2048x2048 on GPU
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        _ = torch.matmul(a, b)

        # 2. ResNet-50 forward pass without weights on batch 8x3x224x224
        model = models.resnet50(weights=None).to(device)
        model.eval()
        dummy_input = torch.randn(8, 3, 224, 224, device=device)
        _ = model(dummy_input)

        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time
    peak_mem_bytes = torch.cuda.max_memory_allocated(0)
    peak_mem_mb = peak_mem_bytes / (1024 ** 2)

    print("GPU test completed successfully!")
    print(f"Execution time:      {elapsed_time:.4f} s")
    print(f"Peak memory:         {peak_mem_mb:.2f} MB")
    print("==================================================")


if __name__ == "__main__":
    main()
