import platform
import torch

#Select the current device and return it
def get_device():
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current device set to:{device}")
    return device

#Display current device information
def display_device_info(device):
    #if torch.cuda.is_available():
    if device==torch.device("cuda"):
        num_cuda_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_cuda_devices}")
        for device_id in range(num_cuda_devices):
            print(f"CUDA_Device{device_id}:")
            print("-------------")
            print(f"Name: {torch.cuda.get_device_name(device_id)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(device_id) / (1024 ** 3):.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(device_id) / (1024 ** 3):.2f} GB")
            print(f"Device Capability: {torch.cuda.get_device_capability(device_id)}")
    else:
        print(f"Processor: {platform.processor()}")
        print(f"System: {platform.system()}")
        print(f"Machine: {platform.machine()}")