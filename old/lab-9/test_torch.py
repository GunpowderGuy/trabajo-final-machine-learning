import torch

print("✅ PyTorch is installed.")
print(f"🧠 Version: {torch.__version__}")
print(f"💻 CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"🖥️ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Running on CPU.")
