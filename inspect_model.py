import torch

MODEL_PATH = "/home/mnegru/Adelina/PASTIS/results/baseline_france_128/model.pth.tar"

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

print("\n=== CHECKPOINT TYPE ===")
print(type(checkpoint))

print("\n=== CHECKPOINT KEYS ===")
if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        print(key)

# Incercam sa gasim automat state_dict-ul
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # uneori checkpoint-ul este chiar un state_dict cu nume de layere
        state_dict = checkpoint
else:
    state_dict = checkpoint

print("\n=== MODEL LAYERS / PARAMETERS ===")

for name, tensor in state_dict.items():
    if torch.is_tensor(tensor):
        print(f"{name:80s} {tuple(tensor.shape)}")
    else:
        print(f"{name:80s} {type(tensor)}")

print("\n=== TOTAL PARAMETERS ===")

total_params = 0
for name, tensor in state_dict.items():
    if torch.is_tensor(tensor):
        total_params += tensor.numel()

print(f"Total parameters: {total_params:,}")