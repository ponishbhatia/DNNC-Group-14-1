import torch
import numpy as np

from models.model import get_model
from config import DEVICE

def load_compressed_model(path):
    print("--- DEPLOYMENT SIMULATION ---")

    # 1. Load compressed weights
    loaded_weights = np.load(path)
    print("Successfully loaded compressed file into RAM.")

    # 2. Initialize fresh model
    deployment_model = get_model(DEVICE)

    # 3. Inject weights
    with torch.no_grad():
        for name, param in deployment_model.named_parameters():
            if name in loaded_weights:
                param.copy_(torch.tensor(loaded_weights[name]))

    deployment_model.eval()
    print("Weights successfully injected into the deployment model.")

    return deployment_model


def run_inference(model):
    # Dummy input
    dummy_image = torch.randn(1, 3, 32, 32).to(DEVICE)

    with torch.no_grad():
        output = model(dummy_image)
        predicted_class = torch.argmax(output, dim=1).item()

    print(f"\nSUCCESS: Model predicted Class {predicted_class}")