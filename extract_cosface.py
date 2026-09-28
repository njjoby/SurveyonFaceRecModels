import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from cosface_model import CosFaceEmbeddingModel


IMAGE_PATH = "input/face.jpeg"
OUTPUT_PATH = "embeddings/cosface_embedding.npy"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])
image_tensor = transform(image).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CosFaceEmbeddingModel().to(device).eval()

with torch.no_grad():
    embedding = model(image_tensor.to(device))[0].cpu().numpy().astype(np.float32)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.save(OUTPUT_PATH, embedding)
print("Embedding shape:", embedding.shape)
print("Embedding norm:", np.linalg.norm(embedding))
print("Saved:", OUTPUT_PATH)
