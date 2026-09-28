import os

import torch
import numpy as np

from PIL import Image

from facenet_pytorch import MTCNN

from sphereface import SphereFace


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 2. Face detector
# --------------------------------------------------

mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    keep_all=False,
    device=device
)


# --------------------------------------------------
# 3. SphereFace model
# --------------------------------------------------

model = SphereFace(
    embedding_size=512
)


# --------------------------------------------------
# 4. Load pretrained weights
# --------------------------------------------------

WEIGHTS_PATH = "sphereface_weights.pth"

if os.path.exists(WEIGHTS_PATH):

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    print("SphereFace weights loaded.")

else:

    print(
        "WARNING: SphereFace weights not found."
    )

    print(
        "The model is randomly initialized."
    )


model = model.to(device)
model.eval()


# --------------------------------------------------
# 5. Load image
# --------------------------------------------------

image_path = "input/face.jpeg"

image = Image.open(
    image_path
).convert("RGB")


# --------------------------------------------------
# 6. Detect and align face
# --------------------------------------------------

face = mtcnn(image)


if face is None:

    print("No face detected.")

    exit()


# --------------------------------------------------
# 7. Add batch dimension
# --------------------------------------------------

face = face.unsqueeze(0)

face = face.to(device)


# --------------------------------------------------
# 8. Generate embedding
# --------------------------------------------------

with torch.no_grad():

    embedding = model(face)


# --------------------------------------------------
# 9. L2 normalize
# --------------------------------------------------

embedding = embedding / embedding.norm(
    p=2,
    dim=1,
    keepdim=True
)


# --------------------------------------------------
# 10. Convert to NumPy
# --------------------------------------------------

embedding = embedding.cpu().numpy()[0]


# --------------------------------------------------
# 11. Display result
# --------------------------------------------------

print()
print("SphereFace embedding:")
print(embedding)

print()
print("Embedding dimension:")
print(embedding.shape)

print()
print("Embedding norm:")
print(np.linalg.norm(embedding))


# --------------------------------------------------
# 12. Create directory
# --------------------------------------------------

os.makedirs(
    "embeddings",
    exist_ok=True
)


# --------------------------------------------------
# 13. Save embedding
# --------------------------------------------------

output_path = (
    "embeddings/sphereface_embedding.npy"
)

np.save(
    output_path,
    embedding
)


print()
print(
    "Embedding saved to:",
    output_path
)