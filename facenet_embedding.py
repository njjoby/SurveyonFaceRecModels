import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


# --------------------------------------------------
# 1. Select device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 3. Load pretrained FaceNet model
# --------------------------------------------------

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)


# --------------------------------------------------
# 4. Load image
# --------------------------------------------------

image_path = "input/face.jpeg"

image = Image.open(image_path).convert("RGB")


# --------------------------------------------------
# 5. Detect and align face
# --------------------------------------------------

face = mtcnn(image)

if face is None:
    print("No face detected.")
    exit()

face = face.unsqueeze(0).to(device)


# --------------------------------------------------
# 6. Generate FaceNet embedding
# --------------------------------------------------

with torch.no_grad():

    embedding = facenet(face)

    # L2 normalization
    embedding = embedding / embedding.norm(p=2, dim=1, keepdim=True)


# --------------------------------------------------
# 7. Convert to NumPy
# --------------------------------------------------

embedding = embedding.cpu().numpy()[0]


# --------------------------------------------------
# 8. Display embedding
# --------------------------------------------------

print("\nFaceNet Embedding:")
print(embedding)

print("\nEmbedding dimension:")
print(embedding.shape)

print("\nEmbedding norm:")
print(np.linalg.norm(embedding))


# --------------------------------------------------
# 9. Save embedding
# --------------------------------------------------

np.save("embeddings/face_embedding.npy", embedding)

print("\nEmbedding saved successfully.")