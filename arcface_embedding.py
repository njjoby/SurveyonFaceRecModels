import os
import cv2
import numpy as np

from insightface.app import FaceAnalysis


# ==================================================
# 1. Check GPU
# ==================================================

providers = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

try:
    import onnxruntime as ort

    available_providers = ort.get_available_providers()

    print("Available ONNX providers:")
    print(available_providers)

except Exception as e:

    print("Could not check ONNX Runtime providers.")
    print(e)

    available_providers = []


# ==================================================
# 2. Select execution provider
# ==================================================

if "CUDAExecutionProvider" in available_providers:

    print("Using NVIDIA GPU")

    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

    ctx_id = 0

else:

    print("CUDA not available. Using CPU.")

    providers = [
        "CPUExecutionProvider"
    ]

    ctx_id = -1


# ==================================================
# 3. Initialize ArcFace model
# ==================================================

print()
print("Loading ArcFace model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=providers
)


# ==================================================
# 4. Prepare model
# ==================================================

app.prepare(
    ctx_id=ctx_id,
    det_size=(640, 640)
)

print("ArcFace model loaded.")


# ==================================================
# 5. Load image
# ==================================================

image_path = "input/face.jpeg"

image = cv2.imread(
    image_path
)


if image is None:

    raise FileNotFoundError(
        f"Could not find image: {image_path}"
    )


# ==================================================
# 6. Detect faces
# ==================================================

faces = app.get(
    image
)


print()
print("Number of faces detected:", len(faces))


# ==================================================
# 7. Check detection
# ==================================================

if len(faces) == 0:

    raise RuntimeError(
        "No face detected in the image."
    )


# ==================================================
# 8. Select largest face
# ==================================================

def face_area(face):

    bbox = face.bbox

    width = bbox[2] - bbox[0]

    height = bbox[3] - bbox[1]

    return width * height


largest_face = max(
    faces,
    key=face_area
)


# ==================================================
# 9. Get ArcFace embedding
# ==================================================

embedding = largest_face.embedding


print()
print("Original embedding:")
print(embedding)


# ==================================================
# 10. Check embedding dimension
# ==================================================

print()
print(
    "Embedding dimension:",
    embedding.shape
)


# ==================================================
# 11. L2 normalization
# ==================================================

embedding = embedding / np.linalg.norm(
    embedding
)


# ==================================================
# 12. Check norm
# ==================================================

print()
print(
    "Normalized embedding norm:",
    np.linalg.norm(embedding)
)


# ==================================================
# 13. Save embedding
# ==================================================

os.makedirs(
    "embeddings",
    exist_ok=True
)


output_path = (
    "embeddings/arcface_embedding.npy"
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


# ==================================================
# 14. Display first few dimensions
# ==================================================

print()
print("First 10 dimensions:")

print(
    embedding[:10]
)