import mxnet as mx
from PIL import Image
import io

rec_path = "CASIA-WebFace/train.rec"
idx_path = "CASIA-WebFace/train.idx"

# Open RecordIO dataset
record = mx.recordio.MXIndexedRecordIO(
    idx_path,
    rec_path,
    "r"
)

# Number of records
print("Number of records:", len(record.keys))

# Get first valid record index
index = record.keys[490624]

print("Reading record index:", index)

# Read record
packed = record.read_idx(index)

# Unpack record
header, img = mx.recordio.unpack(packed)

print("Label:", header.label)
print("Image bytes:", len(img))

# Convert image bytes to PIL image
image = Image.open(io.BytesIO(img)).convert("RGB")

print("Image size:", image.size)

# Save image
image.save("test_face.jpg")

print("Saved test_face.jpg")