from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
import io
from model import SimpleCNN


app = FastAPI()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=30)
model.load_state_dict(torch.load("simplecnn_valacc_71.87.pth", map_location=device))
model.to(device)
model.eval()

# Load class names
with open("classnames.txt", "r") as f:
    class_names = f.read().splitlines()


# Transform (same as training!)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.6511, 0.6167, 0.4756],
        std=[0.2929, 0.2870, 0.3412]
    )
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, dim=1)

    return {
        "class": class_names[idx.item()],
        "confidence": round(confidence.item(), 4)
    }

