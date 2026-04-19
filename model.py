import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import io

# Load class index
with open("class_index.json") as f:
    data = json.load(f)

IDX_TO_CLASS = {int(k): v for k, v in data["idx_to_class"].items()}
IDX_TO_NAME  = {int(k): v for k, v in data["idx_to_name"].items()}
NUM_CLASSES  = data["num_classes"]

# Build same model architecture
def build_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = build_model()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
model.to(device)
print(f"Model loaded on {device}")

# Preprocessing — same as training val transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_bytes: bytes) -> dict:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    # Top 3 predictions
    top3_probs, top3_idxs = torch.topk(probs, 3)
    top3 = [
        {
            "class_code": IDX_TO_CLASS[idx.item()],
            "class_name": IDX_TO_NAME[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        }
        for prob, idx in zip(top3_probs, top3_idxs)
    ]

    return {
        "predicted_class":  top3[0]["class_name"],
        "predicted_code":   top3[0]["class_code"],
        "confidence":       top3[0]["confidence"],
        "top3":             top3,
        "all_probs": {
            IDX_TO_NAME[i]: round(probs[i].item() * 100, 2)
            for i in range(NUM_CLASSES)
        }
    }
