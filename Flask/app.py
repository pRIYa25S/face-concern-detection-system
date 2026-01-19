from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask_cors import CORS

print("🔥 THIS app.py FILE IS RUNNING 🔥")

# ------------------ Flask App ------------------
app = Flask(__name__)
CORS(app)


# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Classes ------------------
classes = [
    'class0_normal',
    'class1_acne',
    'class2_wrinkles',
    'class3_Eczema',
    'class4_Rosacea',
    'class5_dark_spots'
]

# ------------------ Model ------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(classes))
model.to(device)

MODEL_PATH = r"C:\Users\91787\face_concern_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded successfully")

# ------------------ Transform ------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------ Routes ------------------
@app.route("/")
def home():
    return "HOME ROUTE OK - THIS IS THE CORRECT FILE"


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        pred = outputs.argmax(1).item()
        confidence = torch.softmax(outputs, dim=1)[0][pred].item()

    return jsonify({
        "predicted_class": classes[pred],
        "confidence": round(confidence * 100, 2)
    })

# ------------------ Run Server ------------------
if __name__ == "__main__":
    app.run(debug=True)
