import torch
from torchvision import transforms, models
from PIL import Image
import os

num_classes = 5
model_path = "scin_model.pth"

# Check if model exists
if not os.path.exists(model_path):
    print("=" * 70)
    print("⚠️  Model file not found!")
    print("=" * 70)
    print(f"\nThe model file '{model_path}' is missing.")
    print("\nTo create/get the model, you have these options:")
    print("\n1. Run the training script to create a placeholder model:")
    print("   python src/imageanalysis/train_skin_model.py")
    print("\n2. Download a pre-trained model from:")
    print("   - Kaggle skin disease datasets")
    print("   - ISIC Archive (International Skin Imaging Collaboration)")
    print("   - HAM10000 dataset")
    print("\n3. Train your own model with a skin disease dataset")
    print("\n" + "=" * 70)
    raise FileNotFoundError(
        f"Model file '{model_path}' not found.\n"
        "Please run: python src/imageanalysis/train_skin_model.py"
    )

# Load your pre-trained model
print(f"Loading model from: {model_path}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully!")

# Preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

img_path = "/home/sangeethagsk/agent_bootcamp/HandWritingAnalysis/src/Psoriasis.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image file not found: {img_path}")

img = Image.open(img_path).convert("RGB")
img_t = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(img_t)
    _, predicted = torch.max(outputs, 1)

print("Predicted class ID:", predicted.item())

labels = {
    0: "Acne",
    1: "Eczema",
    2: "Psoriasis",
    3: "Urticaria",
    4: "Rosacea"
}

print("Predicted Disease:", labels[predicted.item()])

descriptions = {
    "Acne": "A common skin condition that occurs when hair follicles are clogged with oil and dead skin cells.",
    "Eczema": "A condition that makes your skin red and itchy, often chronic.",
    "Psoriasis": "A skin disease that causes red, scaly patches, often itchy or painful.",
    "Urticaria": "Also known as hives, red, itchy welts caused by an allergic reaction.",
    "Rosacea": "A chronic skin condition that causes redness and visible blood vessels on the face."
}

disease = labels[predicted.item()]
print(f"Disease: {disease}")
print(f"Description: {descriptions[disease]}")
