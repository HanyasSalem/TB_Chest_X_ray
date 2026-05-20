import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from model import TBCNN   # your trained model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("Using device:", device)

# -----------------------------
# Load model + weights
# -----------------------------
def load_model(weights_path):
    model = TBCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

# -----------------------------
# Preprocessing for a single image
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Predict function
# -----------------------------
def predict_image(model, image_path):
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)   # shape (1,1,224,224)

    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()

    label = "Tuberculosis" if prob > 0.5 else "Normal"
    return label, prob

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    model_path = r"D:\TB_project\TB_radiology_task\TB_Chest_X_ray\Checkpoints\final_model.pth"
    model = load_model(model_path)   #  saved weights

    test_image = r"D:\TB_project\TB_radiology_task\TB_Chest_X_ray\sample.jpg"
    label, prob = predict_image(model, test_image)

    print(f"Prediction: {label}")
    print(f"Probability: {prob:.4f}")
