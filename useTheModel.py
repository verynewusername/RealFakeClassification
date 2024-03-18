import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn


# Define the CNN model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: fake and real

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the pre-trained model
model = CNNClassifier()
model.load_state_dict(torch.load("out/CNN-1-2024-03-15 04:53:56/model.pth"))
model.eval()

# Define image transformations
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image_transforms(image).unsqueeze(0)
    return image

# Perform inference on a single image
def predict_image(image_path, model):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_label = "fake" if predicted.item() == 0 else "real"
    return class_label

# Example usage
image_path = "/Users/efe/Desktop/GAN FACES/archivep1/1m_faces_04_05_06_07/1m_faces_04_05_06_07/1m_faces_05/0CHU0FP0PG.jpg"
predicted_label = predict_image(image_path, model)
print("Predicted label:", predicted_label)
