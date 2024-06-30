
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models

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
        self.feature_maps = x  # Save feature maps
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained model
model = CNNClassifier()
# model.load_state_dict(torch.load("/Users/efe/Documents/Github/RealFakeClassification/out/GAN CNN-1-2024-03-15 04:53:56/model.pth"))
# model.load_state_dict(torch.load("/Users/efe/Documents/Github/RealFakeClassification/out/SD CNN 2024-05-01 21:06:18/model.pth",map_location=torch.device('cpu')))

model = models.resnet18(pretrained=True)
model.load_state_dict(torch.load("/Users/efe/Documents/Github/RealFakeClassification/out/GAN TL18-2024-03-17 18:34:35/model.pth",map_location=torch.device('cpu')))
model.eval()

# Define image transformations
image_transforms = transforms.Compose([
    # transforms.Resize((256, 256)),
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
    return class_label, outputs

# Implement Grad-CAM
def grad_cam(model, image_path, target_layer):
    image = preprocess_image(image_path)
    image.requires_grad = True

    # Forward pass
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    # Zero gradients
    model.zero_grad()
    
    # Retain gradients of the feature maps
    model.feature_maps.retain_grad()

    # Backward pass
    one_hot_output = torch.zeros(outputs.shape)
    one_hot_output[0][predicted] = 1
    outputs.backward(gradient=one_hot_output)

    # Get the gradients of the target layer
    gradients = model.feature_maps.grad.data

    # Get the feature maps from the target layer
    feature_maps = model.feature_maps.data

    # Pool the gradients across the width and height dimensions
    weights = torch.mean(gradients, dim=(2, 3))[0, :]

    # Compute the weighted sum of the feature maps
    grad_cam_map = torch.zeros(feature_maps.shape[2:], dtype=torch.float32)
    for i, w in enumerate(weights):
        grad_cam_map += w * feature_maps[0, i, :, :]

    # Apply ReLU to the grad_cam_map
    grad_cam_map = torch.relu(grad_cam_map)

    # Normalize the grad_cam_map
    grad_cam_map -= grad_cam_map.min()
    grad_cam_map /= grad_cam_map.max()

    return grad_cam_map, image, outputs

# Visualize the Grad-CAM heatmap
def visualize_grad_cam(grad_cam_map, image_path, prediction, output):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = np.array(image)

    heatmap = grad_cam_map.numpy()
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.transpose(heatmap, (1, 0))

    heatmap = np.uint8(255 * heatmap)
    heatmap = np.transpose(heatmap, (1, 0))
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize((image.shape[1], image.shape[0]))
    heatmap = np.array(heatmap)

    heatmap = np.expand_dims(heatmap, axis=2)
    heatmap = np.tile(heatmap, (1, 1, 3))

    superimposed_img = heatmap * 0.4 + image

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img.astype('uint8'))
    plt.title('Grad-CAM')
    plt.text(10, 30, f'Prediction: {prediction}', color='white', fontsize=12, backgroundcolor='black')
    # plt.text(10, 50, f'Output: {output[0].detach().numpy()}', color='white', fontsize=12, backgroundcolor='black')

    plt.savefig("gradcam.png")
    plt.show()

# Example usage
image_path = "/Users/efe/Documents/Github/RealFakeClassification/Dataset/supplementary/real/069005.png"
# image_path = "/Users/efe/Documents/Github/RealFakeClassification/Dataset/supplementary/gan/069054.png"
prediction, output = predict_image(image_path, model)
grad_cam_map, image, outputs = grad_cam(model, image_path, target_layer='conv2')
visualize_grad_cam(grad_cam_map, image_path, prediction, outputs)
