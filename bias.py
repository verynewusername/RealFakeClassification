# Author: Efe Gorkem Sirin
# Date: 2024-05-09
# Description: This script is used to calculate the bias of the models

import os
import time
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models

# Define the CNN model ---- Taken from the training script
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

# Load and preprocess the image
def preprocess_image(image_path):
    # Directly taken from the training script
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    # Define image transformations
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

def get_file_names(folder_path):
    items = []
    for item in os.listdir(folder_path):
        # Ignore hidden files and folders
        if not item.startswith('.'):
            item_path = os.path.join(folder_path, item)
            # Check if it's a file
            if os.path.isfile(item_path):
                items.append(item_path)
            # Check if it's a folder
            elif os.path.isdir(item_path):
                items.append(item_path)
    return items

def perform_tests(model_path, directory, info):
    print()
    temp = model_path + "\n" + directory + "\n" + info
    print(temp)
    print()

    model = None

    # Check if this is CNN or Transfer learning
    if "CNN" in model_path:
        model = CNNClassifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    elif "TL" in model_path:
        model = models.resnet18(pretrained=True)

        # Freeze all the parameters in the pre-trained model
        for param in model.parameters():
            param.requires_grad = False

        # Modify the last fully connected layer to match the number of classes in your problem
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: fake and real

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

    assert model is not None, "Model is None!"

    images = get_file_names(directory)

    realCount = 0
    fakeCount = 0

    for image in images:
        predicted_label = predict_image(image, model)
        # print(image, predicted_label)
        if predicted_label == "real":
            realCount += 1
        elif predicted_label == "fake":
            fakeCount += 1
        else:
            raise ValueError("Invalid predicted label:", predicted_label)

    print("Real:", realCount, "Fake:", fakeCount)

    return realCount, fakeCount

def isGANRelated(dir_name):
    if (dir_name == "GAN female"
        or dir_name == "GAN male"
        or dir_name == "Real female"
        or dir_name == "Real male"):
        return True
    return False

def isSDRelated(dir_name):
    if (dir_name == "RSP_M" 
        or dir_name == "RSP_F"
        or dir_name == "RV_M"
        or dir_name == "RV_F"
        or dir_name == "Real female"
        or dir_name == "Real male"):
        return True
    return False

def main():
    file_names = get_file_names("out")
    sets = get_file_names("Bias Testing Set")
    # print(file_names)
    # print()
    count = 0

    for file_name in file_names:
        for set in sets:
            # Read the second and third lines of the file
            with open(file_name + "/details.txt", "r") as file:
                lines = file.readlines()
                # print(lines[1].strip())
                # print(lines[2].strip())

                info = lines[1].strip() + "\n" + lines[2].strip() + "\n"
                model = file_name + "/model.pth"
                set_directory = set

                rest = set_directory.split("/", 1)[-1]  # Split the string by "/", and take the last part
                # print(rest)  # Output will be "rest"

                if lines[1].strip() == "SD - REAL" and isSDRelated(rest) or lines[1].strip() == "GAN - REAL" and isGANRelated(rest):
                    perform_tests(model, set_directory, info)
                    count += 1
    print(count)


if __name__ == "__main__":
    # Get start time
    start_time = time.time()
    main()
    print("Done!")
    # Get end time
    end_time = time.time()
    # Calculate the time difference
    time_diff = end_time - start_time
    # print d h m s
    print("Processing time: ", time.strftime("%H:%M:%S", time.gmtime(time_diff)))
