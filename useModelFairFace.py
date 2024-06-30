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
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


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

def getLabels(image_path, df):

    temp = image_path.split('/')[-2:]
    temp = '/'.join(temp)

    label_to_lookup = temp
    result = df[df['file'] == label_to_lookup]

    # Check if there's any result
    if not result.empty:
        age = result['age'].iloc[0]
        gender = result['gender'].iloc[0]
        race = result['race'].iloc[0]
        return age, gender, race
    else:
        raise ValueError("No row found for label:", label_to_lookup)

# Function to plot stacked bar chart with extra space for title
def plot_stacked_bar(data, title, dir):
    # Ensure data is split correctly into categories and types
    categories = sorted(set(k.rsplit('_', 1)[0] for k in data.keys()))
    
    correct = [data.get(f"{category}_correct", 0) for category in categories]
    incorrect = [data.get(f"{category}_incorrect", 0) for category in categories]

    # Calculate total counts and proportions
    total = [c + i for c, i in zip(correct, incorrect)]
    correct_proportion = [c / t if t != 0 else 0 for c, t in zip(correct, total)]
    incorrect_proportion = [i / t if t != 0 else 0 for i, t in zip(incorrect, total)]

    # Create stacked bar chart
    barWidth = 0.5
    plt.bar(categories, correct_proportion, color='yellow', edgecolor='grey', width=barWidth, label='Correct')
    plt.bar(categories, incorrect_proportion, bottom=correct_proportion, color='red', edgecolor='grey', width=barWidth, label='Incorrect')

    plt.xlabel('Category')
    plt.ylabel('Proportion')
    plt.title(title, pad=20)  # Increase pad value to allocate more space for the title
    if "race" in title.lower() or "age" in title.lower():
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.legend()

    plt.tight_layout()  # Adjust layout to prevent clipping
    
    # Create directory if it doesn't exist
    dir = dir.replace("\n", " ")
    save_dir = f"BiasCharts/{dir}"
    os.makedirs(save_dir, exist_ok=True)
    
    name = ""

    # If "age" in title, plot age histogram
    if "age" in title.lower():
        name = "Age"

    # If "gender" in title, plot gender histogram
    elif "gender" in title.lower():
        name = "Gender"

    # If "race" in title, plot race histogram
    elif "race" in title.lower():
        name = "Race"

    # Save the figure
    plt.savefig(f"{save_dir}/{name}.png")
    # plt.show()

    # Clear the plot
    plt.clf()

def perform_tests(model_path, directory, info, dfInput):
    print()
    temp = model_path + "\n" + directory + "\n" + info
    print(temp)
    print()
    print(info)

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

    print("This script will assume that all images are REAL")

    # Define defaultdicts for each histogram
    hist_gender = defaultdict(int)
    hist_age = defaultdict(int)
    hist_race = defaultdict(int)

    # till = 200

    for image in tqdm(images, desc="Processing images"):
        # till -= 1
        # if till == 0:
        #     break
        age, gender, race = getLabels(image, df=dfInput)
        predicted_label = predict_image(image, model)

        # Increment counts based on predicted label and demographics
        if predicted_label == "real":
            hist_gender[f"{gender}_correct"] += 1
            hist_age[f"{age}_correct"] += 1
            hist_race[f"{race}_correct"] += 1
        elif predicted_label == "fake":
            hist_gender[f"{gender}_incorrect"] += 1
            hist_age[f"{age}_incorrect"] += 1
            hist_race[f"{race}_incorrect"] += 1
        else:
            raise ValueError("Invalid predicted label:", predicted_label)

    # Plotting the data
    plot_stacked_bar(hist_gender, info + "Gender", info)
    plot_stacked_bar(hist_age, info + "Age", info)
    plot_stacked_bar(hist_race, info + "Race", info)


    # Save the data to text file
    dir = info.replace("\n", " ")
    with open(f"BiasCharts/{dir}.txt", "w") as file:
        file.write(str(hist_gender) + "\n")
        file.write(str(hist_age) + "\n")
        file.write(str(hist_race) + "\n")


    # # Make a histogram on matplotlib
    # plt.bar(hist_gender.keys(), hist_gender.values())
    # plt.title("Gender")
    # plt.show()

    # plt.bar(hist_age.keys(), hist_age.values())
    # plt.title("Age")
    # plt.show()

    # plt.bar(hist_race.keys(), hist_race.values())
    # plt.title("Race")
    # plt.show()



    return hist_gender, hist_age, hist_race

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
    # sets = get_file_names("Bias Testing Set")
    sets = ["External Datasets for Bias Testing/fairface-img-margin125-trainval/train"]
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

                # if lines[1].strip() == "SD - REAL" and isSDRelated(rest) or lines[1].strip() == "GAN - REAL" and isGANRelated(rest):
                df = pd.read_csv("External Datasets for Bias Testing/fairface_label_train.csv")
                perform_tests(model, set_directory, info, df)
                count += 1
                print("=" * 20)
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
