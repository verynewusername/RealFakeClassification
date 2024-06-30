import torch
from torch import nn
from torchviz import make_dot

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

# Instantiate the model and create a dummy input
model = CNNClassifier()
x = torch.randn(1, 3, 256, 256)  # Example input tensor

# Perform a forward pass to create the computation graph
y = model(x)

# Visualize the model graph using torchviz
make_dot(y, params=dict(model.named_parameters())).render("cnn_torchviz", format="png")

# import torch.onnx

# import torch
# from torch import nn
# from torchviz import make_dot

# # Define the CNN model
# class CNNClassifier(nn.Module):
#     def __init__(self):
#         super(CNNClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 64 * 64, 128)
#         self.fc2 = nn.Linear(128, 2)  # 2 classes: fake and real

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Instantiate the model and create a dummy input
# model = CNNClassifier()
# x = torch.randn(1, 3, 256, 256)  # Example input tensor

# # Save the model to an ONNX file
# torch.onnx.export(model, x, "cnn_model.onnx")

# # Visualize using Netron
# import netron
# netron.start("cnn_model.onnx")

# =====================================================================================
# import torch
# from torch import nn
# from torchviz import make_dot

# # Define the CNN model
# class CNNClassifier(nn.Module):
#     def __init__(self):
#         super(CNNClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 64 * 64, 128)
#         self.fc2 = nn.Linear(128, 2)  # 2 classes: fake and real

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)  # flatten all dimensions except batch
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Instantiate the model and create a dummy input
# model = CNNClassifier()
# x = torch.randn(1, 3, 256, 256)  # Example input tensor

# # Generate the visualization
# y = model(x)
# dot = make_dot(y, params=dict(model.named_parameters()))

# # Save the visualization to a file
# dot.format = 'png'
# dot.render('cnn_visualization')
