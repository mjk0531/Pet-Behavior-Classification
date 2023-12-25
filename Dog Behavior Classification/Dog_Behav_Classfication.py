# Import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('G:/Cloud/Dropbox/Nature Powered Fuel Cell/공대-의대 과제/Deep Learning/Data/DogMoveData.csv')

# Extract the data from the columns for the features and the behaviors
# and store them in separate variables
feature = df[["DogID", "t_sec", "ABack_x", "ABack_y", "ABack_z", "ANeck_x", "ANeck_y",
              "ANeck_z", "GBack_x", "GBack_y", "GBack_z", "GNeck_x", "GNeck_y", "GNeck_z"]].to_numpy()
target = df[["Behavior_1", "Behavior_2", "Behavior_3"]].to_numpy()

# Load and preprocess dataset
# Assuming your dataset is stored in a numpy array called data
# and has the shape (N, 16), where N is the number of samples
# and 16 is the number of features (3 for DogID, TestNum, t_sec
# and 13 for the accelerometer and gyroscope data)
# and the target is stored in a numpy array called target
# and has the shape (N, 3), where 3 is the number of behaviors

# Split the dataset into train and test sets
train_ratio = 0.8  # You can change this value
train_size = int(train_ratio * len(feature))
test_size = len(feature) - train_size
train_feature, test_feature = feature[:train_size], feature[train_size:]
train_target, test_target = target[:train_size], target[train_size:]

le = LabelEncoder()
train_target_0 = le.fit_transform(train_target[:, 0])
train_target_1 = le.transform(train_target[:, 1])
train_target_2 = le.transform(train_target[:, 2])
train_target = np.stack(
    [train_target_0, train_target_1, train_target_2], axis=1)
test_target_0 = le.transform(test_target[:, 0])
test_target_1 = le.transform(test_target[:, 1])
test_target_2 = le.transform(test_target[:, 2])
test_target = np.stack([test_target_0, test_target_1, test_target_2], axis=1)

# Convert the numpy arrays to PyTorch tensors
train_feature = torch.from_numpy(train_feature).float()
test_feature = torch.from_numpy(test_feature).float()
train_target = torch.from_numpy(train_target).long()
test_target = torch.from_numpy(test_target).long()

# Define the transformations for the data
# You can apply any transformations you want, such as normalization, augmentation, etc.
# Here we just convert the data to a 4D tensor of shape (N, C, H, W)
# where C is the number of channels (1 for grayscale), H and W are the height and width of the image
# We reshape the 16 features into a 4x4 image
transform = transforms.Compose([
    transforms.Lambda(lambda x: x.view(-1, 1, 2, 7)),
])

# Apply the transformations to the train and test data
train_feature = transform(train_feature)
test_feature = transform(test_feature)

# Create the train dataset
train_dataset = TensorDataset(train_feature, train_target)
test_dataset = TensorDataset(test_feature, test_target)

# Define the batch size
batch_size = 256  # You can change this value

# Create the data loaders for the train and test sets
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(96, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # Apply the first convolution and pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        # Apply the second convolution and pooling layers
        x = self.pool2(F.relu(self.conv2(x)))
        # Flatten the output
        x = self.flatten(x)
        # Apply the first fully connected layer
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer
        # Use a softmax activation function on the output layer
        # so that the outputs sum to 1 and can represent probabilities
        return F.log_softmax(self.fc2(x), dim=1)


# Create an instance of the model
model = CNN()
model = model.to(device)

# Define the loss function
# We'll use cross entropy loss, which is suitable for multi-class classification
criterion = nn.L1Loss()

# Define the optimizer
# We'll use Adam, which is a popular and effective optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Define the number of epochs
# This is the number of times we'll loop over the entire training data
epochs = 10


if __name__ == "__main__":
    # Train the model
    for epoch in range(epochs):
        # Initialize the running loss
        running_loss = 0.0
        # Loop over the batches of training data
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass: Compute predicted y by passing data to the model
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            # Compute loss
            loss = criterion(output, target)
            # Zero the gradients
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Update the model parameters
            optimizer.step()
            # Update the running loss
            running_loss += loss.item()
            # Print the statistics every 200 batches
            if (batch_idx + 1) % 200 == 0:
                print(
                    f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 200:.3f}')
                # Reset the running loss
                running_loss = 0.0

    # Evaluate the model on the test data
    # Set the model to evaluation mode
    model.eval()
    # Initialize the number of correct predictions
    correct = 0
    # Initialize the total number of predictions
    total = 0
    # Loop over the batches of test data
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            # Forward pass: Compute predicted y by passing data to the model
            output = model(data)
            # Get the predicted class by taking the argmax of the output
            pred = output.argmax(dim=1, keepdim=True)
            # Convert the target tensor from one-hot vectors to class indices
            target = target.argmax(dim=1, keepdim=True)
            # Compare the predicted class with the true class
            # and count the number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Increment the total number of predictions
            total += target.size(0)

    # Print the accuracy
    print(f'Accuracy: {correct / total:.3f}')
