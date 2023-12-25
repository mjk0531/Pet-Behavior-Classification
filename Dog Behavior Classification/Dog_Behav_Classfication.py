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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('G:/Cloud/Dropbox/Nature Powered Fuel Cell/공대-의대 과제/Deep Learning/Data/DogMoveData.csv')

# Assuming you have loaded your data into a pandas dataframe called df
# The columns are: DogID, TestNum, t_sec, ABack_x, ABack_y, ABack_z, ANeck_x, ANeck_y, ANeck_z, GBack_x, GBack_y, GBack_z, GNeck_x, GNeck_y, GNeck_z, Task, Behavior_1, Behavior_2, Behavior_3, PointEvent
# The target column is Behavior_1, which has 10 possible classes
# The input columns are the sensor readings: ABack_x, ABack_y, ABack_z, ANeck_x, ANeck_y, ANeck_z, GBack_x, GBack_y, GBack_z, GNeck_x, GNeck_y, GNeck_z
# We'll use the DogID and TestNum columns to split the data into train and test sets

# Split the data into train and test sets
# We'll use 80% of the dogs for training and 20% for testing
# We'll also use the TestNum column to ensure that the same test is not in both sets
# This is to avoid data leakage and ensure a fair evaluation
train_dogs = df['DogID'].unique()[:int(len(df['DogID'].unique())*0.7)]
test_dogs = df['DogID'].unique()[int(len(df['DogID'].unique())*0.7):]

train_df = df[df['DogID'].isin(train_dogs)]
test_df = df[df['DogID'].isin(test_dogs)]

# Drop the columns that are not needed for the model
train_df = train_df.drop(['DogID', 'TestNum', 't_sec',
                         'Task', 'Behavior_2', 'Behavior_3', 'PointEvent'], axis=1)
test_df = test_df.drop(['DogID', 'TestNum', 't_sec', 'Task',
                       'Behavior_2', 'Behavior_3', 'PointEvent'], axis=1)

# Convert the dataframes to numpy arrays
X_train = train_df.drop('Behavior_1', axis=1).to_numpy()
y_train = train_df['Behavior_1'].to_numpy()
X_test = test_df.drop('Behavior_1', axis=1).to_numpy()
y_test = test_df['Behavior_1'].to_numpy()

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Convert the numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Create datasets from tensors
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

# Use a larger batch size if possible
# Use pin_memory=True to enable faster memory copy to GPU
# Use num_workers=2*number of GPUs to speed up data loading
train_loader = DataLoader(
    dataset=train_data, batch_size=64, shuffle=True, num_workers=6)
test_loader = DataLoader(
    dataset=test_data, batch_size=64, shuffle=False, num_workers=6)

# Define the CNN model


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # The input shape is (batch_size, 1, 12)
        # We'll use one channel to represent the sensor readings
        # We'll use a 1D convolution with a kernel size of 3 and 32 filters
        # Disable bias for convolutions directly followed by a batch norm
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        # The output shape is (batch_size, 32, 10)
        # We'll use a max pooling layer with a kernel size of 2 and a stride of 2
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # The output shape is (batch_size, 32, 5)
        # We'll use another 1D convolution with a kernel size of 3 and 64 filters
        # Disable bias for convolutions directly followed by a batch norm
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        # The output shape is (batch_size, 64, 3)
        # We'll use another max pooling layer with a kernel size of 2 and a stride of 2
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # The output shape is (batch_size, 64, 1)
        # We'll flatten the output for the fully connected layers
        self.flatten = nn.Flatten()
        # The output shape is (batch_size, 64)
        # We'll use a fully connected layer with 128 units
        self.fc1 = nn.Linear(64, 128)
        # The output shape is (batch_size, 128)
        # We'll use another fully connected layer with 10 units, corresponding to the number of classes
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Reshape the input to have one channel
        x = x.unsqueeze(1)
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

# Define the train function


def train_eval():
    # Create an instance of the model
    model = CNN()
    model = model.to(device)

    # Define the loss function
    # We'll use cross entropy loss, which is suitable for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    # We'll use Adam, which is a popular and effective optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Define the number of epochs
    # This is the number of times we'll loop over the entire training data
    epochs = 1

    # Train the model
    for epoch in range(epochs):
        # Initialize the running loss
        running_loss = 0.0
        # Loop over the batches of training data
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass: Compute predicted y by passing data to the model
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
            # Forward pass: Compute predicted y by passing data to the model
            output = model(data)
            # Get the predicted class by taking the argmax of the output
            pred = output.argmax(dim=1, keepdim=True)
            # Compare the predicted class with the true class
            # and count the number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Increment the total number of predictions
            total += target.size(0)

    # Print the accuracy
    print(f'Accuracy: {correct / total:.3f}')

# Define the main function


def main():
    # Call the train function
    train_eval()


# Check if the script is run directly
if __name__ == "__main__":
    # Call the main function
    main()
