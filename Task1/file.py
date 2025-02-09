"""
Script to train a neural classifier on a selected subset of generated CAPTCHA images (from both the easy and hard subset).
"""

import os
import csv
import random
import shutil
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F

# Defining variables
TRAIN_TEST_RATIO = 0.8      # 80% training, 20% testing
NUM_SAMPLES = 200           # Number of samples from each set
LEARNING_RATE = 0.001       # Learning rate
NUM_EPOCHS = 25             # Number of epochs
PLOT_COUNT = 20             # Just to number the plots for saving

# Paths to the datasets
hard_set_dir = './Task0/HardSet'
easy_set_dir = './Task0/EasySet'
dataset_dir = './Task1/DataSet'

def create_subset(hard_set_dir, easy_set_dir, dataset_dir, num_samples=NUM_SAMPLES):
    """
    Create a subset of images from hard and easy sets.
    """
    print("Creating Dataset...")
    # Ensure the subset directory exists and is empty
    os.makedirs(dataset_dir, exist_ok=True)
    for f in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, f)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)

    # Copy selected images from the hard set
    hard_words = [f for f in os.listdir(hard_set_dir) if os.path.isdir(os.path.join(hard_set_dir, f))]
    for word in hard_words:
        word_dir = os.path.join(hard_set_dir, word)
        images = [f for f in os.listdir(word_dir) if os.path.isfile(os.path.join(word_dir, f))]
        selected_images = random.sample(images, min(num_samples, len(images)))

        # Create a directory for the word in the subset directory
        subset_word_dir = os.path.join(dataset_dir, word)
        os.makedirs(subset_word_dir, exist_ok=True)

        for image in selected_images:
            shutil.copy(os.path.join(word_dir, image), os.path.join(subset_word_dir, image))

    # Copy selected images from the easy set
    easy_words = [f for f in os.listdir(easy_set_dir) if os.path.isdir(os.path.join(easy_set_dir, f))]
    for word in easy_words:
        word_dir = os.path.join(easy_set_dir, word)
        images = [f for f in os.listdir(word_dir) if os.path.isfile(os.path.join(word_dir, f))]
        selected_images = random.sample(images, min(num_samples, len(images)))

        # Create a directory for the word in the subset directory
        subset_word_dir = os.path.join(dataset_dir, word)
        os.makedirs(subset_word_dir, exist_ok=True)

        for image in selected_images:
            shutil.copy(os.path.join(word_dir, image), os.path.join(subset_word_dir, image))
    print("Dataset created.")

class CaptchaDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.images = []
        self.labels = []
        for label in os.listdir(dataset):
            label_dir = os.path.join(dataset, label)
            if os.path.isdir(label_dir):
                for img in os.listdir(label_dir):
                    self.images.append(os.path.join(label_dir, img))
                    self.labels.append(label)
        self.unique_labels = list(set(self.labels))
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(img_name)
        if image is None:
            raise ValueError(f"Image not found or unable to read: {img_name}")
        image = cv2.resize(image, (128, 128))
        label = self.labels[idx]
        label = self.label_to_index[label]
        if self.transform:
            image = self.transform(image)
        return image, label

def preprocess_data(dataset_dir):
    """
    Preprocess the data: split dataset and create DataLoader.
    """
    dataset = CaptchaDataset(dataset_dir, transform=transforms.ToTensor())
    train_size = int(TRAIN_TEST_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, dataset.labels, dataset

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    """
    Train the model and record the training history.
    """
    print("Training Model...")
    history = {'train_loss': [], 'val_loss': []}
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(test_loader.dataset)
        history['val_loss'].append(val_loss)
        model.train()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
    print("Model trained.")
    return history

def evaluate_model(model, test_loader, criterion, dataset):
    """
    Evaluate the model.
    """
    print("Evaluating Model...")
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    results = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)

            for true_label, pred_label in zip(labels.cpu(), predicted.cpu()):
                results.append({
                    'answer': dataset.unique_labels[true_label],
                    'prediction': dataset.unique_labels[pred_label],
                    'correct': true_label == pred_label
                })

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    results_dir = './Task1'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['answer', 'prediction', 'correct'])
        writer.writeheader()
        writer.writerows(results)

    accuracy = 100 * correct / total
    loss = running_loss / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}')
    print("Model evaluated.")
    return accuracy, loss


# Create a subset of images
create_subset(hard_set_dir, easy_set_dir, dataset_dir)
train_loader, test_loader, labels, dataset = preprocess_data(dataset_dir)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build and train the model
model = CNNModel(num_classes=len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model and record the history
history = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)

# Evaluate the model
accuracy, loss = evaluate_model(model, test_loader, criterion, dataset)

# Plot the training history
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.suptitle(f'Epoch: {NUM_EPOCHS}, Sample Count: {NUM_SAMPLES}, Learning Rate: {LEARNING_RATE} \n Accuracy: {accuracy:.2f}, Loss: {loss:.4f}')
plt.savefig(f'./Task1/Plots/{PLOT_COUNT}.png')
plt.show()
