from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torchvision import models

import torch.optim as optim

from evaluation import evaluate_model, plot_learning_curves,plot_confusion_matrix,compute_mAP, compute_confusion_matrix


import os

# Ensure the directory exists
output_dir = "../models/vehicle_dataset/vehicle_test"
os.makedirs(output_dir, exist_ok=True)


# Define parameters
image_size = 180  # Resize images to 224x224 as required by ResNet
batch_size = 32

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(root="../vehicle_dataset/dataset/vehicle_dataset/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(root="../vehicle_dataset/dataset/vehicle_dataset/val", transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)


# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Modify the final fully connected layer to match the number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Print the modified model architecture
print(model)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    best_model_wts = None
    best_accuracy = 0.0




    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0


        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        # Append to tracking lists
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(train_accuracies)


        print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_corrects += (preds == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)
          

        # Append to tracking lists
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model_wts = model.state_dict()

    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# Train the model
num_epochs = 20
trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs)
evaluate_model(trained_model, val_loader, criterion, device)
cm = compute_confusion_matrix(trained_model, val_loader, device)
print(cm)
plot_confusion_matrix(cm, class_names=train_dataset.classes)
plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)

# Save the trained model
torch.save(trained_model.state_dict(),"../models/vehicle_dataset/vehicle_test/best_val_loss.pt")

# Export to ONNX format
dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
torch.onnx.export(
    trained_model,
    dummy_input,
    "../models/vehicle_dataset/vehicle_test/vehicle_test.onnx",
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
