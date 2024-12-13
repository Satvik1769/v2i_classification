import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from evaluation import evaluate_model, plot_learning_curves, plot_confusion_matrix, compute_mAP, compute_confusion_matrix

output_dir = "../models/vehicle_dataset/vehicle_test"
os.makedirs(output_dir, exist_ok=True)

image_size = 180
batch_size = 32

train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root="../vehicle_dataset/dataset/vehicle_dataset/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(root="../vehicle_dataset/dataset/vehicle_dataset/val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(512 * (image_size // 32) * (image_size // 32), 1024), 
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNNModel(num_classes=num_classes)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    best_model_wts = None
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

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

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

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

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model_wts = model.state_dict()

    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    model.load_state_dict(best_model_wts)
    return model

num_epochs = 20
trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs)
evaluate_model(trained_model, val_loader, criterion, device)
cm = compute_confusion_matrix(trained_model, val_loader, device)
print(cm)
plot_confusion_matrix(cm, class_names=train_dataset.classes)
plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)
mAP, APs = compute_mAP(trained_model, val_loader, device)
print(mAP)
print(APs)

torch.save(trained_model.state_dict(),"../models/vehicle_dataset/vehicle_test/best_val_loss_cnn.pt")

dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
torch.onnx.export(
    trained_model,
    dummy_input,
    "../models/vehicle_dataset/vehicle_test/vehicle_test_cnn.onnx",
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
