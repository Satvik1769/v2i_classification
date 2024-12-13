import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from cleanlab.outlier import OutOfDistribution
from cleanlab.rank import find_top_issues
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_names[idx]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # For ResNet50
])

truck_dir = "../vehicle_dataset/dataset/vehicle_dataset/train/truck"

dataset = CustomImageDataset(truck_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=50, shuffle=False)



model = timm.create_model('resnet50', pretrained=True, num_classes=0)  
model.eval()  

# Function to extract features from the images
def embed_images(model, dataloader):
    feature_embeddings = []
    with torch.no_grad():
        for data in dataloader:
            images, _ = data
            embeddings = model(images)
            feature_embeddings.extend(embeddings.cpu().numpy())
    return np.array(feature_embeddings)

# Extract feature embeddings
feature_embeddings = embed_images(model, dataloader)
print(f"Feature embeddings shape: {feature_embeddings.shape}")



# Initialize Cleanlab's OutOfDistribution detector
ood = OutOfDistribution()

# Fit the outlier detector and calculate outlier scores
ood_scores = ood.fit_score(features=feature_embeddings)

top_outlier_indices = find_top_issues(quality_scores=ood_scores, top=15)

def visualize_outliers(outlier_indices, dataset):
    for idx in outlier_indices:
        image, image_name = dataset[idx]
        plt.imshow(image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
        plt.title(f"Outlier: {image_name}")
        plt.show()

visualize_outliers(top_outlier_indices, dataset)



outlier_image_names = [dataset.image_names[idx] for idx in top_outlier_indices]

for outlier in outlier_image_names:
    os.remove(os.path.join(truck_dir, outlier))
    print(f"Removed outlier image: {outlier}")
