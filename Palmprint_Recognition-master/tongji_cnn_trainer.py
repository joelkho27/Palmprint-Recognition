"""
Tongji Palmprint Database Handler and CNN Training Script

This script handles the Tongji Palmprint Database and trains a CNN model for palmprint recognition.
Tongji Database: 20,000+ images from 300 palms with multiple lighting conditions.

Download the database from: 
- Contact Tongji University or check their computer vision lab website
- Some versions may be available through academic databases

Expected structure after download:
Tongji_Palmprint/
    ├── Subject_001/
    │   ├── palm_001_001.bmp
    │   ├── palm_001_002.bmp
    │   └── ...
    ├── Subject_002/
    └── ...
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import json

class PalmPrintDataset(Dataset):
    """Custom Dataset for Palmprint Images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not load image {img_path}")
            # Return a blank image if loading fails
            image = np.zeros((128, 128), dtype=np.uint8)
        
        # Convert to PIL format for transforms
        image = image.astype(np.uint8)
        
        if self.transform:
            # Convert to 3-channel for standard transforms
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = self.transform(image)
        else:
            # Convert to tensor manually
            image = torch.from_numpy(image).float() / 255.0
            image = image.unsqueeze(0)  # Add channel dimension
        
        label = self.labels[idx]
        return image, label

class PalmPrintCNN(nn.Module):
    """CNN Architecture for Palmprint Recognition"""
    
    def __init__(self, num_classes):
        super(PalmPrintCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def load_tongji_dataset(data_dir):
    """
    Load Tongji Palmprint Database
    Expected structure:
    data_dir/
        ├── Subject_001/
        │   ├── palm_001_001.bmp
        │   └── ...
        ├── Subject_002/
        └── ...
    """
    image_paths = []
    labels = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory {data_dir} not found!")
        print("Please download the Tongji Palmprint Database and extract it.")
        return [], []
    
    subject_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    subject_dirs.sort()
    
    print(f"Found {len(subject_dirs)} subjects in dataset")
    
    for label, subject_dir in enumerate(subject_dirs):
        subject_path = os.path.join(data_dir, subject_dir)
        
        # Get all image files in subject directory
        image_files = [f for f in os.listdir(subject_path) 
                      if f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg'))]
        
        for image_file in image_files:
            image_path = os.path.join(subject_path, image_file)
            image_paths.append(image_path)
            labels.append(label)
    
    print(f"Loaded {len(image_paths)} images from {len(set(labels))} subjects")
    return image_paths, labels

def load_existing_dataset():
    """Load the existing palmprint dataset as fallback"""
    train_dir = "./Palmprint/Palmprint/training"
    test_dir = "./Palmprint/Palmprint/testing"
    
    image_paths = []
    labels = []
    
    # Load training data
    if os.path.exists(train_dir):
        train_files = os.listdir(train_dir)
        for file in train_files:
            if file.endswith('.bmp'):
                subject_id = int(file.split('-')[0]) - 1  # Convert to 0-based
                image_paths.append(os.path.join(train_dir, file))
                labels.append(subject_id)
    
    # Load testing data
    if os.path.exists(test_dir):
        test_files = os.listdir(test_dir)
        for file in test_files:
            if file.endswith('.bmp'):
                subject_id = int(file.split('-')[0]) - 1  # Convert to 0-based
                image_paths.append(os.path.join(test_dir, file))
                labels.append(subject_id)
    
    print(f"Loaded existing dataset: {len(image_paths)} images from {len(set(labels))} subjects")
    return image_paths, labels

def create_data_transforms():
    """Create data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Train the CNN model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0
    best_model_path = f"best_palmprint_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': model.classifier[-1].out_features
            }, best_model_path)
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        scheduler.step()
    
    return train_losses, val_losses, train_accuracies, val_accuracies, best_model_path

def main():
    """Main training function"""
    print("Tongji Palmprint Database CNN Training")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use existing dataset (current palmprint database)
    print("Using existing palmprint dataset for CNN training...")
    image_paths, labels = load_existing_dataset()
    
    if not image_paths:
        # Try to load Tongji dataset as fallback
        tongji_data_dir = "./Tongji_Palmprint"
        if os.path.exists(tongji_data_dir):
            print("Fallback: Loading Tongji Palmprint Database...")
            image_paths, labels = load_tongji_dataset(tongji_data_dir)
        else:
            print("To use additional Tongji dataset:")
            print("1. Download from Tongji University")
            print("2. Extract to ./Tongji_Palmprint/")
            print("3. Run this script again")
    
    if not image_paths:
        print("No dataset found! Please ensure you have palmprint data available.")
        return
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Number of classes: {len(set(labels))}")
    
    # Create transforms
    train_transform, val_transform = create_data_transforms()
    
    # Create datasets
    train_dataset = PalmPrintDataset(train_paths, train_labels, train_transform)
    val_dataset = PalmPrintDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create model
    num_classes = len(set(labels))
    model = PalmPrintCNN(num_classes).to(device)
    
    print(f"Model created with {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs, best_model_path = train_model(
        model, train_loader, val_loader, num_epochs=50, device=device
    )
    
    print(f"\nTraining completed!")
    print(f"Best model saved as: {best_model_path}")
    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accs,
        'val_accuracies': val_accs,
        'num_classes': num_classes,
        'dataset_size': len(image_paths)
    }
    
    history_file = f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved as: {history_file}")

if __name__ == "__main__":
    main()