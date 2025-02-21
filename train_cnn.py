import numpy as np
import cv2
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Add tqdm for progress tracking

# Dummy CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: AI and Natural

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# FFT processing functions
def apply_fft(image):
    fft_channels = []
    for channel in cv2.split(image):
        fft = np.fft.fft2(channel)
        fft_shifted = np.fft.fftshift(fft)
        fft_channels.append(fft_shifted)
    return fft_channels

def filter_fft_percentage(fft_channels, percentage):
    filtered_fft = []
    for fft_data in fft_channels:
        magnitude = np.abs(fft_data)
        sorted_mag = np.sort(magnitude.flatten())[::-1]
        num_keep = int(len(sorted_mag) * percentage / 100)
        threshold = sorted_mag[num_keep - 1] if num_keep > 0 else 0
        mask = magnitude >= threshold
        filtered_fft.append(fft_data * mask)
    return filtered_fft

def extract_fft_features(image_paths, labels, percentage):
    features = []
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        fft_channels = apply_fft(image)
        filtered_fft = filter_fft_percentage(fft_channels, percentage)
        magnitude = np.abs(filtered_fft[0])  # Use the first channel's magnitude for simplicity
        features.append([image_path, labels[i], np.max(magnitude), np.mean(magnitude), np.count_nonzero(magnitude)])
    return features

# Load images and labels
def load_images_from_folders(folderA, folderB):
    image_paths = []
    labels = []
    for filename in os.listdir(folderA):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_paths.append(os.path.join(folderA, filename))
            labels.append(0)  # Natural
    for filename in os.listdir(folderB):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_paths.append(os.path.join(folderB, filename))
            labels.append(1)  # AI
    return image_paths, labels

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    patience = 5
    trigger_times = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Add progress bar
            for images, labels in tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch+1}/{num_epochs}'):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                trigger_times = 0
            elif phase == 'val':
                trigger_times += 1
                if trigger_times >= patience:
                    print('Early stopping!')
                    model.load_state_dict(best_model_wts)
                    return model

    model.load_state_dict(best_model_wts)
    return model

# Main function
def main():
    folderA = 'Natural_images'  # Use relative path
    folderB = 'AI_images'       # Use relative path
    image_paths, labels = load_images_from_folders(folderA, folderB)

    # Extract FFT features
    percentage = 10.0  # Example percentage
    features = extract_fft_features(image_paths, labels, percentage)

    # Save features to CSV
    with open('fft_features.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image', 'Label', 'Max Magnitude', 'Mean Magnitude', 'Non-zero Count'])
        csv_writer.writerows(features)

    print("FFT features extracted and saved to fft_features.csv.")

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create datasets and dataloaders
    train_dataset = ImageDataset(train_paths, train_labels, transform=transform)
    val_dataset = ImageDataset(val_paths, val_labels, transform=transform)
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
    }

    # Initialize model, criterion, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)

    # Save the model weights
    torch.save(model.state_dict(), 'cnn_model_weights.pth')
    print("Model training completed and weights saved.")

if __name__ == "__main__":
    main()
