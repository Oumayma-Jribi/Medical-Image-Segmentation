import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize


# Directory paths (replace with actual directory paths)
training_volumes_dir = 'C:\\Users\\R I B\\.cache\\kagglehub\\datasets\\anhoangvo\\acdc-dataset\\versions\\1\\ACDC_preprocessed\\ACDC_training_volumes'  # Path to your training volumes directory
testing_volumes_dir = 'C:\\Users\\R I B\\.cache\\kagglehub\\datasets\\anhoangvo\\acdc-dataset\\versions\\1\\ACDC_preprocessed\\ACDC_testing_volumes'    # Path to your testing volumes directory

def resize_image(image, target_shape=(256, 256)):
    """
    Resize the image to the target shape (height, width).
    :param image: The input image (e.g., 2D or 3D image array).
    :param target_shape: The desired output shape (height, width).
    :return: Resized image
    """
    # If the image is 3D (e.g., with depth), resize each slice individually
    if image.ndim == 3:
        # Resize each slice in the 3rd dimension (assuming (C, H, W) or (H, W, D))
        resized_image = np.stack([resize(slice, target_shape, mode='constant', preserve_range=True) for slice in image], axis=0)
    else:
        resized_image = resize(image, target_shape, mode='constant', preserve_range=True)
    
    return resized_image


def load_h5_data_from_directory(directory, target_shape=(256, 256)):
    images = []
    scribbles = []
    
    # Iterate through all .h5 files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.h5'):
            file_path = os.path.join(directory, file_name)
            
            # Open the .h5 file and read the datasets
            with h5py.File(file_path, 'r') as f:
                # Access the 'image' and 'scribble' datasets
                image = f['image'][:]
                scribble = f['scribble'][:]
                
                # Resize the image and scribble if necessary
                image = resize_image(image, target_shape)
                scribble = resize_image(scribble, target_shape)  # Resize scribble to the same shape as image

                # Print the shape of the datasets (for debugging)
                print(f"File: {file_name}")
                print(f"Resized image shape: {image.shape}")
                print(f"Resized scribble shape: {scribble.shape}")

                images.append(image)
                scribbles.append(scribble)

    # Convert lists to numpy arrays
    images = np.array(images)
    scribbles = np.array(scribbles)
    
    return images, scribbles



# Example to load and resize data from the training and testing directories
X_train, y_train = load_h5_data_from_directory(training_volumes_dir, target_shape=(256, 256))
X_test, y_test = load_h5_data_from_directory(testing_volumes_dir, target_shape=(256, 256))

# Check if images have the same shape, if not we may need to resize or pad them
print("Training images shape:", X_train.shape)
print("Testing images shape:", X_test.shape)

# Normalize the images to range [0, 1]
X_train = X_train.astype(np.float32) / np.max(X_train)
X_test = X_test.astype(np.float32) / np.max(X_test)

# Optionally split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# PyTorch Dataset class
class CardiacDataset(Dataset):
    def __init__(self, images, scribbles, transform=None):
        self.images = images
        self.scribbles = scribbles
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        scribble = self.scribbles[idx]

        # Add an additional dimension for channels if it's not present
        image = image.unsqueeze(0)  # Assuming images are 3D (C, H, W)

        return image, scribble

# Create DataLoader for training and validation
train_dataset = CardiacDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = CardiacDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

test_dataset = CardiacDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


def dice_coeff(pred, target, smooth=1e-6):
    """
    Dice Similarity Coefficient for segmentation tasks.
    
    :param pred: The predicted mask (tensor)
    :param target: The ground truth mask (tensor)
    :param smooth: A small constant to avoid division by zero.
    
    :return: Dice score
    """
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    
    # Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# For multi-class Dice score (e.g., 3 classes in segmentation)
def multi_class_dice(pred, target, num_classes=3, smooth=1e-6):
    """
    Compute multi-class Dice score for segmentation.
    
    :param pred: The predicted mask (tensor) with shape (batch_size, num_classes, H, W)
    :param target: The ground truth mask (tensor) with shape (batch_size, H, W)
    :param num_classes: Number of classes in the segmentation task
    :param smooth: Small constant to avoid division by zero
    :return: Average Dice score across all classes
    """
    dice_scores = []
    # Iterate over each class
    for class_idx in range(num_classes):
        # Convert predictions to one-hot encoding
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        # Compute Dice coefficient for the current class
        dice = dice_coeff(pred_class, target_class, smooth)
        dice_scores.append(dice)
    
    return torch.mean(torch.stack(dice_scores))

# Define the U-Net model (simplified version as before)
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):  # 1 input channel, 3 output channels (endocardium, epicardium, RV)
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(64, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2

# Create the model
model = UNet(in_channels=1, out_channels=3).cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        # Zero gradients, perform a backward pass, and update weights
        optimizer.zero_grad()
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Evaluation function
# Evaluation function
def evaluate(model, dataloader, num_classes=3):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            
            # Apply argmax to get class predictions
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Stack all predictions and labels into tensors
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute the multi-class Dice score
    dice_score = multi_class_dice(all_preds, all_labels, num_classes=num_classes)
    print(f"Dice Similarity Coefficient: {dice_score:.4f}")
    
evaluate(model, val_loader, num_classes=3)
evaluate(model, test_loader, num_classes=3)