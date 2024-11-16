import kagglehub
from glob import glob
import h5py
import matplotlib.pyplot as plt



# Open the HDF5 file
file_path = r'C:\Users\R I B\.cache\kagglehub\datasets\anhoangvo\acdc-dataset\versions\1\ACDC_preprocessed\ACDC_training_volumes\patient001_frame01.h5'
with h5py.File(file_path, 'r') as h5_file:
    # Access the 'image' and 'scribble' datasets
    images = h5_file['image']
    scribbles = h5_file['scribble']
    
    # Check that the number of images and scribbles match
    print("Images shape:", images.shape)
    print("Scribbles shape:", scribbles.shape)
    
    # Display the first three images with their corresponding scribble masks overlaid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        # Display the image
        axes[i].imshow(images[i], cmap='gray')  # Show the base image
        # Overlay the scribble (segmentation) mask with transparency
        axes[i].imshow(scribbles[i], cmap='jet', alpha=0.5)  # Adjust alpha for transparency
        axes[i].axis('off')  # Turn off axis for each subplot
        axes[i].set_title(f"Image {i + 1} with Scribble Overlay")
    
    plt.tight_layout()
    plt.show()