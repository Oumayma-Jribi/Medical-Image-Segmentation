import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Directory where the NIfTI files are downloaded and stored
downloaded_files_dir = 'downloaded_zip_files'

# Get all .nii.gz files in the downloaded files directory
nii_file_paths = [os.path.join(downloaded_files_dir, f) for f in os.listdir(downloaded_files_dir) if f.endswith('.nii.gz')]

# Iterate through each NIfTI file in the list
for f in nii_file_paths:
    if not os.path.exists(f):
        print(f"File not found: {f}")
    else:
        # Load the NIfTI image
        nifti_img = nib.load(f)
        data = nifti_img.get_fdata()

        # Print information about the image
        print(f"NIfTI image shape: {data.shape}")
        print(f"Affine transformation matrix:\n{nifti_img.affine}")

        # Preprocess the data (normalize)
        def preprocess_data(data):
            normalized_data = data / np.max(data)  # Normalize the data to the range [0, 1]
            return normalized_data

        normalized_data = preprocess_data(data)

        # Function to visualize 3D slices
        def visualize_3d_slices(data, n_slices=5):
            if data.ndim != 3:
                print("Data is not 3D.")
                return

            for i in range(min(n_slices, data.shape[2])):
                plt.figure(figsize=(10, 5))
                plt.imshow(data[:, :, i], cmap='gray')  # Visualizing the i-th slice
                plt.title(f"Slice {i + 1} of 3D NIfTI Image")
                plt.axis('off')
                plt.show()

        # Function to visualize 4D slices
        def visualize_4d_slices(data, n_slices=5):
            if data.ndim != 4:
                print("Data is not 4D.")
                return

            for i in range(min(n_slices, data.shape[3])):
                plt.figure(figsize=(10, 5))
                slice_index = data.shape[2] // 2  # Taking the middle slice of the 3rd dimension
                plt.imshow(data[:, :, slice_index, i], cmap='gray')  # Visualizing the i-th frame
                plt.title(f"Frame {i + 1} of 4D NIfTI Image - Middle Slice")
                plt.axis('off')
                plt.show()

        # Visualize based on data dimensionality
        if normalized_data.ndim == 3:
            visualize_3d_slices(normalized_data, n_slices=5)
        elif normalized_data.ndim == 4:
            visualize_4d_slices(normalized_data, n_slices=5)
        else:
            print("Unsupported data dimensionality.")
