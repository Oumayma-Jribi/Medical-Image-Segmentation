import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

nii_file_path = ['C:\\Users\\R I B\\Desktop\\Medical Segmentation\\Medical-Image-Segmentation\\DCM03-OH-AL_V2_12.nii','C:\\Users\\R I B\\Desktop\\Medical Segmentation\\Medical-Image-Segmentation\\patient001_4d.nii.gz','C:\\Users\\R I B\\Desktop\\Medical Segmentation\\Medical-Image-Segmentation\\patient002_4d.nii.gz','C:\\Users\\R I B\\Desktop\\Medical Segmentation\\Medical-Image-Segmentation\\patient003_4d.nii.gz']  # Update this path 

for f in nii_file_path:
    if not os.path.exists(f):
        print(f"File not found: {f}")
    else:
        nifti_img = nib.load(f)

        data = nifti_img.get_fdata()

        print(f"NIfTI image shape: {data.shape}")
        print(f"Affine transformation matrix:\n{nifti_img.affine}")

        def preprocess_data(data):
            normalized_data = data / np.max(data)  
            return normalized_data

        normalized_data = preprocess_data(data)

        def visualize_3d_slices(data, n_slices=5):
            if data.ndim != 3:
                print("Data is not 3D.")
                return

            for i in range(min(n_slices, data.shape[2])):
                plt.figure(figsize=(10, 5))
                plt.imshow(data[:, :, i], cmap='gray')  
                plt.title(f"Slice {i + 1} of 3D NIfTI Image")
                plt.axis('off')
                plt.show()

        def visualize_4d_slices(data, n_slices=5):
            if data.ndim != 4:
                print("Data is not 4D.")
                return

            for i in range(min(n_slices, data.shape[3])):
                plt.figure(figsize=(10, 5))
                slice_index = data.shape[2] // 2  
                plt.imshow(data[:, :, slice_index, i], cmap='gray') 
                plt.title(f"Frame {i + 1} of 4D NIfTI Image - Middle Slice")
                plt.axis('off')
                plt.show()

        if normalized_data.ndim == 3:
            visualize_3d_slices(normalized_data, n_slices=5)
        elif normalized_data.ndim == 4:
            visualize_4d_slices(normalized_data, n_slices=5)
        else:
            print("Unsupported data dimensionality.")