# Exploring-Vision-Transformers-and-Hybrid-Architectures-for-Medical-Image-Segmentation


## Team Information
- **Team Name**: SO Segment
- **Team Members**:
  - Oumaima Jribi (Neptune Code: [EBVZ24])
  - Selim Ben Haj Braiek (Neptune Code: [N5OQI9])


## Project Description
This project focuses on the automatic segmentation of cardiac structures from medical imaging data using NIfTI files. The primary goal is to compare the performance of various automatic segmentation methods on the left ventricular endocardium, epicardium, and right ventricular endocardium during both end-diastolic and end-systolic phases.

The project includes tools for loading, preprocessing, and visualizing 3D and 4D NIfTI images. By utilizing these tools, we aim to provide insights into the effectiveness of different segmentation techniques and contribute to improving medical image analysis.

## Functions of Files in the Repository
- `visualize_nifti.py`: This script contains functions to:
  - Load NIfTI files using the `nibabel` library.
  - Normalize the image data for better visualization.
  - Visualize slices from 3D or frames from 4D NIfTI images.
