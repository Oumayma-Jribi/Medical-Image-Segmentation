# Exploring Vision Transformers and Hybrid Architectures for Medical Image Segmentation

## Team Information
- **Team Name**: SO Segment
- **Team Members**:
  - Oumaima Jribi (Neptune Code: [EBVZ24])
  - Selim Ben Haj Braiek (Neptune Code: [N5OQI9])

## Project Description
This project focuses on the automatic segmentation of cardiac structures from medical imaging data using NIfTI files. The primary goal is to compare the performance of various automatic segmentation methods on the left ventricular endocardium, epicardium, and right ventricular endocardium during both end-diastolic and end-systolic phases.

The project includes tools for loading, preprocessing, and visualizing 3D and 4D NIfTI images. By utilizing these tools, we aim to provide insights into the effectiveness of different segmentation techniques and contribute to improving medical image analysis.

## Functions of Files in the Repository
- `data_acquisition.py`: This script contains functions to:
  - Load NIfTI files using the `nibabel` library.
  - Normalize the image data for better visualization.
  - Visualize slices from 3D or frames from 4D NIfTI images.

--- 
## Find 2nd milestone results under 2nd milestone folder

# MRI Image Segmentation with U-Net

## Project Overview
This project focuses on the segmentation of MRI images using a U-Net architecture. Our primary goal is to segment specific regions of the heart from MRI scans, using masks provided in the ACDC dataset. The implementation involves adapting a pre-existing U-Net model for multi-class segmentation, improving data handling, and addressing challenges in evaluating the model's performance.

## Dataset
- **Dataset Name**: Automated Cardiac Diagnosis Challenge (ACDC).
- **Source**: The dataset was downloaded from Kaggle using the Kaggle API. The original dataset can be found [here](https://www.kaggle.com/).
- **Contents**: The dataset includes MRI images, segmentation masks, and scribble annotations.
- **Scope**: For this project, we exclusively used the segmentation masks, which label each pixel in the image with one of four classes:
  - 0: Background
  - 1: Right Ventricle (RV)
  - 2: Myocardium
  - 3: Left Ventricle (LV)

## Model
### U-Net Architecture
The U-Net is a convolutional neural network architecture widely used for biomedical image segmentation. It features an encoder-decoder structure:
1. **Encoder**: Captures contextual information through downsampling (contracting path).
2. **Decoder**: Performs upsampling and combines it with features from the encoder to provide precise localization (expansive path).
3. **Skip Connections**: Merge features from encoder layers with decoder layers to improve spatial resolution.

**Specific Details for This Project**:
- The base model code was sourced from [this repository](https://github.com/](https://github.com/arp95/cnn_architectures_image_segmentation ) (originally targeting binary segmentation of MRI masks).
- We adapted the model to handle multi-class segmentation with four classes, using categorical cross-entropy as the loss function.

## Challenges
1. **Accuracy Misrepresentation**:
   - Initial runs of the model achieved high training accuracy (95%-99%). However, this was primarily due to the model correctly predicting the dominant background class (class 0), which comprises the majority of pixels in the masks.
   - **Solution**: We adjusted the accuracy calculation to exclude background pixels, focusing only on the actual mask classes (1, 2, 3).
   
2. **Voluminous Data**:
   - The dataset is large, so we trained with a batch size of 4 and limited initial experiments to 10 epochs.

## Data Handling
- The dataset is loaded and preprocessed using a custom `H5FolderDataset` class to handle HDF5 files.
- Data loading and acquisition were updated from the first milestone. The updated notebook can be found in:
  - `2nd milestone/Data Acquisition and Loading.ipynb`.

## Loss Function
We used **Categorical Cross-Entropy** as the loss function:
- **Why Categorical Cross-Entropy?**
  - It is well-suited for multi-class classification problems.
  - Measures the difference between the predicted probability distribution and the true class distribution for each pixel.
  - Encourages the model to assign high probability to the correct class while penalizing incorrect predictions.

## Future Work
For the next milestone, we plan to:
1. **Add Transformations for Data Augmentation**:
   - Apply techniques such as rotation, flipping, and elastic deformations to improve generalization.
   
2. **Explore Further Improvements**:
   - Experiment with learning rate scheduling.
   - Introduce post-processing methods to refine segmentation outputs.
   - Evaluate additional metrics (e.g., Dice Coefficient or IoU) for a better understanding of segmentation performance.

## Training Details
- **Batch Size**: 4
- **Epochs**: 10
- **Optimization Algorithm**: Adam
- **Learning Rate**: Default Adam optimizer learning rate.
## Results
1.	Last epoch shows preliminary results of 90% accuracy for both training and validation.
2.	Running the last saved model on the test set shows accuracy of 94%.
## How to Use
1. Clone this repository.
2. Download the ACDC dataset using the Kaggle API or from the official source.
3. Preprocess the dataset using the notebook in `2nd milestone/Data Acquisition and Loading.ipynb`.
4. Train the model using the training script (TBA).
5. Evaluate the model on test data and visualize predictions with masks.
