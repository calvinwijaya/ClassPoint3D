# DGCNN-Classifier
This repository is personal made GUI for point cloud semantic segmentation with DGCNN algorithm. The model provided is trained to classify point cloud into three classes: Ground (2), Vegetation (5), and Building (6) only. 

## Installation
To install the GUI, just ensure you have an Anaconda or Miniconda installed, find the bat directory of conda (open prompt and write ```where conda```, copy-paste it into ```CONDA_PATH``` in ```dgcnn.bat```. Click it twice to install the venv and it will close automatically. Click it twice again to install all library needed and Done!.

## How to use:
1. After installed, just click twice ```dgcnn.bat``` and it will open the GUI.
2. Browse the point cloud data (las format) to classify in *Load Point Cloud to Clasify*.
3. Browse the DGCNN model. In the repo, I provide the DGCNN model that trained to classify 3 classes inside the `model` folder.
4. Browse the ouput folder in *Select Output Directory*.
5. Start! the process.

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/8ab73120-eaad-40a1-a24e-de0d9243c944" />

Sample result:

<img width="700" alt="Screenshot 2025-07-13 102828" src="https://github.com/user-attachments/assets/eb58647b-474b-4c15-b418-a6ae804308d0" />


Performance: 1 km x 1 km area with 17.000.000 points will classified in approximately 5 minutes.
