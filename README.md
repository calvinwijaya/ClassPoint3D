# Class Point 3D
This repository is personal made GUI for point cloud semantic segmentation with DGCNN and XGBoost algorithm. The model provided is trained to classify point cloud into three classes: Ground (2), Vegetation (5), and Building (6) only. 

## Installation
To install the GUI, just ensure you have an Anaconda or Miniconda installed, find the bat directory of conda (open prompt and write ```where conda```, copy-paste it into ```CONDA_PATH``` in ```dgcnn.bat```. Click it twice to install the venv and it will close automatically. Click it twice again to install all library needed and Done!.

## How to use:
1. After installed, just click twice ```classpoint3d.bat``` and it will open the GUI.
2. In *Project Configuration*, choose classification algorithm to use. There are 2 options: (1) Deep Learning (with DGCNN) or (2) Machine Learning (with XGBoost)
3. Browse the point cloud data (.las/.laz format) or folder containing point cloud to classify in *Point Cloud Input*. The input point cloud will preview in Input Point Clodu (RGB) (for folder input, only the first point cloud will show).
4. Browse the trained model (.t7 for deep learning and .pkl for machine learning). In the repo, I provide the 3 DGCNN model and 1 XGBoost model that trained to classify 3 classes inside the `model` folder.
5. Browse the ouput folder in *Select Output Directory*.
6. Start! the process. The result will saved into the output folder and preview in Classication Result (for folder input, only the first classification result will show).

<img width="1692" height="956" alt="Class Point 3D Interface" src="https://github.com/user-attachments/assets/a92c039b-7d8c-4be3-8e84-8b847d37f322" />

Sample result:

<img width="1692" height="956" alt="Class Point 3D Result" src="https://github.com/user-attachments/assets/0f2d7b09-b89b-46c6-b6b3-a5e92e581213" />
