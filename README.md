# Class Point 3D
This repository is personal made GUI for point cloud semantic segmentation with DGCNN algorithm. The model provided is trained to classify point cloud into three classes: Ground (2), Vegetation (5), and Building (6) only. 

## Installation
To install the GUI, just ensure you have an Anaconda or Miniconda installed, find the bat directory of conda (open prompt and write ```where conda```, copy-paste it into ```CONDA_PATH``` in ```dgcnn.bat```. Click it twice to install the venv and it will close automatically. Click it twice again to install all library needed and Done!.

## How to use:
1. After installed, just click twice ```classpoint3d.bat``` and it will open the GUI.
2. Browse the point cloud data (las format) to classify in *Load Point Cloud to Clasify*.
3. Browse the DGCNN model. In the repo, I provide the DGCNN model that trained to classify 3 classes inside the `model` folder.
4. Browse the ouput folder in *Select Output Directory*.
5. Start! the process.

src="<img width="1692" height="956" alt="Class Point 3D Interface" src="https://github.com/user-attachments/assets/c16001e3-89dd-42ce-bdbf-0adf66ad2947" />

Sample result:

src="<img width="1692" height="956" alt="Class Point 3D Result" src="https://github.com/user-attachments/assets/fb7a35a3-7917-4d8e-8348-a410635d9dd7" />

Performance: 1 km x 1 km area with 17.000.000 points will classified in approximately 5 minutes.
