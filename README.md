# FYP Digital Makeup

Perform Semantic Segmentation and Reinhard Color Transfer to retrieve facial features and transfer makeup from source image to destination image<br><br>
<img src="https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/542392bc-67b3-4750-9bbe-8f7eb78dc17e" width="170" height="170"><br><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/zijian99/FYP_DigitalMakeup/blob/main/LICENSE)


## Features:
1. **Semantic Segmentation:**<br>
Perform semantic segmentation on image in order to retrieve the color mapping of facial features<br><br>
2. **Color Transfer:**<br>
Perform color transfer from source image to destination image<br><br>

**Based on the 2 features above, we are able to perform simple makeup transfer.**


## Installation Guide

### Tools Required
In this project, Python 3.9 and Anaconda 2022 is used.<br>
Just installing Python 3.9 is sufficient but I prefer to use python in Anaconda.<br>

#### Optional
CUDA can be installed to have a faster speed at performing semantic segmentation

Can be downloaded from the link [here](https://developer.nvidia.com/cuda-11-8-0-download-archive)


### Library Packages Version
| Library       | Version       | 
| ------------- |:-------------:| 
| torch         | 2.1.0         | 
| torchaudio    | 2.1.0         | 
| torchvision   | 0.16.0        | 
| numpy         | 1.24.0        | 
| opencv-python | 4.8.1.78      | 

------

***Virtual Environment need to be created to install library packages to prevent dependencies conflict of existing packages***<br><br>

### Option I : If you are using Anaconda......
#### Creating Virtual Environment with Anaconda
```bash
conda create --name dmakeup python=3.9
```
#### Installing Required Library Packages 
```bash
pip install -r requirement.txt
```
#### To activate Virtual Environment
```bash
conda activate dmakeup
```
<br>

------

### Option II : If you are using Python 3.9 only......
#### Creating Virtual Environment with Python
```bash
# Refer to the link here
https://www.geeksforgeeks.org/create-virtual-environment-using-venv-python/ 
```
#### Installing Required Library Packages 
```bash
pip install -r requirement.txt
```
#### To activate Virtual Environment
```bash
.\venv\Scripts\activate
```

<br>
<img src="https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/f0259962-dac3-4d64-8269-c2917fa3c39f" width="170" height="170">

