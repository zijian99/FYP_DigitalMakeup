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

<br><br>

## FLowchart of Digital Makeup Project:

![Flowchart Digital Makeup](https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/b6e1a1ae-8b2b-4f15-bada-8554232ed0cb)



<br><br>

## Installation Guide

### Tools Required
In this project, Python 3.9 and Anaconda 2022 is used.<br>
Just installing Python 3.9 is sufficient but I prefer to use python in Anaconda.<br>

#### (Optional)
CUDA can be installed to have a ***faster speed*** at performing semantic segmentation.

Can be downloaded from the link [here](https://developer.nvidia.com/cuda-11-8-0-download-archive)
<br>

### Library Packages Version
| Library       | Version       | 
| ------------- |:-------------:| 
| torch         | 2.1.0         | 
| torchaudio    | 2.1.0         | 
| torchvision   | 0.16.0        | 
| numpy         | 1.24.0        | 
| opencv-python | 4.8.1.78      | 

------

***Do note that:***
1. Virtual Environment need to be created to install library packages to prevent dependencies conflict of existing packages
2. This installation guide is for **WINDOWS**, for **MacOS and Linux** should have similar steps for installation, please refer to online sources for similar installation commands.

<br>

#### Before following the installation guide below, open ***Command Prompt*** and make sure that you are current file path is inside this code folder<br>

Example:<br>
![Capture](https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/f6b014f3-02c3-466d-9a6a-14830c0f08e4)


## Non-CUDA Installation
### Option I : If you are using Anaconda......
#### 1. Creating Virtual Environment with Anaconda
```bash
conda create --name dmakeup python=3.9
```
#### 2. To activate Virtual Environment
```bash
conda activate dmakeup
```
#### 3. Installing Required Library Packages 
```bash
pip install -r requirement.txt
```
<br>



### Option II : If you are using Python 3.9 only......
#### 1. Creating Virtual Environment with Python
```bash
# Refer to the link here
https://www.geeksforgeeks.org/create-virtual-environment-using-venv-python/ 
```
#### 2. To activate Virtual Environment
```bash
.\venv\Scripts\activate
```
#### 3. Installing Required Library Packages 
```bash
pip install -r requirement.txt
```
<br>


## CUDA Installation 
**(Refer to Guide above for similar steps)**
1. Create and activate virtual environment
2. Click [here](https://pytorch.org/get-started/locally/) to get the command for installing pytorch library packages
3. Install numpy and opencv-python library packages via these commands
```bash
pip install numpy
```
```bash
pip install opencv-python==4.8.1.78
```
4. The code should be ready to run! 
<br><br><br>

## References
The source code of semantic segmentation models is referenced from this [repo](https://github.com/zllrunning/face-parsing.PyTorch)

<br><br>

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/zj99)

<img src="https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/f0259962-dac3-4d64-8269-c2917fa3c39f" width="170" height="170">

