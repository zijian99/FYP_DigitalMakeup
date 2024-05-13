# FYP Digital Makeup

Perform Semantic Segmentation and Reinhard Color Transfer to retrieve facial features and transfer makeup from source image to destination image<br><br>
[My FYP Report](https://dr.ntu.edu.sg/handle/10356/175268)<br><br>

<img src="https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/542392bc-67b3-4750-9bbe-8f7eb78dc17e" width="170" height="170"><br><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/zijian99/FYP_DigitalMakeup/blob/main/LICENSE)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Features:
1. **Semantic Segmentation:**<br>
Perform semantic segmentation on image in order to retrieve the color mapping of facial features<br><br>
2. **Color Transfer:**<br>
Perform color transfer from source image to destination image<br><br>

**Based on the 2 features above, we are able to perform simple makeup transfer.**

<br><br>


## FLowchart of Digital Makeup Project:

<div align="center">
  
![Flowchart Digital Makeup](https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/cc663a45-ca6b-4396-84e5-0451637e1d86)

</div>



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

<br>

> [!NOTE] 
> 1.  Virtual Environment need to be created to install library packages to prevent dependencies conflict of existing packages
> 2.  This installation guide is for **WINDOWS**, for **MacOS and Linux** should have similar steps for installation, please refer to online sources for similar installation commands.
 
<br>
<br>

> [!IMPORTANT]
> #### Before following the installation guide below, open ***Command Prompt*** and make sure that you are current file path is inside this code folder<br>
> Example:<br>
> ![Capture](https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/f6b014f3-02c3-466d-9a6a-14830c0f08e4)

<br>
<br>

### Non-CUDA Installation
#### Option I : If you are using Anaconda......
##### 1. Creating Virtual Environment with Anaconda
```bash
conda create --name dmakeup python=3.9
```
##### 2. To activate Virtual Environment
```bash
conda activate dmakeup
```
##### 3. Installing Required Library Packages 
```bash
pip install -r requirement.txt
```
<br>



#### Option II : If you are using Python 3.9 only......
##### 1. Creating Virtual Environment with Python
```bash
# Refer to the link here
https://www.geeksforgeeks.org/create-virtual-environment-using-venv-python/ 
```
##### 2. To activate Virtual Environment
```bash
.\venv\Scripts\activate
```
##### 3. Installing Required Library Packages 
```bash
pip install -r requirement.txt
```
<br>


### CUDA Installation 
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

## Example Digital Makeup Results
<table>

<tr>
<th>&nbsp;</th>
<th>Hair</th>
<th>Lip</th>
</tr>


<!-- Line 1: Original Input -->
<tr>
<td><em>Original Input</em></td>
<td><img src="https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/e1cfd574-f511-4e91-877f-9fac2f753ffe" height="256" width="256" alt="Original Input"></td>
<td><img src="https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/2bfac976-8e39-4900-8b1d-fc23659ad712" height="256" width="256" alt="Original Input"></td>
</tr>

<!-- Line 3: Color -->

<tr>
<td>Color</td>
<td><img src="https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/d3ae3142-d866-420d-9efa-ac1eac532172" height="256" width="256" alt="Color"></td>
<td><img src="https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/0bc5c913-8dd1-4642-8611-81c73745a452" height="256" width="256" alt="Color"></td>
</tr>

</table>

## References
1. **BiSeNet Semantic Segmentation** [Official GitHub Repository](https://github.com/zllrunning/face-parsing.PyTorch) 

2. **SCANet Semantic Segmentation** [Official GitHub Repository](https://github.com/Seungeun-Han/SCANet_Real-Time_Face_Parsing_Using_Spatial_and_Channel_Attention)
<br>

> [!NOTE]
> Access [here](https://drive.google.com/drive/u/0/folders/188a_pHxfhAn4z2kwoP9tWXpt8L1M0u7J?ths=true) for the pre-trained model.<br>
> ***Please go to the official repository if you wished to access the pre-trained model for SCANet.***

<br>
<br>
(Note: test.py file is modified to get the color mask/mapping of the segmented facial features)

<br><br>

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/zj99)

<img src="https://github.com/zijian99/FYP_DigitalMakeup/assets/92379986/f0259962-dac3-4d64-8269-c2917fa3c39f" width="170" height="170">

