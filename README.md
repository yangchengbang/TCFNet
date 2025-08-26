# Title
Enhancing Salient Object Detection in RGB-D Videos via Tri-modal Complementary Fusion

This code is directly related to the manuscript “Enhancing Salient Object Detection in RGB-D Videos via Tri-modal Complementary Fusion” submitted to The Visual Computer. Please cite the manuscript to support related research.
# Usage
## Requirements
   - Python 3.8.12
   - PyTorch 2.0.1
   - Torchvision 0.15.2
   - Cuda 11.7

## Training
1. Download the training datasets (DAVIS、DAVSOD、FBMS and DUTS) from [Baidu Driver](https://pan.baidu.com/s/1ZF5CWg_g2GdR3G9Xj03HzA) (PSW: V2y2) and save it at './dataset/'. 
2. Download the pre_trained RGB, depth and flow stream models from [Baidu Driver](https://pan.baidu.com/s/1JMVDZaLk6u89kSR4BNitkA) (PSW: Vc2u) to './checkpoints/'.
3. Run `python train.py` in terminal.

## Testing
1. Download the testing datasets (DAVIS、DAVSOD、FBMS、SegTrack-V2 and VOS) from [Baidu Driver](https://pan.baidu.com/s/1knPUX5pYDCwgpcrC2d7EWg) (PSW: y5Hu) and save it at './dataset/'.
2. Download the trained model from [Baidu Driver](https://pan.baidu.com/s/1JMVDZaLk6u89kSR4BNitkA) (PSW: Vc2u, final_bone.pth) to './checkpoints/'.
3. Run `python test.py` in the terminal.

## Results
The saliency maps of our TCFNet can be download from [Baidu Driver](https://pan.baidu.com/s/1gyo-VDz7zyH_WdfEq5E13w) (PSW: 1UPa, TCFNet_result)
