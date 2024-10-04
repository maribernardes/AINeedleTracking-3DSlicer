# AINeedleTracking-3DSlicer

## Installation:
First, use the PyTorch Utils module to install the appropriate torch version in 3DSlicer: 
https://github.com/fepegar/SlicerPyTorch

```
slicer.util.pip_install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
```

Then, install the following python packages to 3D Slicer (use 3D Slicer Python Interactor):
```
slicer.util.pip_install('monai[all]')
```

## Use:
### BRPRobot Project:
- Setup
  * Input mode: magnitude/phase
  * Model: UNet_needle_segmentation
- Optional
  * Not in use for BRPRobot
- Tracking
  * Magnitude: input image
  * Phase: input image
  * Mask (optional): none or SegmentationNode
- Advanced:
  * Debug flag: off (less verbose)
  
### OpenIGTLink Project:
- Setup
  * Input mode: magnitude/phase
  * Model: UNet_needle_segmentation
- Optional
  * Push to scanner: PLAN_0 (requires IGTLink Server, initializes with current slice in selected viewer)
  * Push to robot: target and current tip (requires IGTLink Client, ZFrame Transform, List of markups fiducials)
- Tracking
  * Magnitude: input image
  * Phase: input image
  * Mask (optional): none or SegmentationNode
- Advanced:
  * Debug flag: off (less verbose)
