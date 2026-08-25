# AINeedleTracking-3DSlicer

## Installation:
First, install the appropriate torch version in 3DSlicer.

### MacOS (Apple Silicon)

For Slicer 5.10 type in the Python Interactor:

```
slicer.util.pip_install("numpy<2.0.0 torch==2.2.2 torchvision==0.17.2 monai==1.4.0 --force-reinstall")
slicer.util.pip_install('nibabel')
slicer.util.pip_install('scikit-image')
```

### Linux

Use the PyTorch Utils module (https://github.com/fepegar/SlicerPyTorch)

OR

install it manually using 3D Slicer Python Interactor (and adjusting the url for the appropriate version):
```
slicer.util.pip_install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
```

Then, install the following python packages to 3D Slicer (use 3D Slicer Python Interactor):
```
slicer.util.pip_install('monai')
slicer.util.pip_install('nibabel')
slicer.util.pip_install('scikit-image')
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
