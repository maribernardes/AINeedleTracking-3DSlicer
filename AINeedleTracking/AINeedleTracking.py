import logging
import os

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import SimpleITK as sitk
import sitkUtils
import numpy as np
from skimage.restoration import unwrap_phase

from math import sqrt, pow

import torch
from monaiUtils.sitkIO import LoadSitkImaged, PushSitkImaged
from monai.transforms import Compose, ConcatItemsd, EnsureChannelFirstd, ScaleIntensityd, Orientationd, Spacingd
from monai.transforms import Invertd, Activationsd, AsDiscreted, RemoveSmallObjectsd
from monai.networks.nets import UNet 
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.handlers.utils import from_engine

class AINeedleTracking(ScriptedLoadableModule):

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "AINeedleTracking"
    self.parent.categories = ["IGT"] 
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Mariana Bernardes (BWH), Junichi Tokuda (BWH)"] 
    self.parent.helpText = """ This is a 3D needle tracking module used to track the needle tip in RT-MRI images. Input requirement: 
    Magnitude/Phase image or Real/Imaginary image. Uses a MONAI UNet model trained with synthetic data"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """ """

    # Additional initialization step after application startup is complete
    # TODO: include sample data and testing routines
    # slicer.app.connect("startupCompleted()", registerSampleData)

################################################################################################################################################
# Custom Widget  - Separator
################################################################################################################################################
class SeparatorWidget(qt.QWidget):
    def __init__(self, label_text='Separator Widget Label', parent=None):
        super().__init__(parent)

        spacer = qt.QWidget()
        spacer.setFixedHeight(10)
        
        self.label = qt.QLabel(label_text)
        font = qt.QFont()
        font.setItalic(True)
        self.label.setFont(font)
        
        line = qt.QFrame()
        line.setFrameShape(qt.QFrame.HLine)
        line.setFrameShadow(qt.QFrame.Sunken)
        
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(qt.Qt.AlignVCenter)
        layout.addWidget(spacer)
        layout.addWidget(self.label)
        layout.addWidget(line)
        
        self.setLayout(layout)


################################################################################################################################################
# Widget Class
################################################################################################################################################

class AINeedleTrackingWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
  """
  # Called when the user opens the module the first time and the widget is initialized.
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    ####################################
    ##                                ##
    ## UI Components                  ##
    ##                                ##
    ####################################
    
    ## Setup                
    ####################################
    
    imagesCollapsibleButton = ctk.ctkCollapsibleButton()
    imagesCollapsibleButton.text = 'Setup'
    self.layout.addWidget(imagesCollapsibleButton)
    imagesFormLayout = qt.QFormLayout(imagesCollapsibleButton)
    
    # Input mode
    self.inputModeMagPhase = qt.QRadioButton('Magnitude/Phase')
    self.inputModeRealImag = qt.QRadioButton('Real/Imaginary')
    self.inputModeMagPhase.checked = 1
    self.inputModeButtonGroup = qt.QButtonGroup()
    self.inputModeButtonGroup.addButton(self.inputModeMagPhase)
    self.inputModeButtonGroup.addButton(self.inputModeRealImag)
    inputModeHBoxLayout = qt.QHBoxLayout()
    inputModeHBoxLayout.addWidget(self.inputModeMagPhase)
    inputModeHBoxLayout.addWidget(self.inputModeRealImag)
    imagesFormLayout.addRow('Input Mode:',inputModeHBoxLayout)
    
    ## Needle Tracking                
    ####################################

    trackingCollapsibleButton = ctk.ctkCollapsibleButton()
    trackingCollapsibleButton.text = 'Tracking'
    self.layout.addWidget(trackingCollapsibleButton)
    trackingFormLayout = qt.QFormLayout(trackingCollapsibleButton)
    
    # Input magnitude/real volume (first volume)
    self.firstVolumeSelector = slicer.qMRMLNodeComboBox()
    self.firstVolumeSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.firstVolumeSelector.selectNodeUponCreation = True
    self.firstVolumeSelector.addEnabled = True
    self.firstVolumeSelector.removeEnabled = True
    self.firstVolumeSelector.noneEnabled = True
    self.firstVolumeSelector.showHidden = False
    self.firstVolumeSelector.showChildNodeTypes = False
    self.firstVolumeSelector.setMRMLScene(slicer.mrmlScene)
    self.firstVolumeSelector.setToolTip('Select the magnitude/real image')
    trackingFormLayout.addRow('Magnitude/Real: ', self.firstVolumeSelector)

    # Input phase/imaginary volume (second volume)
    self.secondVolumeSelector = slicer.qMRMLNodeComboBox()
    self.secondVolumeSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.secondVolumeSelector.selectNodeUponCreation = True
    self.secondVolumeSelector.addEnabled = True
    self.secondVolumeSelector.removeEnabled = True
    self.secondVolumeSelector.noneEnabled = True
    self.secondVolumeSelector.showHidden = False
    self.secondVolumeSelector.showChildNodeTypes = False
    self.secondVolumeSelector.setMRMLScene(slicer.mrmlScene)
    self.secondVolumeSelector.setToolTip('Select the phase/imaginary image')
    trackingFormLayout.addRow('Phase/Imaginary: ', self.secondVolumeSelector)
    
    # Start/Stop tracking 
    trackingHBoxLayout = qt.QHBoxLayout()    
    self.startTrackingButton = qt.QPushButton('Start Tracking')
    self.startTrackingButton.toolTip = 'Start needle tracking in image sequence'
    self.startTrackingButton.enabled = False
    trackingHBoxLayout.addWidget(self.startTrackingButton)
    self.stopTrackingButton = qt.QPushButton('Stop Tracking')
    self.stopTrackingButton.toolTip = 'Stop the needle tracking'
    self.stopTrackingButton.enabled = False    
    trackingHBoxLayout.addWidget(self.stopTrackingButton)
    trackingFormLayout.addRow('', trackingHBoxLayout)
    
    ## Advanced parameters            
    ####################################

    advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    advancedCollapsibleButton.text = 'Advanced'
    advancedCollapsibleButton.collapsed=1
    self.layout.addWidget(advancedCollapsibleButton)
    advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)
    
    # Debug mode check box (output images at intermediate steps)
    self.debugFlagCheckBox = qt.QCheckBox()
    self.debugFlagCheckBox.checked = False
    self.debugFlagCheckBox.setToolTip('If checked, output images at intermediate steps')
    advancedFormLayout.addRow('Debug', self.debugFlagCheckBox)

    self.layout.addStretch(1)
    
    ####################################
    ##                                ##
    ## UI Behavior                    ##
    ##                                ##
    ####################################
    
    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
    # TODO: Create observer for phase image sequence and link to the self.receivedImage callback function

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.inputModeMagPhase.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.inputModeRealImag.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.firstVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.secondVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.debugFlagCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    
    # Connect UI buttons to event calls
    self.startTrackingButton.connect('clicked(bool)', self.startTracking)
    self.stopTrackingButton.connect('clicked(bool)', self.stopTracking)
    self.firstVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.secondVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    
    # Internal variables
    # self.segmentationNode = None
    self.isTrackingOn = False
    self.firstVolume = None
    self.secondVolume = None
    self.inputMode = None
    self.debugFlag = None

    # Initialize module logic
    self.logic = AINeedleTrackingLogic()
  
    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()
          
    # Refresh button states
    self.updateButtons()

  # Called when the application closes and the module widget is destroyed.
  def cleanup(self):
    self.removeObservers()

  # Called each time the user opens this module.
  # Make sure parameter node exists and observed
  def enter(self):
    self.initializeParameterNode() 

  # Called each time the user opens a different module.
  # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
  def exit(self):
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  # Called just before the scene is closed.
  # Parameter node will be reset, do not use it anymore
  def onSceneStartClose(self, caller, event):
    self.setParameterNode(None)

  # Called just after the scene is closed.
  # If this module is shown while the scene is closed then recreate a new parameter node immediately
  def onSceneEndClose(self, caller, event):
    if self.parent.isEntered:
      self.initializeParameterNode()
        
  # Ensure parameter node exists and observed
  # Parameter node stores all user choices in parameter values, node selections, etc.
  # so that when the scene is saved and reloaded, these settings are restored.
  def initializeParameterNode(self):
    # Load default parameters in logic module
    self.setParameterNode(self.logic.getParameterNode())
            
  # Set and observe parameter node.
  # Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
  def setParameterNode(self, inputParameterNode):
    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)
    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
        self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    # Initial GUI update
    self.updateGUIFromParameterNode()

  # This method is called whenever parameter node is changed.
  # The module GUI is updated to show the current state of the parameter node.
  def updateGUIFromParameterNode(self, caller=None, event=None):
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True
    # Update node selectors and sliders
    # self.manualMaskSelector.setCurrentNode(self._parameterNode.GetNodeReference('ManualMaskSegmentation'))
    self.firstVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference('FirstVolume'))
    self.secondVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference('SecondVolume'))
    self.inputModeMagPhase.checked = (self._parameterNode.GetParameter('InputMode') == 'MagPhase')
    self.inputModeRealImag.checked = (self._parameterNode.GetParameter('InputMode') == 'RealImag')
    self.debugFlagCheckBox.checked = (self._parameterNode.GetParameter('Debug') == 'True')
    
    # Update buttons states
    self.updateButtons()
    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  # This method is called when the user makes any change in the GUI.
  # The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
  def updateParameterNodeFromGUI(self, caller=None, event=None):
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return
    # Modify all properties in a single batch
    wasModified = self._parameterNode.StartModify()  
    # self._parameterNode.SetNodeReferenceID('ManualMaskSegmentation', self.manualMaskSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('FirstVolume', self.firstVolumeSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('SecondVolume', self.secondVolumeSelector.currentNodeID)
    self._parameterNode.SetParameter('InputMode', 'MagPhase' if self.inputModeMagPhase.checked else 'RealImag')
    self._parameterNode.SetParameter('Debug', 'True' if self.debugFlagCheckBox.checked else 'False')
    self._parameterNode.EndModify(wasModified)
                        
  # Update button states
  def updateButtons(self):
    rtNodesDefined = self.firstVolumeSelector.currentNode() and self.secondVolumeSelector.currentNode()
    self.startTrackingButton.enabled = rtNodesDefined and not self.isTrackingOn
    self.stopTrackingButton.enabled = self.isTrackingOn
    
  def startTracking(self):
    print('UI: startTracking()')
    self.isTrackingOn = True
    self.updateButtons()
    # Get parameters
    # Get selected nodes
    self.firstVolume = self.firstVolumeSelector.currentNode()
    self.secondVolume = self.secondVolumeSelector.currentNode()    
    self.logic.initializeTracking()
    # Create listener to sequence node
    self.addObserver(self.secondVolume, self.secondVolume.ImageDataModifiedEvent, self.receivedImage)
  
  def stopTracking(self):
    self.isTrackingOn = False
    self.updateButtons()
    #TODO: Define what should to be refreshed
    print('UI: stopTracking()')
    self.removeObserver(self.secondVolume, self.secondVolume.ImageDataModifiedEvent, self.receivedImage)
  
  def receivedImage(self, caller=None, event=None):
    # Execute one tracking cycle
    if self.isTrackingOn:
      print('UI: receivedImage()')
      # Get parameters
      self.inputMode = 'MagPhase' if self.inputModeMagPhase.checked else 'RealImag'
      self.debugFlag = self.debugFlagCheckBox.checked
      # Get needle tip
      if self.logic.getNeedle(self.firstVolume, self.secondVolume, self.inputMode, debugFlag=self.debugFlag):
        print('Tracking successful')
      else:
        print('Tracking failed')
      
    
################################################################################################################################################
# Logic Class
################################################################################################################################################

class AINeedleTrackingLogic(ScriptedLoadableModuleLogic):

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.cliParamNode = None
    print('Logic: __init__')

    # Image file writer
    self.path = os.path.dirname(os.path.abspath(__file__))
    self.debug_path = os.path.join(self.path,'Debug')
    self.fileWriter = sitk.ImageFileWriter()

    # Check if labelmap node exists, if not, create a new one
    try:
        self.needleLabelMapNode = slicer.util.getNode('NeedleLabelMap')
    except:
        self.needleLabelMapNode = slicer.vtkMRMLLabelMapVolumeNode()
        slicer.mrmlScene.AddNode(self.needleLabelMapNode)
        self.needleLabelMapNode.SetName('NeedleLabelMap')
        print('Created Needle Segmentation Node')

    # Check if tracked tip node exists, if not, create a new one
    try:
        self.tipTrackedNode = slicer.util.getNode('CurrentTrackedTipTransform')
    except:
        self.tipTrackedNode = slicer.vtkMRMLLinearTransformNode()
        slicer.mrmlScene.AddNode(self.tipTrackedNode)
        self.tipTrackedNode.SetName('CurrentTrackedTipTransform')
        print('Created Tracked Tip TransformNode')

    # Base ITK images
    # self.sitk_mask = None
    self.count = None
    
  # Initialize parameter node with default settings
  def setDefaultParameters(self, parameterNode):
    if not parameterNode.GetParameter('Debug'):
        parameterNode.SetParameter('Debug', 'False')   
  
  def pushSitkToSlicerLabelMap(self, sitk_image, name, debugFlag=False):
    # Get labelmap node
    labelmap_name = name+'LabelMap'
    try:
      labelmap_node = slicer.util.getNode(labelmap_name)
    except:
      labelmap_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
      labelmap_node.SetName(labelmap_name)
    # Push sitk image to labelmap volume 
    sitkUtils.PushVolumeToSlicer(sitk_image, labelmap_node)
    if (debugFlag==True):
      self.fileWriter.Execute(sitk_image, os.path.join(self.debug_path, labelmap_name)+'_seg.nrrd', False, 0)
    return labelmap_node

  # Create Slicer node and push ITK image to it
  def pushSitkToSlicerScalarVolume(self, sitk_image, node_name, debugFlag=False):
    # Check if node exists, if not, create a new one
    try:
      volume_node = slicer.util.getNode(node_name)
    except:
      volume_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
      volume_node.SetName(node_name)
    sitkUtils.PushVolumeToSlicer(sitk_image, volume_node)
    if (debugFlag==True):
      self.fileWriter.Execute(sitk_image, os.path.join(self.debug_path, node_name)+'.nrrd', False, 0)
    return volume_node

  # Return sitk Image from numpy array
  def numpyToitk(self, array, sitkReference, type=None):
    image = sitk.GetImageFromArray(array, isVector=False)
    if (type is None):
      image = sitk.Cast(image, sitkReference.GetPixelID())
    else:
      image = sitk.Cast(image, type)
    image.CopyInformation(sitkReference)
    return image
  
  def realImagToMagPhase(self, realVolume, imagVolume):
    # Pull the real/imaginary volumes from the MRML scene and convert them to magnitude/phase volumes
    sitk_real = sitkUtils.PullVolumeFromSlicer(realVolume)
    sitk_imag = sitkUtils.PullVolumeFromSlicer(imagVolume)
    numpy_real = sitk.GetArrayFromImage(sitk_real)
    numpy_imag = sitk.GetArrayFromImage(sitk_imag)
    numpy_comp = numpy_real + 1.0j * numpy_imag
    numpy_magn = np.absolute(numpy_comp)
    numpy_phase = np.angle(numpy_comp)
    sitk_magn = self.numpyToitk(numpy_magn, sitk_real)
    sitk_phase = self.numpyToitk(numpy_phase, sitk_real)
    return (sitk_magn, sitk_phase)

  # Return blank itk Image with same information from reference volume
  # Optionals: choose different pixelValue, type (pixel ID)
  # This is a simplified version from the Util.py method. Does NOT choose from different direction (and center image at the reference volume)
  def createBlankItk(sitkReference, type=None, pixelValue=0, spacing=None):
      image = sitk.Image(sitkReference.GetSize(), sitk.sitkUInt8)
      if (pixelValue != 0):
          image = pixelValue*sitk.Not(image)
      if (type is None):
          image = sitk.Cast(image, sitkReference.GetPixelID())
      else:
          image = sitk.Cast(image, type)  
      image.CopyInformation(sitkReference)
      if (spacing is not None):
          image.SetSpacing(spacing)                 # Set spacing
      return image

  # def getMaskFromSegmentation(self, segmentationNode, referenceVolumeNode):
  #   if segmentationNode is None:
  #     sitk_reference = sitkUtils.PullVolumeFromSlicer(referenceVolumeNode)
  #     sitk_mask = self.createBlankItk(sitk_reference, sitk.sitkUInt8)
  #     return sitk.Not(sitk_mask)
  #   labelmapVolumeNode = slicer.util.getFirstNodeByName('mask_labelmap')
  #   if labelmapVolumeNode is None or labelmapVolumeNode.GetClassName() != 'vtkMRMLLabelMapVolumeNode':      
  #     labelmapVolumeNode = slicer.vtkMRMLLabelMapVolumeNode()
  #     slicer.mrmlScene.AddNode(labelmapVolumeNode)
  #     labelmapVolumeNode.SetName('mask_labelmap')
  #   slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode, referenceVolumeNode)
  #   sitk_mask = sitkUtils.PullVolumeFromSlicer(labelmapVolumeNode)
  #   sitk_mask  = sitk.Cast(sitk_mask, sitk.sitkFloat32)
  #   return sitk_mask

  # Initialize the tracking
  def initializeTracking(self, in_channels=2, out_channels=3, orientation='PIL', pixel_dim=(3.6, 1.171875, 1.171875), min_size_obj=50):
    # Initialize sequence counter
    self.count = 0
    # Setup UNet model
    print('Setting UNet')
    model_file= os.path.join(self.path, 'Models', 'model_MP_multi.pth')
    model_unet = UNet(
      spatial_dims=3, # Mariana: dimensions=3 was deprecated
      in_channels=in_channels,
      out_channels=out_channels,
      channels=[16, 32, 64, 128], 
      strides=[(1, 2, 2), (1, 2, 2), (1, 1, 1)], 
      num_res_units=2,
      norm=Norm.BATCH,
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.model = model_unet.to(device)
    self.model.load_state_dict(torch.load(model_file, map_location=device))
    # Setup transforms
    # Define pre-inference transforms
    if in_channels==2:
      pre_array = [
        # 2-channel input
        LoadSitkImaged(keys=["image_1", "image_2"]),
        EnsureChannelFirstd(keys=["image_1", "image_2"]), 
        ConcatItemsd(keys=["image_1", "image_2"], name="image"),
      ]        
    else:
      pre_array = [
        # 1-channel input
        LoadSitkImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
      ]
    pre_array.append(ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True))
    pre_array.append(Orientationd(keys=["image"], axcodes=orientation))
    pre_array.append(Spacingd(keys=["image"], pixdim=pixel_dim, mode=("bilinear")))
    self.pre_transforms = Compose(pre_array)
    # Define post-inference transforms
    self.post_transforms = Compose([
      Invertd(
        keys="pred",
        transform=self.pre_transforms,
        orig_keys="image", 
        meta_keys="pred_meta_dict", 
        orig_meta_keys="image_meta_dict",  
        meta_key_postfix="meta_dict",  
        nearest_interp=False,
        to_tensor=True,
      ),
      Activationsd(keys="pred", sigmoid=True),
      AsDiscreted(keys="pred", argmax=True, num_classes=3),
      RemoveSmallObjectsd(keys="pred", min_size=int(min_size_obj), connectivity=1, independent_channels=False),
      PushSitkImaged(keys="pred", meta_keys="pred_meta_dict", resample=False, output_dtype=np.uint16, print_log=False),
    ])  
    
  def getNeedle(self, firstVolume, secondVolume, inputMode, in_channels=2, out_channels=3, window_size=(3,48,48), debugFlag=False):
    # Using only one slice volumes for now
    # TODO: extend to 3 stacked slices
    print('Logic: getNeedle()')    
    # Increment sequence counter
    self.count += 1    
    # Get itk images from MRML volume nodes 
    if (inputMode == 'RealImag'): # Convert to magnitude/phase
      (sitk_img_m, sitk_img_p) = self.realImagToMagPhase(firstVolume, secondVolume)
    else:                         # Already as magnitude/phase
      sitk_img_m = sitkUtils.PullVolumeFromSlicer(firstVolume)
      sitk_img_p = sitkUtils.PullVolumeFromSlicer(secondVolume)
    # Cast it to 32Float
    sitk_img_m = sitk.Cast(sitk_img_m, sitk.sitkFloat32)
    sitk_img_p = sitk.Cast(sitk_img_p, sitk.sitkFloat32)
    # Push debug images to Slicer     
    if debugFlag:
      self.pushSitkToSlicerScalarVolume(sitk_img_m, 'debug_img_m', debugFlag)
      self.pushSitkToSlicerScalarVolume(sitk_img_p, 'debug_img_p', debugFlag)

    ######################################
    ##                                  ##
    ## Step 0: Set input dictionary     ##
    ##                                  ##
    ######################################
    # Set input dictionary
    if in_channels==2:
      input_dict = [{'image_1': sitk_img_m, 'image_2': sitk_img_p}]
    else:
      input_dict = [{'image':sitk_img_m}]

    ######################################
    ##                                  ##
    ## Step 1: Inference                ##
    ##                                  ##
    ######################################

    # Apply pre_transforms
    data = self.pre_transforms(input_dict)[0]

    # Evaluate model
    self.model.eval()
    with torch.no_grad():
        batch_input = data['image'].unsqueeze(0)
        val_inputs = batch_input.to(torch.device('cpu'))
        val_outputs = sliding_window_inference(val_inputs, window_size, 4, self.model)
        data['pred'] = decollate_batch(val_outputs)[0]
        sitk_output = self.post_transforms(data)['pred']

    self.pushSitkToSlicerLabelMap(sitk_output,'Needle', debugFlag=debugFlag)

    ######################################
    ##                                  ##
    ## Step 2: Get centroid for tip     ##
    ##                                  ##
    ######################################

    # # Threshold roi to create blobs
    # sitk_blobs = (sitk_phaseGradient > blobThreshold)
    # # Put slice in the volume
    # sitk_blobsVolume = self.createBlankItk(sitk_roi, sitk_blobs.GetPixelID())
    # sitk_blobsVolume[:,:,sliceIndex] = sitk_blobs
    # # Plot
    # if debugFlag:
    #   self.pushitkToSlicerVolume(sitk_blobsVolume, 'debug_blobs', debugFlag)  
          
    # # Label blobs
    # stats = sitk.LabelShapeStatisticsImageFilter()
    # stats.Execute(sitk.ConnectedComponent(sitk_blobsVolume))

    # # Get blobs sizes and centroid physical coordinates
    # labels_size = []
    # labels_centroid = []
    # labels_depth = []
    # for l in stats.GetLabels():
    #     if debugFlag:
    #         print('Label %s: -> Size: %s, Center: %s, Flatness: %s, Elongation: %s' %(l, stats.GetNumberOfPixels(l), stats.GetCentroid(l), stats.GetFlatness(l), stats.GetElongation(l)))
    #     if (stats.GetElongation(l) < 4): 
    #         labels_size.append(stats.GetNumberOfPixels(l))
    #         labels_centroid.append(stats.GetCentroid(l))    
    #         labels_depth.append(stats.GetCentroid(l)[2])

    # ####################################
    # ##                                ##
    # ## Step 6: Get tip physical point ##
    # ##                                ##
    # ####################################
    # # Check number of centroids found
    # if len(labels_size)>15:
    #   print('Too many centroids, probably noise')
    #   return False
    # # Reasonable number of centroids
    # # Get larger one
    # try:
    #   sorted_by_size = np.argsort(labels_size) 
    #   first_largest = sorted_by_size[-1]
    #   # second_largest = sorted_by_size[-2]
    # except:
    #   print('No centroids found')
    #   return False
    
    # # Check centroid size with respect to ROI size
    # if (labels_size[first_largest] > 0.5*roiSize*0.5*roiSize):
    #   print('Centroid too big, probably noise')
    #   return False
    
    # # # Get significantly bigger centroid
    # # if (labels_size[first_largest] > 3.5*labels_size[second_largest]):
    # #   label_index = first_largest
    # # else: # Get centroid further inserted
    # #   label_index = labels_depth.index(max(labels_depth))

    # # Get selected centroid center
    # # center = labels_centroid[label_index]
    # center = labels_centroid[first_largest]
    # # Convert to 3D Slicer coordinates (RAS)
    # centerRAS = (-center[0], -center[1], center[2])

    # # Plot
    # if debugFlag:
    #   # print('Chosen label: %i' %(label_index+1))
    #   print('Chosen label: %i' %(first_largest+1))
    #   print(centerRAS)

    # # Calculate prediction error
    # predError = sqrt(pow((tipRAS[0]-centerRAS[0]),2)+pow((tipRAS[1]-centerRAS[1]),2)+pow((tipRAS[2]-centerRAS[2]),2))
    # # Check error threshold
    # if(predError>errorThreshold):
    #   print('Tip too far from prediction')
    #   return False
    
    # # Push coordinates to tip Node
    # transformMatrix.SetElement(0,3, centerRAS[0])
    # transformMatrix.SetElement(1,3, centerRAS[1])
    # transformMatrix.SetElement(2,3, centerRAS[2])
    # self.tipTrackedNode.SetMatrixTransformToParent(transformMatrix)

    return True
