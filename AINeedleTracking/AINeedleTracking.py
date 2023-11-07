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
from monai.transforms import Invertd, Activationsd, AsDiscreted, KeepLargestConnectedComponentd, RemoveSmallObjectsd
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
    
    setupCollapsibleButton = ctk.ctkCollapsibleButton()
    setupCollapsibleButton.text = 'Setup'
    self.layout.addWidget(setupCollapsibleButton)
    setupFormLayout = qt.QFormLayout(setupCollapsibleButton)
    
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
    setupFormLayout.addRow('Input Mode:',inputModeHBoxLayout)
    
    # AI model
    self.modelFileSelector = qt.QComboBox()
    modelPath= os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models')
    self.modelList = [f for f in os.listdir(modelPath) if os.path.isfile(os.path.join(modelPath, f))]
    self.modelFileSelector.addItems(self.modelList)
    setupFormLayout.addRow('AI Model:',self.modelFileSelector)

    ## Optional                
    ####################################
    
    optionalCollapsibleButton = ctk.ctkCollapsibleButton()
    optionalCollapsibleButton.collapsed = True
    optionalCollapsibleButton.text = 'Optional'
    self.layout.addWidget(optionalCollapsibleButton)
    optionalFormLayout = qt.QFormLayout(optionalCollapsibleButton)

    igtlHBoxLayout = qt.QHBoxLayout()   

    # Push scan plane to scanner
    self.pushScanPlaneCheckBox = qt.QCheckBox()
    self.pushScanPlaneCheckBox.checked = False
    self.pushScanPlaneCheckBox.setToolTip('If checked, enables pushing scan plane to scanner')
    igtlHBoxLayout.addWidget(qt.QLabel('Push Scan Plane'))
    igtlHBoxLayout.addWidget(self.pushScanPlaneCheckBox)

    # Push target coordinates to robot
    self.pushTargetToRobotCheckBox = qt.QCheckBox()
    self.pushTargetToRobotCheckBox.checked = False
    self.pushTargetToRobotCheckBox.setToolTip('If checked, pushes target position to robot in zFrame coordinates')
    igtlHBoxLayout.addWidget(qt.QLabel('Push Target to Robot'))
    igtlHBoxLayout.addWidget(self.pushTargetToRobotCheckBox)

    # Push tip coordinates to robot
    self.pushTipToRobotCheckBox = qt.QCheckBox()
    self.pushTipToRobotCheckBox.checked = False
    self.pushTipToRobotCheckBox.setToolTip('If checked, pushes current tip position to robot in zFrame coordinates')
    igtlHBoxLayout.addWidget(qt.QLabel('Push Tip to Robot'))
    igtlHBoxLayout.addWidget(self.pushTipToRobotCheckBox)

    optionalFormLayout.addRow(igtlHBoxLayout)

    separator1 = SeparatorWidget('MRI Scan Plane')
    optionalFormLayout.addRow(separator1)

    # Select MRI Bridge OpenIGTLink connection
    self.bridgeConnectionSelector = slicer.qMRMLNodeComboBox()
    self.bridgeConnectionSelector.nodeTypes = ['vtkMRMLIGTLConnectorNode']
    self.bridgeConnectionSelector.selectNodeUponCreation = True
    self.bridgeConnectionSelector.addEnabled = False
    self.bridgeConnectionSelector.removeEnabled = False
    self.bridgeConnectionSelector.noneEnabled = True
    self.bridgeConnectionSelector.showHidden = False
    self.bridgeConnectionSelector.showChildNodeTypes = False
    self.bridgeConnectionSelector.setMRMLScene(slicer.mrmlScene)
    self.bridgeConnectionSelector.setToolTip('Select MRI Bridge OpenIGTLink connection')
    self.bridgeConnectionSelector.enabled = False
    optionalFormLayout.addRow('IGTLServer MRI:', self.bridgeConnectionSelector)

    # Select which scene view to initialize PLAN_0 and send to scanner
    planeHBoxLayout = qt.QHBoxLayout()
    self.sceneViewButton_red = qt.QRadioButton('Red')
    self.sceneViewButton_yellow = qt.QRadioButton('Yellow')
    self.sceneViewButton_green = qt.QRadioButton('Green')
    self.sceneViewButton_green.checked = 1
    self.sceneViewButtonGroup = qt.QButtonGroup()
    self.sceneViewButtonGroup.addButton(self.sceneViewButton_red)
    self.sceneViewButtonGroup.addButton(self.sceneViewButton_yellow)
    self.sceneViewButtonGroup.addButton(self.sceneViewButton_green)
    self.sceneViewButton_red.enabled = False
    self.sceneViewButton_yellow.enabled = False
    self.sceneViewButton_green.enabled = False
    sliceLabel = qt.QLabel('Inital scan plane:')
    sliceLabel.setToolTip('Select initial slice position for scan plane')
    planeHBoxLayout.addWidget(sliceLabel)
    planeHBoxLayout.addWidget(self.sceneViewButton_red)
    planeHBoxLayout.addWidget(self.sceneViewButton_yellow)
    planeHBoxLayout.addWidget(self.sceneViewButton_green)
    self.sendPlaneButton = qt.QPushButton('Set Initial Scan Plane')
    self.sendPlaneButton.toolTip = 'Send scan plane to scanner'
    self.sendPlaneButton.setFixedWidth(170)
    self.sendPlaneButton.enabled = False
    planeHBoxLayout.addWidget(self.sendPlaneButton)
    optionalFormLayout.addRow(planeHBoxLayout)   

    # UpdateScanPlan mode check box (output images at intermediate steps)
    self.updateScanPlaneCheckBox = qt.QCheckBox()
    self.updateScanPlaneCheckBox.checked = False
    self.updateScanPlaneCheckBox.setToolTip('If checked, updates scan plane with current tip position')
    optionalFormLayout.addRow('Auto update:', self.updateScanPlaneCheckBox)

    separator2 = SeparatorWidget('Target and Needle Tip')
    optionalFormLayout.addRow(separator2)

    # Select Robot OpenIGTLink connection
    self.robotConnectionSelector = slicer.qMRMLNodeComboBox()
    self.robotConnectionSelector.nodeTypes = ['vtkMRMLIGTLConnectorNode']
    self.robotConnectionSelector.selectNodeUponCreation = True
    self.robotConnectionSelector.addEnabled = False
    self.robotConnectionSelector.removeEnabled = False
    self.robotConnectionSelector.noneEnabled = True
    self.robotConnectionSelector.showHidden = False
    self.robotConnectionSelector.showChildNodeTypes = False
    self.robotConnectionSelector.setMRMLScene(slicer.mrmlScene)
    self.robotConnectionSelector.setToolTip('Select robot OpenIGTLink connection')
    self.robotConnectionSelector.enabled = False
    optionalFormLayout.addRow('IGTLClient Robot:', self.robotConnectionSelector)

    # Select zFrameToWorld transform
    self.transformSelector = slicer.qMRMLNodeComboBox()
    self.transformSelector.nodeTypes = ['vtkMRMLLinearTransformNode']
    self.transformSelector.selectNodeUponCreation = True
    self.transformSelector.addEnabled = False
    self.transformSelector.removeEnabled = False
    self.transformSelector.noneEnabled = True
    self.transformSelector.showHidden = False
    self.transformSelector.showChildNodeTypes = False
    self.transformSelector.setMRMLScene(slicer.mrmlScene)
    self.transformSelector.setToolTip('Select ZFrame registration transform')
    self.transformSelector.enabled = False
    optionalFormLayout.addRow('ZFrame Transform:', self.transformSelector)

    # Select target and send with OpenIGTLink server
    targetHBoxLayout = qt.QHBoxLayout()
    self.targetSelector = slicer.qMRMLNodeComboBox()
    self.targetSelector.nodeTypes = ['vtkMRMLMarkupsFiducialNode']
    self.targetSelector.selectNodeUponCreation = True
    self.targetSelector.addEnabled = False
    self.targetSelector.removeEnabled = False
    self.targetSelector.noneEnabled = True
    self.targetSelector.showHidden = False
    self.targetSelector.showChildNodeTypes = False
    self.targetSelector.setMRMLScene(slicer.mrmlScene)
    self.targetSelector.setToolTip('Select target')
    self.targetSelector.enabled = False
    targetHBoxLayout.addWidget(qt.QLabel('Target Markups:'))
    targetHBoxLayout.addWidget(self.targetSelector)
    
    self.sendTargetButton = qt.QPushButton('Push Target')
    self.sendTargetButton.toolTip = 'Send target to robot'
    self.sendTargetButton.setFixedWidth(170)
    self.sendTargetButton.enabled = False
    targetHBoxLayout.addWidget(self.sendTargetButton)

    optionalFormLayout.addRow(targetHBoxLayout)
    
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
    
    # Select a segmentation for masking (optional)
    self.segmentationMaskSelector = slicer.qMRMLNodeComboBox()
    self.segmentationMaskSelector.nodeTypes = ['vtkMRMLSegmentationNode']
    self.segmentationMaskSelector.selectNodeUponCreation = True
    self.segmentationMaskSelector.noneEnabled = True
    self.segmentationMaskSelector.showChildNodeTypes = False
    self.segmentationMaskSelector.showHidden = False
    self.segmentationMaskSelector.setMRMLScene(slicer.mrmlScene)
    self.segmentationMaskSelector.setToolTip('Select segmentation for masking input images')
    trackingFormLayout.addRow('Mask (optional): ', self.segmentationMaskSelector)
    
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
    self.segmentationMaskSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.debugFlagCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.pushScanPlaneCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.pushTipToRobotCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.pushTargetToRobotCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.sceneViewButton_red.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.sceneViewButton_yellow.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.sceneViewButton_green.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.updateScanPlaneCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.bridgeConnectionSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.robotConnectionSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.transformSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.targetSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.modelFileSelector.connect('currentIndexChanged(str)', self.updateParameterNodeFromGUI)
    
    # Connect UI buttons to event calls
    self.startTrackingButton.connect('clicked(bool)', self.startTracking)
    self.stopTrackingButton.connect('clicked(bool)', self.stopTracking)
    self.sendPlaneButton.connect('clicked(bool)', self.sendPlane)
    self.sendTargetButton.connect('clicked(bool)', self.sendTarget)
    
    self.firstVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.secondVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.pushScanPlaneCheckBox.connect("toggled(bool)", self.updateButtons)
    self.pushTipToRobotCheckBox.connect("toggled(bool)", self.updateButtons)
    self.pushTargetToRobotCheckBox.connect("toggled(bool)", self.updateButtons)
    self.updateScanPlaneCheckBox.connect("toggled(bool)", self.updateButtons)
    self.bridgeConnectionSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.robotConnectionSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.transformSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.targetSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    
    # Internal variables
    self.isTrackingOn = False
    self.inputMode = None
    self.firstVolume = None
    self.secondVolume = None
    self.debugFlag = None
    self.pushScanPlane = None
    self.pushTipToRobot = None
    self.pushTargetToRobot = None
    self.updateScanPlane = None
    self.serverNode = None
    self.clientNode = None
    self.zFrameTransform = None

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
    self.segmentationMaskSelector.setCurrentNode(self._parameterNode.GetNodeReference('Mask'))
    self.inputModeMagPhase.checked = (self._parameterNode.GetParameter('InputMode') == 'MagPhase')
    self.inputModeRealImag.checked = (self._parameterNode.GetParameter('InputMode') == 'RealImag')
    self.debugFlagCheckBox.checked = (self._parameterNode.GetParameter('Debug') == 'True')
    self.pushScanPlaneCheckBox.checked = (self._parameterNode.GetParameter('PushScanPlane') == 'True')
    self.pushTipToRobotCheckBox.checked = (self._parameterNode.GetParameter('PushTipToRobot') == 'True')
    self.pushTargetToRobotCheckBox.checked = (self._parameterNode.GetParameter('PushTargetToRobot') == 'True')
    self.updateScanPlaneCheckBox.checked = (self._parameterNode.GetParameter('UpdateScanPlane') == 'True')
    self.bridgeConnectionSelector.setCurrentNode(self._parameterNode.GetNodeReference('Server'))
    self.robotConnectionSelector.setCurrentNode(self._parameterNode.GetNodeReference('Client'))
    self.transformSelector.setCurrentNode(self._parameterNode.GetNodeReference('zFrame'))
    self.targetSelector.setCurrentNode(self._parameterNode.GetNodeReference('Target'))
    self.modelFileSelector.setCurrentIndex(int(self._parameterNode.GetParameter('Model')))
    
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
    self._parameterNode.SetNodeReferenceID('Mask', self.segmentationMaskSelector.currentNodeID)
    self._parameterNode.SetParameter('InputMode', 'MagPhase' if self.inputModeMagPhase.checked else 'RealImag')
    self._parameterNode.SetParameter('Debug', 'True' if self.debugFlagCheckBox.checked else 'False')
    self._parameterNode.SetParameter('PushScanPlane', 'True' if self.pushScanPlaneCheckBox.checked else 'False')
    self._parameterNode.SetParameter('PushTipToRobot', 'True' if self.pushTipToRobotCheckBox.checked else 'False')
    self._parameterNode.SetParameter('PushTargetToRobot', 'True' if self.pushTargetToRobotCheckBox.checked else 'False')
    self._parameterNode.SetParameter('UpdateScanPlane', 'True' if self.updateScanPlaneCheckBox.checked else 'False')
    self._parameterNode.SetNodeReferenceID('Server', self.bridgeConnectionSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('Client', self.robotConnectionSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('zFrame', self.transformSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('Target', self.targetSelector.currentNodeID)
    self._parameterNode.SetParameter('Model', str(self.modelFileSelector.currentIndex))
    self._parameterNode.EndModify(wasModified)
                        
  # Update button states
  def updateButtons(self):
    # Initialize requirements as all optional (True)
    serverDefined = True      # Not required
    clientDefined = True      # Not required
    transformDefined = True   # Not required
    targetDefined = True      # Not required 
    # Not tracking = ENABLE SELECTION
    if not self.isTrackingOn:
      # Logic for required variables selection: 
      self.inputModeMagPhase.enabled = True
      self.inputModeRealImag.enabled = True
      self.modelFileSelector.enabled = True
      self.pushScanPlaneCheckBox.enabled = True
      self.pushTipToRobotCheckBox.enabled = True
      self.pushTargetToRobotCheckBox.enabled = True
      self.firstVolumeSelector.enabled = True
      self.secondVolumeSelector.enabled = True
      self.segmentationMaskSelector.enabled = True
      # Logic for optional variables selection
      # 1) Push Scan Plane
      if self.pushScanPlaneCheckBox.checked:
        self.updateScanPlaneCheckBox.enabled = True
        self.sceneViewButton_red.enabled = True
        self.sceneViewButton_yellow.enabled = True
        self.sceneViewButton_green.enabled = True    
        self.bridgeConnectionSelector.enabled = True
        serverDefined = self.bridgeConnectionSelector.currentNode()
        if serverDefined:
          self.sendPlaneButton.enabled = True
        else:
          self.sendPlaneButton.enabled = False
      else:
        self.updateScanPlaneCheckBox.enabled = False
        self.sceneViewButton_red.enabled = False
        self.sceneViewButton_yellow.enabled = False
        self.sceneViewButton_green.enabled = False
        self.bridgeConnectionSelector.enabled = False
        self.sendPlaneButton.enabled = False
      # 2) Push target and tip (required for both)
      if self.pushTipToRobotCheckBox.checked or self.pushTargetToRobotCheckBox.checked:
        self.robotConnectionSelector.enabled = True
        self.transformSelector.enabled = True
        clientDefined = self.robotConnectionSelector.currentNode()
        transformDefined = self.transformSelector.currentNode()
      else:
        self.robotConnectionSelector.enabled = False
        self.transformSelector.enabled = False
      # 3) Push target only
      if self.pushTargetToRobotCheckBox.checked:
        self.targetSelector.enabled = True
        targetDefined = self.targetSelector.currentNode()
        if targetDefined and transformDefined:
          self.sendTargetButton.enabled = True
        else:
          self.sendTargetButton.enabled = False
      else:
        self.targetSelector.enabled = False
        self.sendTargetButton.enabled = False
    # When tracking = DISABLE SELECTION
    else:
      self.inputModeMagPhase.enabled = False
      self.inputModeRealImag.enabled = False
      self.modelFileSelector.enabled = False
      self.pushScanPlaneCheckBox.enabled = False
      self.pushTipToRobotCheckBox.enabled = False
      self.pushTargetToRobotCheckBox.enabled = False
      self.sendPlaneButton.enabled = False
      self.updateScanPlaneCheckBox.enabled = False
      self.firstVolumeSelector.enabled = False
      self.secondVolumeSelector.enabled = False
      self.segmentationMaskSelector.enabled = False
      self.sceneViewButton_red.enabled = False
      self.sceneViewButton_yellow.enabled = False
      self.sceneViewButton_green.enabled = False
      self.bridgeConnectionSelector.enabled = False
      self.robotConnectionSelector.enabled = False
      self.transformSelector.enabled = False
      self.targetSelector.enabled = False
      self.sendTargetButton.enabled = False

    # Check if Tracking is enabled
    rtNodesDefined = self.firstVolumeSelector.currentNode() and self.secondVolumeSelector.currentNode()
    self.startTrackingButton.enabled = rtNodesDefined and serverDefined and clientDefined and transformDefined and targetDefined and not self.isTrackingOn
    self.stopTrackingButton.enabled = self.isTrackingOn
    
  # Get selected scene view for initializing scan plane (PLANE_0)
  def getSelectedView(self):
    selectedView = None
    if (self.sceneViewButton_red.checked == True):
      selectedView = ('Red')
    elif (self.sceneViewButton_yellow.checked ==True):
      selectedView = ('Yellow')
    elif (self.sceneViewButton_green.checked ==True):
      selectedView = ('Green')
    return selectedView

  # Get center coordinates from current selected view
  def getSelectetViewCenterCoordinates(self, selectedView):
    # Get slice widget from selected view
    sliceNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNode'+str(selectedView))
    # Get the slice center coordinates
    m = sliceNode.GetSliceToRAS()
    centerRAS = (m.GetElement(0,3), m.GetElement(1,3), m.GetElement(2,3))
    return centerRAS

  def startTracking(self):
    print('UI: startTracking()')
    self.isTrackingOn = True
    self.updateButtons()
    # Store selected parameters
    model = self.modelFileSelector.currentText
    self.pushScanPlane = self.pushScanPlaneCheckBox.checked 
    self.pushTipToRobot = self.pushTipToRobotCheckBox.checked 
    self.firstVolume = self.firstVolumeSelector.currentNode()
    self.secondVolume = self.secondVolumeSelector.currentNode() 
    self.segmentationNode = self.segmentationMaskSelector.currentNode()
    self.inputMode = 'MagPhase' if self.inputModeMagPhase.checked else 'RealImag'
    self.updateScanPlane = self.updateScanPlaneCheckBox.checked 
    self.serverNode = self.bridgeConnectionSelector.currentNode()
    self.clientNode = self.robotConnectionSelector.currentNode()
    self.zFrameTransform = self.transformSelector.currentNode()
    # Initialize tracking logic
    self.logic.initializeTracking(model, self.segmentationNode, self.firstVolume)
    # Initialize PLAN_0
    if self.updateScanPlane == True:
      viewCoordinates = self.getSelectetViewCenterCoordinates(self.getSelectedView())
      self.logic.initializeScanPlane(coordinates=viewCoordinates, plane='COR') # Reinitialize PLAN_0 at center position
    # Initialize zFrame transform
    if self.pushTipToRobot == True:
      self.logic.initializeZFrame(self.zFrameTransform)
    # Check for needle in the current images
    self.getNeedle() 
    # Create listener to image sequence node
    self.addObserver(self.secondVolume, self.secondVolume.ImageDataModifiedEvent, self.receivedImage)
  
  def stopTracking(self):
    self.isTrackingOn = False
    self.updateButtons()
    #TODO: Define what should to be refreshed
    print('UI: stopTracking()')
    self.removeObserver(self.secondVolume, self.secondVolume.ImageDataModifiedEvent, self.receivedImage)
  
  def sendPlane(self):
    print('UI: sendPlane()')
    # Get parameters
    self.serverNode = self.bridgeConnectionSelector.currentNode()
    viewCoordinates = self.getSelectetViewCenterCoordinates(self.getSelectedView())
    # Set PLANE_0
    self.logic.initializeScanPlane(coordinates=viewCoordinates, plane='COR') # PLAN_0 at selected view slice (Default: sliceOnly=True)
    # Push PLAN_0    
    self.logic.pushScanPlaneToIGTLink(self.serverNode)
    
  def sendTarget(self):
    print('UI: sendTarget()')
    # Get parameters
    self.target = self.targetSelector.currentNode()
    self.clientNode = self.robotConnectionSelector.currentNode()
    self.zFrameTransform = self.transformSelector.currentNode()
    # Set zFrame transformation
    self.logic.initializeZFrame(self.zFrameTransform)
    # Push target
    self.logic.pushTargetToIGTLink(self.clientNode, self.target)

  def receivedImage(self, caller=None, event=None):
    self.getNeedle()

  def getNeedle(self):
    # Execute one tracking cycle
    if self.isTrackingOn:
      debugFlag = self.debugFlagCheckBox.checked
      # Get needle tip
      confidence = self.logic.getNeedle(self.firstVolume, self.secondVolume, self.inputMode, debugFlag=debugFlag) 
      if confidence is None:
        print('Tracking failed')
      else:
        print('Tracked with %s confidence' %confidence)
        if self.updateScanPlane is True:
          self.logic.updateScanPlane(checkConfidence=True)
          self.logic.pushScanPlaneToIGTLink(self.serverNode)
          print('PLAN_0 updated')
        if self.pushTipToRobot is True:
          self.logic.pushTipToIGTLink(self.clientNode)
          print('Tip pushed to robot')
    
################################################################################################################################################
# Logic Class
################################################################################################################################################

class AINeedleTrackingLogic(ScriptedLoadableModuleLogic):

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.cliParamNode = None

    # Image file writer
    self.path = os.path.dirname(os.path.abspath(__file__))
    self.debug_path = os.path.join(self.path,'Debug')
    self.fileWriter = sitk.ImageFileWriter()
    
    # Input image masking
    self.maskFilter = sitk.LabelMapMaskImageFilter()
    self.maskFilter.SetBackgroundValue(0)
    self.maskFilter.SetNegated(False)
    self.sitk_mask = None
    
    # Used for saving data from experiments
    self.count = None

    # Check if PLANE_0 node exists, if not, create a new one
    self.scanPlaneTransformNode = slicer.util.getFirstNodeByName('PLANE_0')
    if self.scanPlaneTransformNode is None or self.scanPlaneTransformNode.GetClassName() != 'vtkMRMLLinearTransformNode':
        self.scanPlaneTransformNode = slicer.vtkMRMLLinearTransformNode()
        self.scanPlaneTransformNode.SetName('PLANE_0')
        # self.scanPlaneTransformNode.SetHideFromEditors(True)
        slicer.mrmlScene.AddNode(self.scanPlaneTransformNode)
    self.initializeScanPlane(plane='COR')
    # Check if needle labelmap node exists, if not, create a new one
    self.needleLabelMapNode = slicer.util.getFirstNodeByName('NeedleLabelMap')
    if self.needleLabelMapNode is None or self.needleLabelMapNode.GetClassName() != 'vtkMRMLLabelMapVolumeNode':
        self.needleLabelMapNode = slicer.vtkMRMLLabelMapVolumeNode()
        self.needleLabelMapNode.SetName('NeedleLabelMap')
        slicer.mrmlScene.AddNode(self.needleLabelMapNode)
        colorTableNode = self.createColorTable()
        self.needleLabelMapNode.CreateDefaultDisplayNodes()
        self.needleLabelMapNode.GetDisplayNode().SetAndObserveColorNodeID(colorTableNode.GetID())
    # Check if text node exists, if not, create a new one
    self.needleConfidenceNode = slicer.util.getFirstNodeByName('CurrentTipConfidence')
    if self.needleConfidenceNode is None or self.needleConfidenceNode.GetClassName() != 'vtkMRMLTextNode':
        self.needleConfidenceNode = slicer.vtkMRMLTextNode()
        self.needleConfidenceNode.SetName('CurrentTipConfidence')
        slicer.mrmlScene.AddNode(self.needleConfidenceNode)
    # Check if tracked tip node exists, if not, create a new one
    self.tipTrackedNode = slicer.util.getFirstNodeByName('CurrentTrackedTipTransform')
    if self.tipTrackedNode is None or self.tipTrackedNode.GetClassName() != 'vtkMRMLLinearTransformNode':
        self.tipTrackedNode = slicer.vtkMRMLLinearTransformNode()
        self.tipTrackedNode.SetName('CurrentTrackedTipTransform')
        slicer.mrmlScene.AddNode(self.tipTrackedNode)
    # Check if zFrame tracked tip node exists, if not, create a new one
    self.tipTrackedZNode = slicer.util.getFirstNodeByName('CurrentTrackedTipZ')
    if self.tipTrackedZNode is None or self.tipTrackedZNode.GetClassName() != 'vtkMRMLLinearTransformNode':
        self.tipTrackedZNode = slicer.vtkMRMLLinearTransformNode()
        self.tipTrackedZNode.SetName('CurrentTrackedTipZ')
        self.tipTrackedZNode.SetHideFromEditors(True)
        slicer.mrmlScene.AddNode(self.tipTrackedZNode)
    # Check if WorldToZFrame transform node exists, if not, create a new one
    self.worldToZFrameNode = slicer.util.getFirstNodeByName('WorldToZFrame')
    if self.worldToZFrameNode is None or self.worldToZFrameNode.GetClassName() != 'vtkMRMLLinearTransformNode':
        self.worldToZFrameNode = slicer.vtkMRMLLinearTransformNode()
        self.worldToZFrameNode.SetName('WorldToZFrame')
        self.worldToZFrameNode.SetHideFromEditors(True)
        slicer.mrmlScene.AddNode(self.worldToZFrameNode)
    # Check if TargetZ point list node exists, if not, create a new one
    self.targetZNode = slicer.util.getFirstNodeByName('TargetZ')
    if self.targetZNode is None or self.targetZNode.GetClassName() != 'vtkMRMLMarkupsFiducialNode':
        self.targetZNode = slicer.vtkMRMLMarkupsFiducialNode()
        self.targetZNode.SetName('TargetZ')
        self.targetZNode.SetHideFromEditors(True)
        slicer.mrmlScene.AddNode(self.targetZNode)
    displayNode = self.targetZNode.GetDisplayNode()
    if displayNode:
      displayNode.SetVisibility(False)
  # Initialize parameter node with default settings
  def setDefaultParameters(self, parameterNode):
    if not parameterNode.GetParameter('Debug'):
      parameterNode.SetParameter('Debug', 'False')  
    if not parameterNode.GetParameter('PushScanPlane'): 
      parameterNode.SetParameter('PushScanPlane', 'False')  
    if not parameterNode.GetParameter('PushTipToRobot'): 
      parameterNode.SetParameter('PushTipToRobot', 'False')  
    if not parameterNode.GetParameter('PushTargetToRobot'): 
      parameterNode.SetParameter('PushTargetToRobot', 'False')
    if not parameterNode.GetParameter('UpdateScanPlane'): 
      parameterNode.SetParameter('UpdateScanPlane', 'False')    
    if not parameterNode.GetParameter('Model'): 
      parameterNode.SetParameter('Model', '0')
        
  # Create a ColorTable for the LabelMapNode
  # Lack of ColorTable was generating vtk error messages in the log for Slicer when running in my Mac
  def createColorTable(self):
    label_list = [("shaft", 1, 0.2, 0.5, 0.8), ('tip', 2, 1.0, 0.8, 0.7)]
    colorTableNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLColorTableNode")
    colorTableNode.SetHideFromEditors(True)
    colorTableNode.SetTypeToUser()
    colorTableNode.SetName('NeedleColorMap')
    slicer.mrmlScene.AddNode(colorTableNode); 
    colorTableNode.HideFromEditorsOff()  # make the color table selectable in the GUI outside Colors module
    colorTableNode.UnRegister(None)
    largestLabelValue = max([name_value[1] for name_value in label_list])
    colorTableNode.SetNumberOfColors(largestLabelValue + 1)
    colorTableNode.SetNamesInitialised(True) # prevent automatic color name generation
    for labelName, labelValue, labelColorR, labelColorG, labelColorB in label_list:
      colorTableNode.SetColor(labelValue, labelName, labelColorR, labelColorG, labelColorB)
    return colorTableNode
  
  def saveSitkImage(self, sitk_image, name, path, is_label=False):
    if is_label is True:
      self.fileWriter.Execute(sitk_image, os.path.join(path, name)+'_seg.nrrd', False, 0)
    else:
      self.fileWriter.Execute(sitk_image, os.path.join(path, name)+'.nrrd', False, 0)
  
  # Push an sitk image to a given volume node in Slicer
  # Volume node can be an object volume node (use already created node) or 
  # a string with node name (checks for node with this name and if non existant, creates it with provided type)
  def pushSitkToSlicerVolume(self, sitk_image, node: slicer.vtkMRMLScalarVolumeNode or slicer.vtkMRMLLabelMapVolumeNode or str, type='vtkMRMLScalarVolumeNode', debugFlag=False):
    # Provided a name (str)
    if isinstance(node, str):
      node_name = node
      # Check if node exists, if not, create a new one
      try:
        volume_node = slicer.util.getNode(node_name)
        volume_type = volume_node.GetClassName()
        if (volume_type != 'vtkMRMLScalarVolumeNode') and (volume_type != 'vtkMRMLLabelMapVolumeNode'):
          print('Error: node already exists and is not slicer.vtkMRMLScalarVolumeNode or slicer.vtkMRMLLabelMapVolumeNode')
          return False
      except:
        volume_type = type
        volume_node = slicer.mrmlScene.AddNewNodeByClass(volume_type)
        volume_node.SetName(node_name)
        if volume_type == 'vtkMRMLLabelMapVolumeNode': # For LabelMap node, create ColorTable
          colorTableNode = self.createColorTable()
          volume_node.GetDisplayNode().SetAndObserveColorNodeID(colorTableNode.GetID())
    elif isinstance(node, slicer.vtkMRMLScalarVolumeNode) or isinstance(node, slicer.vtkMRMLLabelMapVolumeNode):
      node_name = node.GetName()
      volume_node = node
      volume_type = volume_node.GetClassName()
    else:
      print('Error: variable labelmap is not valid (slicer.vtkMRMLScalarVolumeNode or slicer.vtkMRMLLabelMapVolumeNode or str)')
      return False
    sitkUtils.PushVolumeToSlicer(sitk_image, volume_node)
    return True

  # Given a binary skeleton image (single pixel-wide), find the physical coordinates of extremity closer to the image center
  def getShaftTipCoordinates(self, sitk_line):
    # Get the coordinates of all non-zero pixels in the binary image
    nonzero_coords = np.argwhere(sitk.GetArrayFromImage(sitk_line) == 1)
    # Calculate the distance of each non-zero pixel to all others
    distances = np.linalg.norm(nonzero_coords[:, None, :] - nonzero_coords[None, :, :], axis=-1)
    # Find the two points with the maximum distance; these are the extremity points
    extremity_indices = np.unravel_index(np.argmax(distances), distances.shape)
    extremitys_numpy = [nonzero_coords[index] for index in extremity_indices]
    # Conver to sitk array order
    extremity1 = (int(extremitys_numpy[0][2]), int(extremitys_numpy[0][1]), int(extremitys_numpy[0][0]))
    extremity2 = (int(extremitys_numpy[1][2]), int(extremitys_numpy[1][1]), int(extremitys_numpy[1][0]))
    # Calculate the center coordinates of the image volume
    image_shape = sitk_line.GetSize()
    center_coordinates = np.array(image_shape) / 2.0
    # Calculate the distances from each extremity point to the center
    distance1 = np.linalg.norm(np.array(extremity1) - center_coordinates)
    distance2 = np.linalg.norm(np.array(extremity2) - center_coordinates)
    # Determine which extremity is closer to the center and return physical coordinates
    if distance1 < distance2:
        return sitk_line.TransformIndexToPhysicalPoint(extremity1)
    else:
        return sitk_line.TransformIndexToPhysicalPoint(extremity2)

  def getNeedleDirection(labelmap):
    # Get the voxel coordinates of the labeled points (non-zero values) in the labelmap
    coordinates = np.array(np.where(labelmap))
    # Center the data by subtracting the mean
    centered_coordinates = coordinates - np.mean(coordinates, axis=1, keepdims=True)
    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_coordinates)
    # Perform PCA to find the principal direction (eigenvector) and its corresponding eigenvalue
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # Sort eigenvectors by decreasing eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    principal_direction = eigenvectors[:, 0]
    return principal_direction
  
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

  # Build a sitk mask volume from a segmentation node
  def getMaskFromSegmentation(self, segmentationNode, referenceVolumeNode):
    if segmentationNode is not None:
      # Create a temporary labelmap node
      maskLabelMapNode = slicer.vtkMRMLLabelMapVolumeNode()
      slicer.mrmlScene.AddNode(maskLabelMapNode)
      # Create mask from segmentation
      slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, maskLabelMapNode, referenceVolumeNode)
      sitk_mask = sitkUtils.PullVolumeFromSlicer(maskLabelMapNode)
      # Remove temporary labelmap node
      slicer.mrmlScene.RemoveNode(maskLabelMapNode)
      # Cast to valid type to be used with sitkMaskFilter
      return sitk.Cast(sitk_mask, sitk.sitkLabelUInt8)
    else:
      return None
  
  def setupUNet(self, model, in_channels=2, out_channels=3, orientation='PIL', pixel_dim=(3.6, 1.171875, 1.171875), min_size_obj=50):
    # Setup UNet model
    model_file= os.path.join(self.path, 'Models', model)
    model_unet = UNet(
      spatial_dims=3,
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
    ## Setup transforms
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
      # RemoveSmallObjectsd(keys="pred", min_size=int(min_size_obj), connectivity=1, independent_channels=True),
      # KeepLargestConnectedComponentd(keys="pred", independent=True),
      PushSitkImaged(keys="pred", meta_keys="pred_meta_dict", resample=False, output_dtype=np.uint16, print_log=False),
    ])  
  
  # Initialize the tracking logic
  def initializeTracking(self, model, segmentationNode, firstVolume):
    self.setupUNet(model) # Setup UNet
    self.count = 0        # Initialize sequence counter
    self.sitk_mask = self.getMaskFromSegmentation(segmentationNode, firstVolume)    # Update mask (None if nothing in segmentationNode)

  def initializeZFrame(self, zFrameToWorld):
    # Get world to ZFrame transformations
    worldToZFrame = vtk.vtkMatrix4x4()
    zFrameToWorld.GetMatrixTransformFromWorld(worldToZFrame)
    # Set it to worldToZFrameNode
    self.worldToZFrameNode.SetMatrixTransformToParent(worldToZFrame)
  
  # Set Scan Plane Orientation
  # Default position is (0,0,0), unless center is specified 
  # If sliceOnly, will set position of the slice only (keep the other coordinates zero)
  def initializeScanPlane(self, coordinates=(0,0,0), plane='COR', sliceOnly=True):
    m = vtk.vtkMatrix4x4()
    self.scanPlaneTransformNode.GetMatrixTransformToParent(m)
    position = [coordinates[0], coordinates[1], coordinates[2]]
    # Set rotation
    if plane == 'AX':
      m.SetElement(0, 0, 1); m.SetElement(0, 1, 0); m.SetElement(0, 2, 0)
      m.SetElement(1, 0, 0); m.SetElement(1, 1, 1); m.SetElement(1, 2, 0)
      m.SetElement(2, 0, 0); m.SetElement(2, 1, 0); m.SetElement(2, 2, 1)
      if sliceOnly:
        position = [0,0, coordinates[2]]
    elif plane == 'SAG':
      m.SetElement(0, 0, 0); m.SetElement(0, 1, 0); m.SetElement(0, 2, 1)
      m.SetElement(1, 0, 0); m.SetElement(1, 1, 1); m.SetElement(1, 2, 0)
      m.SetElement(2, 0, -1); m.SetElement(2, 1, 0); m.SetElement(2, 2, 0)
      if sliceOnly:
        position = [coordinates[0], 0, 0]
    else: #COR
      m.SetElement(0, 0, 1); m.SetElement(0, 1, 0); m.SetElement(0, 2, 0)
      m.SetElement(1, 0, 0); m.SetElement(1, 1, 0); m.SetElement(1, 2, -1)
      m.SetElement(2, 0, 0); m.SetElement(2, 1, 1); m.SetElement(2, 2, 0)
      if sliceOnly:
        position = [0, coordinates[1], 0]
    # Set translation
    m.SetElement(0, 3, position[0])
    m.SetElement(1, 3, position[1])
    m.SetElement(2, 3, position[2])
    self.scanPlaneTransformNode.SetMatrixTransformToParent(m)

  def updateScanPlane(self, checkConfidence=True, plane='COR', sliceOnly=True):
    if checkConfidence:
      confidentUpdate = self.needleConfidenceNode.GetText() != 'None'
    else:
      confidentUpdate = True
    if confidentUpdate:
      # Get current PLAN_0
      plane_matrix = vtk.vtkMatrix4x4()
      self.scanPlaneTransformNode.GetMatrixTransformToParent(plane_matrix)
      # Get current tip transform
      tip_matrix = vtk.vtkMatrix4x4()
      self.tipTrackedNode.GetMatrixTransformToParent(tip_matrix)
      # Update transform with current tip
      if not sliceOnly: # Update all coordinates
          plane_matrix.SetElement(0, 3, tip_matrix.GetElement(0, 3))
          plane_matrix.SetElement(1, 3, tip_matrix.GetElement(1, 3))
          plane_matrix.SetElement(2, 3, tip_matrix.GetElement(2, 3))      
      else:             # Update only slice coordinate
        if plane == 'SAG':
          plane_matrix.SetElement(0, 3, tip_matrix.GetElement(0, 3))
        elif plane == 'COR':
          plane_matrix.SetElement(1, 3, tip_matrix.GetElement(1, 3))
        elif plane == 'AX':
          plane_matrix.SetElement(2, 3, tip_matrix.GetElement(2, 3))
      self.scanPlaneTransformNode.SetMatrixTransformToParent(plane_matrix)
    else:
      print('Scan Plane not updated - No confidence on needle tracking')
    return

  def pushScanPlaneToIGTLink(self, connectionNode):
    #  Push to IGTLink
    connectionNode.RegisterOutgoingMRMLNode(self.scanPlaneTransformNode)
    connectionNode.PushNode(self.scanPlaneTransformNode)
    connectionNode.UnregisterOutgoingMRMLNode(self.scanPlaneTransformNode)

  def pushTargetToIGTLink(self, connectionNode, targetNode):
    # Apply zTransform to currentTip
    self.targetZNode.CopyContent(targetNode)
    dispNode = self.targetZNode.GetDisplayNode()
    if dispNode:
      dispNode.SetVisibility(False)
    self.targetZNode.SetAndObserveTransformNodeID(self.worldToZFrameNode.GetID())
    self.targetZNode.HardenTransform()
    #  Push to IGTLink
    connectionNode.RegisterOutgoingMRMLNode(self.targetZNode)
    connectionNode.PushNode(self.targetZNode)
    connectionNode.UnregisterOutgoingMRMLNode(self.targetZNode)

  def pushTipToIGTLink(self, connectionNode):
    # Apply zTransform to currentTip
    self.tipTrackedZNode.CopyContent(self.tipTrackedNode)
    self.tipTrackedZNode.SetAndObserveTransformNodeID(self.worldToZFrameNode.GetID())
    self.tipTrackedZNode.HardenTransform()
    #  Push to IGTLink:
    # zFrame
    connectionNode.RegisterOutgoingMRMLNode(self.tipTrackedZNode)
    connectionNode.PushNode(self.tipTrackedZNode)
    connectionNode.UnregisterOutgoingMRMLNode(self.tipTrackedZNode)
    # world frame
    connectionNode.RegisterOutgoingMRMLNode(self.tipTrackedNode)
    connectionNode.PushNode(self.tipTrackedNode)
    connectionNode.UnregisterOutgoingMRMLNode(self.tipTrackedNode)

  def getNeedle(self, firstVolume, secondVolume, inputMode, in_channels=2, out_channels=3, window_size=(3,48,48), debugFlag=False):
    # Using only one slice volumes for now
    # TODO: Maybe we will extend to 3 stacked slices? Not sure if will be necessary
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
      self.pushSitkToSlicerVolume(sitk_img_m, 'debug_img_m', debugFlag=debugFlag)
      self.pushSitkToSlicerVolume(sitk_img_p, 'debug_img_p', debugFlag=debugFlag)
      self.saveSitkImage(sitk_img_m, name='debug_img_m_'+str(self.count), path=os.path.join(self.path, 'Debug'))
      self.saveSitkImage(sitk_img_p, name='debug_img_p_'+str(self.count), path=os.path.join(self.path, 'Debug'))
      if self.sitk_mask is not None:
        sitk_mask = sitk.Cast(self.sitk_mask, sitk.sitkUInt8)
        self.pushSitkToSlicerVolume(sitk_mask, 'debug_mask', debugFlag=debugFlag)
        self.saveSitkImage(sitk_mask, name='debug_mask_'+str(self.count), path=os.path.join(self.path, 'Debug'))
        

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

    # Push segmentation to Slicer
    self.pushSitkToSlicerVolume(sitk_output, self.needleLabelMapNode, debugFlag=debugFlag)
    self.saveSitkImage(sitk_output, name='debug_labelmap_'+str(self.count), path=os.path.join(self.path, 'Debug'), is_label=True)

    ######################################
    ##                                  ##
    ## Step 2: Get segmentations        ##
    ##                                  ##
    ######################################

    # Apply segmentation mask (optional)
    if self.sitk_mask is not None:
      self.sitk_mask.SetOrigin(sitk_output.GetOrigin())                  # Update origin (due to A-P change in PLAN_0)
      sitk_output = self.maskFilter.Execute(self.sitk_mask, sitk_output) # Apply mask to labels

    # Separate labels
    sitk_tip = (sitk_output==2)
    sitk_shaft = (sitk_output==1)
    
    center = None
    shaft_tip = None
    
    ######################################
    ##                                  ##
    ## Step 3: Define tip labels        ##
    ##                                  ##
    ######################################    
    
    # Try to select tip from segmentation
    if sitk.GetArrayFromImage(sitk_tip).sum() > 0:
      # Get labels from segmentation
      stats = sitk.LabelShapeStatisticsImageFilter()
      # TODO: With fewer false positives, check if really necessary to compute the radius to check shaft pairing
      # stats.SetComputeFeretDiameter(True)
      stats.Execute(sitk.ConnectedComponent(sitk_tip))
      # Get labels sizes and centroid physical coordinates
      labels_size = []
      labels_centroid = []
      # labels_max_radius = []
      for l in stats.GetLabels():
        number_pixels = stats.GetNumberOfPixels(l)
        centroid = stats.GetCentroid(l)
        # max_radius = stats.GetFeretDiameter(l)
        if debugFlag:
          print('Tip Label %s: -> Size: %s, Center: %s' %(l, number_pixels, centroid))
          # print('Tip Label %s: -> Size: %s, Center: %s, Max Radius: %s' %(l, number_pixels, centroid, max_radius))
        labels_size.append(number_pixels)
        labels_centroid.append(centroid)    
        # labels_max_radius.append(max_radius)
      # Get tip estimate position
      index_largest = labels_size.index(max(labels_size)) # Find index of largest centroid
      if debugFlag:
        print('Selected tip = %s' %str(index_largest+1))
      center = labels_centroid[index_largest]             # Get the largest centroid center
      # max_distance = 1.1*labels_max_radius[index_largest] # Maximum acceptable distance between the tip centroid and the shaft tip
      max_distance = 15
      
    ######################################
    ##                                  ##
    ## Step 4: Define shaft label       ##
    ##                                  ##
    ######################################        
    
    # Try to select shaft from segmentation
    if sitk.GetArrayFromImage(sitk_shaft).sum() > 0:
      # Get labels from segmentation
      stats = sitk.LabelShapeStatisticsImageFilter()
      # TODO: Use oriented bounding box instead of skeleton for getting the shaft limit points (and needle orientation)
      # stats.SetComputeFeretDiameter(True)
      # stats.SetComputeOrientedBoundingBox(True)
      
      # TODO: If shaft is constantly broke: Check need for BinaryClosingByReconstructionImageFilter
      # Looks like it won't be necessary with new model
      # SetKernelRadius([x,y,z])
      # SetFullyConnected(True) # Test
      
      sitk_shaft_labeled = sitk.ConnectedComponent(sitk_shaft)
      stats.Execute(sitk_shaft_labeled)
      # Get labels sizes and centroid physical coordinates
      labels_size = []
      labels_centroid = []
      # labels_max_radius = []
      labels = stats.GetLabels()
      for l in labels:
        number_pixels = stats.GetNumberOfPixels(l)
        centroid = stats.GetCentroid(l)
        # max_radius = stats.GetFeretDiameter(l)
        # obb = stats.GetOrientedBoundingBoxVertices(l)
        # obb_dir = stats.GetOrientedBoundingBoxDirection(l)
        # obb_center = stats.GetOrientedBoundingBoxOrigin(l)
        # obb_size = stats.GetOrientedBoundingBoxSize(l)
        if debugFlag:
          print('Shaft Label %s: -> Size: %s, Center: %s' %(l, number_pixels, centroid)) 
          # print('Bounding box %s: -> Direction: %s, Center: %s, Size: %s' %(l, obb_dir, obb_center, obb_size))
        labels_size.append(number_pixels)
        labels_centroid.append(centroid)    
        # labels_max_radius.append(max_radius)
      
      # Get the largest shaft
      index_largest = labels_size.index(max(labels_size)) # Find index of largest centroid
      label = labels[index_largest]
      # Create a binary mask for the specific label
      sitk_selected_shaft = sitk.BinaryThreshold(sitk_shaft_labeled, lowerThreshold=label, upperThreshold=label)
      # Create a skeleton
      sitk_skeleton = sitk.BinaryThinning(sitk_selected_shaft)
      shaft_tip = self.getShaftTipCoordinates(sitk_skeleton)
      # shaft_direction = self.getShaftDirection()


    ######################################
    ##                                  ##
    ## Step 5: Select tip and shaft     ##
    ##                                  ##
    ######################################    
    
    #TODO: Define best tip/shaft combination from labels

    ######################################
    ##                                  ##
    ## Step 6: Set confidence for tip   ##
    ##                                  ##
    ######################################

    # Define confidence on tip estimate
    if center is None:
      center = shaft_tip      # Use shaft tip (no tip segmentation)
      confidence = 'Low'      # Set estimate as low (no tip segmentation, just shaft)
    elif shaft_tip is None:
      confidence = 'Medium'   # Set estimate as medium (use tip, no shaft segmentation available)
    else:
      distance = np.linalg.norm(np.array(center) - np.array(shaft_tip))  # Calculate distance between tip and shaft
      if distance <= max_distance:                             
        confidence = 'High'   # Set estimate as high (both tip and shaft segmentation and they are close)
      else:
        confidence = 'Medium' # Set estimate as medium (use tip segmentation, but shaft is not close)
    if center is None:
      if debugFlag:
        print('Center coordinates: None')
        print('Confidence: None')
      return None
    
    ####################################
    ##                                ##
    ## Step 7: Push to tipTrackedNode ##
    ##                                ##
    ####################################

    # Convert to 3D Slicer coordinates (RAS)
    centerRAS = (-center[0], -center[1], center[2])     
    # # Plot
    if debugFlag:
      print('Center coordinates: ' + str(centerRAS))
      print('Confidence: ' + confidence)

    # Push coordinates to tip Node
    transformMatrix = vtk.vtkMatrix4x4()
    transformMatrix.SetElement(0,3, centerRAS[0])
    transformMatrix.SetElement(1,3, centerRAS[1])
    transformMatrix.SetElement(2,3, centerRAS[2])
    self.tipTrackedNode.SetMatrixTransformToParent(transformMatrix)

    # Push confidence to Node
    self.needleConfidenceNode.SetText(confidence) 
    return confidence
  
  ############################################
