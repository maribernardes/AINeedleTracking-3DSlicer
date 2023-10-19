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

    separator1 = SeparatorWidget('Tracking setup')
    setupFormLayout.addRow(separator1)
    
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

    separator2 = SeparatorWidget('Optional IGTLink')
    setupFormLayout.addRow(separator2)

    igtlHBoxLayout = qt.QHBoxLayout()   
    # Push tip coordinates to robot
    self.pushTipToRobotCheckBox = qt.QCheckBox()
    self.pushTipToRobotCheckBox.checked = False
    self.pushTipToRobotCheckBox.setToolTip('If checked, pushes current tip position to robot in zFrame coordinates')
    igtlHBoxLayout.addWidget(qt.QLabel('Push Tip to Robot'))
    igtlHBoxLayout.addWidget(self.pushTipToRobotCheckBox)
    
    # UpdateScanPlan mode check box (output images at intermediate steps)
    self.updateScanPlaneCheckBox = qt.QCheckBox()
    self.updateScanPlaneCheckBox.checked = False
    self.updateScanPlaneCheckBox.setToolTip('If checked, updates scan plane with current tip position')
    igtlHBoxLayout.addWidget(qt.QLabel('Update Scan Plane'))
    igtlHBoxLayout.addWidget(self.updateScanPlaneCheckBox)

    setupFormLayout.addRow(igtlHBoxLayout)

    # Select OpenIGTLink connection
    self.connectionSelector = slicer.qMRMLNodeComboBox()
    self.connectionSelector.nodeTypes = ['vtkMRMLIGTLConnectorNode']
    self.connectionSelector.selectNodeUponCreation = True
    self.connectionSelector.addEnabled = False
    self.connectionSelector.removeEnabled = False
    self.connectionSelector.noneEnabled = True
    self.connectionSelector.showHidden = False
    self.connectionSelector.showChildNodeTypes = False
    self.connectionSelector.setMRMLScene(slicer.mrmlScene)
    self.connectionSelector.setToolTip('Select OpenIGTLink connection')
    self.connectionSelector.enabled = False
    setupFormLayout.addRow('OpenIGTLink Server:', self.connectionSelector)

    # Select WorldToZFrame transform
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
    setupFormLayout.addRow('ZFrame Transform:', self.transformSelector)

    # Select which scene view to track
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
    layout = qt.QHBoxLayout()
    layout.addWidget(self.sceneViewButton_red)
    layout.addWidget(self.sceneViewButton_yellow)
    layout.addWidget(self.sceneViewButton_green)
    setupFormLayout.addRow('Scene view:',layout)   
    
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
    self.updateScanPlaneCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.pushTipToRobotCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.sceneViewButton_red.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.sceneViewButton_yellow.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.sceneViewButton_green.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.connectionSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.transformSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.modelFileSelector.connect('currentIndexChanged(str)', self.updateParameterNodeFromGUI)
    
    # Connect UI buttons to event calls
    self.startTrackingButton.connect('clicked(bool)', self.startTracking)
    self.stopTrackingButton.connect('clicked(bool)', self.stopTracking)
    self.firstVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.secondVolumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.updateScanPlaneCheckBox.connect("toggled(bool)", self.updateButtons)
    self.pushTipToRobotCheckBox.connect("toggled(bool)", self.updateButtons)
    self.connectionSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.transformSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    
    # Internal variables
    self.isTrackingOn = False
    self.inputMode = None
    self.firstVolume = None
    self.secondVolume = None
    self.debugFlag = None
    self.updateScanPlane = None
    self.pushTipToRobot = None
    self.serverNode = None
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
    self.inputModeMagPhase.checked = (self._parameterNode.GetParameter('InputMode') == 'MagPhase')
    self.inputModeRealImag.checked = (self._parameterNode.GetParameter('InputMode') == 'RealImag')
    self.debugFlagCheckBox.checked = (self._parameterNode.GetParameter('Debug') == 'True')
    self.updateScanPlaneCheckBox.checked = (self._parameterNode.GetParameter('UpdateScanPlane') == 'True')
    self.pushTipToRobotCheckBox.checked = (self._parameterNode.GetParameter('PushTipToRobot') == 'True')
    self.connectionSelector.setCurrentNode(self._parameterNode.GetNodeReference('Connection'))
    self.transformSelector.setCurrentNode(self._parameterNode.GetNodeReference('zFrame'))
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
    self._parameterNode.SetParameter('InputMode', 'MagPhase' if self.inputModeMagPhase.checked else 'RealImag')
    self._parameterNode.SetParameter('Debug', 'True' if self.debugFlagCheckBox.checked else 'False')
    self._parameterNode.SetParameter('UpdateScanPlane', 'True' if self.updateScanPlaneCheckBox.checked else 'False')
    self._parameterNode.SetParameter('PushTipToRobot', 'True' if self.pushTipToRobotCheckBox.checked else 'False')
    self._parameterNode.SetNodeReferenceID('Connection', self.connectionSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('zFrame', self.transformSelector.currentNodeID)
    self._parameterNode.SetParameter('Model', str(self.modelFileSelector.currentIndex))
    self._parameterNode.EndModify(wasModified)
                        
  # Update button states
  def updateButtons(self):
    # Not tracking = ENABLE SELECTION
    if not self.isTrackingOn:
      # Logic for required variables selection: 
      self.inputModeMagPhase.enabled = True
      self.inputModeRealImag.enabled = True
      self.modelFileSelector.enabled = True
      self.pushTipToRobotCheckBox.enabled = True
      self.updateScanPlaneCheckBox.enabled = True
      self.firstVolumeSelector.enabled = True
      self.secondVolumeSelector.enabled = True
      connectionDefined = True # Not required
      transformDefined = True  # Not required      
      # Logic for optional variables selection
      if self.updateScanPlaneCheckBox.checked:
        self.sceneViewButton_red.enabled = True
        self.sceneViewButton_yellow.enabled = True
        self.sceneViewButton_green.enabled = True    
        self.connectionSelector.enabled = True
        self.transformSelector.enabled = False
        connectionDefined = self.connectionSelector.currentNode()
        transformDefined = True  # Not required
      if self.pushTipToRobotCheckBox.checked:
        if not self.updateScanPlaneCheckBox.checked:
          self.sceneViewButton_red.enabled = False
          self.sceneViewButton_yellow.enabled = False
          self.sceneViewButton_green.enabled = False    
        self.connectionSelector.enabled = True
        self.transformSelector.enabled = True
        connectionDefined = self.connectionSelector.currentNode()
        transformDefined = self.transformSelector.currentNode()
      if (not self.updateScanPlaneCheckBox.checked) and not self.pushTipToRobotCheckBox.checked:
        self.connectionSelector.enabled = False
        self.transformSelector.enabled = False 
    # When tracking = DISABLE SELECTION
    else:
      self.inputModeMagPhase.enabled = False
      self.inputModeRealImag.enabled = False
      self.modelFileSelector.enabled = False
      self.pushTipToRobotCheckBox.enabled = False
      self.updateScanPlaneCheckBox.enabled = False
      self.firstVolumeSelector.enabled = False
      self.secondVolumeSelector.enabled = False
      self.sceneViewButton_red.enabled = False
      self.sceneViewButton_yellow.enabled = False
      self.sceneViewButton_green.enabled = False
      self.connectionSelector.enabled = False
      self.transformSelector.enabled = False
      connectionDefined = True # Not required
      transformDefined = True  # Not required      

    # Check if Tracking is enabled
    rtNodesDefined = self.firstVolumeSelector.currentNode() and self.secondVolumeSelector.currentNode()
    self.startTrackingButton.enabled = rtNodesDefined and connectionDefined and transformDefined and not self.isTrackingOn
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
    layoutManager = slicer.app.layoutManager()
    sliceWidgetLogic = layoutManager.sliceWidget(str(selectedView)).sliceLogic()
    # Get slice index and volume node
    sliceIndex = sliceWidgetLogic.GetSliceIndexFromOffset(sliceWidgetLogic.GetSliceOffset()) - 1
    compositeNode = sliceWidgetLogic.GetSliceCompositeNode()
    volumeNode = slicer.util.getNode(compositeNode.GetBackgroundVolumeID())
    # Get image from volumeNode
    sitk_image = sitkUtils.PullVolumeFromSlicer(volumeNode)
    # Get the slice center coordinates
    image_size = sitk_image.GetSize()
    centerIndex = (int(image_size[0]/2), int(image_size[1]/2), sliceIndex)
    centerLPS = sitk_image.TransformIndexToPhysicalPoint(centerIndex)
    centerRAS = (-centerLPS[0], -centerLPS[1], centerLPS[2])   
    return centerRAS

  def startTracking(self):
    print('UI: startTracking()')
    self.isTrackingOn = True
    self.updateButtons()
    # Store selected parameters
    model = self.modelFileSelector.currentText
    self.updateScanPlane = self.updateScanPlaneCheckBox.checked 
    self.pushTipToRobot = self.pushTipToRobotCheckBox.checked 
    self.firstVolume = self.firstVolumeSelector.currentNode()
    self.secondVolume = self.secondVolumeSelector.currentNode() 
    self.inputMode = 'MagPhase' if self.inputModeMagPhase.checked else 'RealImag'
    self.serverNode = self.connectionSelector.currentNode()
    self.zFrameTransform = self.transformSelector.currentNode()
    # Initialize tracking logic
    self.logic.initializeTracking(model)
    # Initialize PLAN_0
    if self.updateScanPlane == True:
      center = self.getSelectetViewCenterCoordinates(self.getSelectedView())
      self.logic.initializeScanPlane(center, plane='COR') # Reinitialize PLAN_0 at center position
    # Initialize zFrame transform
    if self.pushTipToRobot == True:
      self.logic.initializePushTip(self.zFrameTransform)
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
          self.logic.updateScanPlane()
          self.logic.pushScanPlaneToIGTLink(self.serverNode)
          print('PLAN_0 updated')
        if self.pushTipToRobot is True:
          self.logic.pushTipToIGTLink(self.serverNode)
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

    # Used for saving data from experiments
    self.count = None

    # Check if PLANE_0 node exists, if not, create a new one
    try:
        self.scanPlaneTransformNode = slicer.util.getNode('PLANE_0')
        self.initializeScanPlane(plane='COR')
    except:
      self.scanPlaneTransformNode = slicer.vtkMRMLLinearTransformNode()
      self.scanPlaneTransformNode.SetName('PLANE_0')
      slicer.mrmlScene.AddNode(self.scanPlaneTransformNode)
      self.initializeScanPlane(plane='COR')

    # Check if labelmap node exists, if not, create a new one
    try:
        self.needleLabelMapNode = slicer.util.getNode('NeedleLabelMap')
    except:
        self.needleLabelMapNode = slicer.vtkMRMLLabelMapVolumeNode()
        self.needleLabelMapNode.SetName('NeedleLabelMap')
        slicer.mrmlScene.AddNode(self.needleLabelMapNode)
        colorTableNode = self.createColorTable()
        self.needleLabelMapNode.CreateDefaultDisplayNodes()
        self.needleLabelMapNode.GetDisplayNode().SetAndObserveColorNodeID(colorTableNode.GetID())

    # Check if text node exists, if not, create a new one
    try:
        self.needleConfidenceNode = slicer.util.getNode('CurrentTipConfidence')
    except:
        self.needleConfidenceNode = slicer.vtkMRMLTextNode()
        self.needleConfidenceNode.SetName('CurrentTipConfidence')
        slicer.mrmlScene.AddNode(self.needleConfidenceNode)
        
    # Check if tracked tip node exists, if not, create a new one
    try:
        self.tipTrackedNode = slicer.util.getNode('CurrentTrackedTipTransform')
    except:
        self.tipTrackedNode = slicer.vtkMRMLLinearTransformNode()
        self.tipTrackedNode.SetName('CurrentTrackedTipTransform')
        slicer.mrmlScene.AddNode(self.tipTrackedNode)

    # Check if zFrame tracked tip node exists, if not, create a new one
    try:
        self.tipTrackedZNode = slicer.util.getNode('CurrentTrackedTipZ')
    except:
        self.tipTrackedZNode = slicer.vtkMRMLLinearTransformNode()
        self.tipTrackedZNode.SetName('CurrentTrackedTipZ')
        self.tipTrackedZNode.SetHideFromEditors(True)
        slicer.mrmlScene.AddNode(self.tipTrackedZNode)

    # Check if WorldToZFrame transform node exists, if not, create a new one
    try:
        self.worldToZFrameNode = slicer.util.getNode('WorldToZFrame')
    except:
        self.worldToZFrameNode = slicer.vtkMRMLLinearTransformNode()
        self.worldToZFrameNode.SetName('WorldToZFrame')
        self.worldToZFrameNode.SetHideFromEditors(True)
        slicer.mrmlScene.AddNode(self.worldToZFrameNode)
        
  # Initialize parameter node with default settings
  def setDefaultParameters(self, parameterNode):
    if not parameterNode.GetParameter('Debug'):
      parameterNode.SetParameter('Debug', 'False')  
    if not parameterNode.GetParameter('UpdateScanPlane'): 
      parameterNode.SetParameter('UpdateScanPlane', 'False')  
    if not parameterNode.GetParameter('PushTipToRobot'): 
      parameterNode.SetParameter('PushTipToRobot', 'False')  
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
  def initializeTracking(self, model):
    self.setupUNet(model) # Setup UNet
    self.count = 0        # Initialize sequence counter

  def initializePushTip(self, zFrameToWorld):
    # Get world to ZFrame transformations
    worldToZFrame = vtk.vtkMatrix4x4()
    zFrameToWorld.GetMatrixTransformFromWorld(worldToZFrame)
    # Set it to worldToZFrameNode
    self.worldToZFrameNode.SetMatrixTransformToParent(worldToZFrame)
  
  # Set Scan Plane Orientation (position is zero)
  def initializeScanPlane(self, center=(0,0,0), plane='COR'):
    m = vtk.vtkMatrix4x4()
    self.scanPlaneTransformNode.GetMatrixTransformToParent(m)
    # Set rotation
    if plane == 'AX':
      m.SetElement(0, 0, 1); m.SetElement(0, 1, 0); m.SetElement(0, 2, 0)
      m.SetElement(1, 0, 0); m.SetElement(1, 1, 1); m.SetElement(1, 2, 0)
      m.SetElement(2, 0, 0); m.SetElement(2, 1, 0); m.SetElement(2, 2, 1)
    elif plane == 'SAG':
      m.SetElement(0, 0, 0); m.SetElement(0, 1, 0); m.SetElement(0, 2, 1)
      m.SetElement(1, 0, 0); m.SetElement(1, 1, 1); m.SetElement(1, 2, 0)
      m.SetElement(2, 0, -1); m.SetElement(2, 1, 0); m.SetElement(2, 2, 0)
    else: #COR
      m.SetElement(0, 0, 1); m.SetElement(0, 1, 0); m.SetElement(0, 2, 0)
      m.SetElement(1, 0, 0); m.SetElement(1, 1, 0); m.SetElement(1, 2, -1)
      m.SetElement(2, 0, 0); m.SetElement(2, 1, 1); m.SetElement(2, 2, 0)
    # Set translation
    m.SetElement(0, 3, center[0])
    m.SetElement(1, 3, center[1])
    m.SetElement(2, 3, center[2])
    self.scanPlaneTransformNode.SetMatrixTransformToParent(m)

  def updateScanPlane(self):
    if self.needleConfidenceNode.GetText() != 'None':
      # Get current PLAN_0
      plane_matrix = vtk.vtkMatrix4x4()
      self.scanPlaneTransformNode.GetMatrixTransformToParent(plane_matrix)
      # Get current tip transform
      tip_matrix = vtk.vtkMatrix4x4()
      self.tipTrackedNode.GetMatrixTransformToParent(tip_matrix)
      # Update transform with current tip
      plane_matrix.SetElement(0, 3, tip_matrix.GetElement(0, 3))
      plane_matrix.SetElement(1, 3, tip_matrix.GetElement(1, 3))
      plane_matrix.SetElement(2, 3, tip_matrix.GetElement(2, 3))
      self.scanPlaneTransformNode.SetMatrixTransformToParent(plane_matrix)
    else:
      print('Scan Plane not updated - No confidence on needle tracking')
    return
  
  def pushTipToIGTLink(self, connectionNode):
    # Apply zTransform to currentTip
    self.tipTrackedZNode.CopyContent(self.tipTrackedNode)
    self.tipTrackedZNode.SetAndObserveTransformNodeID(self.worldToZFrameNode.GetID())
    self.tipTrackedZNode.HardenTransform()
    #  Push to IGTLink Server
    connectionNode.RegisterOutgoingMRMLNode(self.tipTrackedZNode)
    connectionNode.PushNode(self.tipTrackedZNode)

  def pushScanPlaneToIGTLink(self, connectionNode):
    #  Push to IGTLink Server
    connectionNode.RegisterOutgoingMRMLNode(self.scanPlaneTransformNode)
    connectionNode.PushNode(self.scanPlaneTransformNode)

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
    ## Step 2: Get coordinates for tip  ##
    ##                                  ##
    ######################################

    # Separate labels
    sitk_tip = (sitk_output==2)
    sitk_shaft = (sitk_output==1)

    center = None
    shaft_tip = None
    # Try to get tip estimate from tip segmentation
    if sitk.GetArrayFromImage(sitk_tip).sum() > 0:
      # Get blobs from segmentation
      stats = sitk.LabelShapeStatisticsImageFilter()
      stats.SetComputeFeretDiameter(True)
      stats.Execute(sitk.ConnectedComponent(sitk_tip))
      # Get blobs sizes and centroid physical coordinates
      labels_size = []
      labels_centroid = []
      labels_max_radius = []
      for l in stats.GetLabels():
        number_pixels = stats.GetNumberOfPixels(l)
        centroid = stats.GetCentroid(l)
        max_radius = stats.GetFeretDiameter(l)
        if debugFlag:
          print('Label %s: -> Size: %s, Center: %s, Max Radius: %s' %(l, number_pixels, centroid, max_radius))
        labels_size.append(number_pixels)
        labels_centroid.append(centroid)    
        labels_max_radius.append(max_radius)
      # Get tip estimate position
      index_largest = labels_size.index(max(labels_size)) # Find index of largest centroid
      center = labels_centroid[index_largest]             # Get the largest centroid center
      max_distance = 1.1*labels_max_radius[index_largest] # Maximum acceptable distance between the tip centroid and the shaft tip
    # Try to get tip estimate from shaft segmentation
    if sitk.GetArrayFromImage(sitk_shaft).sum() > 0:
      sitk_skeleton = sitk.BinaryThinning(sitk_shaft)
      shaft_tip = self.getShaftTipCoordinates(sitk_skeleton)
      # shaft_direction = self.getShaftDirection()


    ######################################
    ##                                  ##
    ## Step 3: Set confidence for tip   ##
    ##                                  ##
    ######################################

    # Define confidence on tip estimate
    if center is None:
      center = shaft_tip                            # Use shaft tip (no tip segmentation)
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
    ## Step 4: Push to tipTrackedNode ##
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