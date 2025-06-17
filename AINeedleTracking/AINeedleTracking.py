# TODO: Only enable tracking when model is selected
# TODO: Allow tracking without openIGTLink connection


import logging
import os
import time
import copy

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import SimpleITK as sitk
import sitkUtils
import numpy as np

from math import sqrt, pow
import statistics

import torch
from monaiUtils.sitkMonaiIO import LoadSitkImaged, PushSitkImaged
from monai.transforms import Compose, ConcatItemsd, EnsureChannelFirstd, ScaleIntensityd, Orientationd, Spacingd
from monai.transforms import Invertd, Activationsd, AsDiscreted, KeepLargestConnectedComponentd, RemoveSmallObjectsd
from monai.networks.nets import UNet 
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.handlers.utils import from_engine
from skimage.restoration import unwrap_phase


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
# Timestamp class
################################################################################################################################################

class TimestampTracker:
  def __init__(self, step_names):
    self.step_names = step_names
    self.timestamps = {name: [] for name in step_names}

  def clear(self):
    for key in self.timestamps:
      self.timestamps[key] = []

  def mark(self, step_name):
    #now = time.perf_counter()
    now = time.time()
    self.timestamps[step_name].append(now)

  def elapsed_ms(self, step1, step2, cycle=-1):
    t1 = self.timestamps[step1][cycle]
    t2 = self.timestamps[step2][cycle]
    return (t2 - t1) * 1000

  def print_pairwise_durations(self, step_pairs):
      all_durations = {pair: [] for pair in step_pairs}
      for key1, key2 in step_pairs:
        len1 = len(self.timestamps.get(key1, []))
        len2 = len(self.timestamps.get(key2, []))
        print(f"{key1}: {len1} values, {key2}: {len2} values")
      
      # Get number of cycles based on any step's timestamp count
      try:
        num_cycles = len(next(iter(self.timestamps.values())))
      except StopIteration:
        print("No timestamps available.")
        return
      # Per-cycle durations
      for i in range(num_cycles):
        print(f"\nImage #{i+1}")
        for s1, s2 in step_pairs:
          try:
            delta = self.elapsed_ms(s1, s2, cycle=i)
            all_durations[(s1, s2)].append(delta)
            print(f"{s1} → {s2}: {delta:.2f} ms")
          except (IndexError, KeyError):
            print(f"{s1} → {s2}: not enough data")
      # Summary stats
      print("\n=== Mean and Std Deviation ===")
      for s1, s2 in step_pairs:
        durations = all_durations[(s1, s2)]
        if durations:
          mean_val = statistics.mean(durations)
          std_val = statistics.stdev(durations) if len(durations) > 1 else 0
          print(f"{s1} → {s2}: mean = {mean_val:.2f} ms | std = {std_val:.2f} ms")
        else:
          print(f"{s1} → {s2}: no data")

################################################################################################################################################
# Custom Widget  - Separator
################################################################################################################################################
class SeparatorWidget(qt.QWidget):
    def __init__(self, label_text='Separator Widget Label', useLine=True, parent=None):
        super().__init__(parent)

        spacer = qt.QWidget()
        spacer.setFixedHeight(10)
        
        self.label = qt.QLabel(label_text)
        font = qt.QFont()
        font.setItalic(True)
        self.label.setFont(font)
        
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(qt.Qt.AlignVCenter)
        layout.addWidget(spacer)
        layout.addWidget(self.label)
        if useLine:
          line = qt.QFrame()
          line.setFrameShape(qt.QFrame.HLine)
          line.setFrameShadow(qt.QFrame.Sunken)
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

    # Set validator for floating points
    regex = qt.QRegularExpression(r"^-?[0-9]*\.?[0-9]*$")# Regular expression for numbers and dots (e.g., 123.45)
    self.floatValidator = qt.QRegularExpressionValidator(regex)
    self.path = os.path.dirname(os.path.abspath(__file__))

    ## Model                
    ####################################
    
    setupCollapsibleButton = ctk.ctkCollapsibleButton()
    setupCollapsibleButton.text = 'Setup'
    self.layout.addWidget(setupCollapsibleButton)
    setupFormLayout = qt.QFormLayout(setupCollapsibleButton)
    
    #### Model configuration ####
    sectionModel = SeparatorWidget('Model Selection')
    setupFormLayout.addRow(sectionModel)    
    inputHBoxLayout1 = qt.QHBoxLayout()
    hSpacer = qt.QSpacerItem(20, 20, qt.QSizePolicy.Fixed, qt.QSizePolicy.Minimum)

    inputHBoxLayout1.addWidget(qt.QLabel('Mode:'))
    self.inputModeMagPhase = qt.QRadioButton('Mag/Phase')
    self.inputModeRealImag = qt.QRadioButton('Real/Imag')
    self.inputModeMagPhase.checked = 1
    self.inputModeButtonGroup = qt.QButtonGroup()
    self.inputModeButtonGroup.addButton(self.inputModeMagPhase)
    self.inputModeButtonGroup.addButton(self.inputModeRealImag)    
    inputHBoxLayout1.addWidget(self.inputModeMagPhase)
    inputHBoxLayout1.addWidget(self.inputModeRealImag)
    # inputHBoxLayout1.addItem(hSpacer)
    inputHBoxLayout1.addStretch(1)

    inputHBoxLayout1.addWidget(qt.QLabel('Volume:'))
    self.inputVolume2D = qt.QRadioButton('2D (single slice)')
    self.inputVolume3D = qt.QRadioButton('3D (stack)')
    self.inputVolume2D.checked = 1
    self.inputVolumeButtonGroup = qt.QButtonGroup()
    self.inputVolumeButtonGroup.addButton(self.inputVolume2D)
    self.inputVolumeButtonGroup.addButton(self.inputVolume3D)
    inputHBoxLayout1.addWidget(self.inputVolume2D)
    inputHBoxLayout1.addWidget(self.inputVolume3D)
    # inputHBoxLayout1.addItem(hSpacer)
    inputHBoxLayout1.addStretch(1)
    # setupFormLayout.addRow(inputHBoxLayout1)

    # inputHBoxLayout2 = qt.QHBoxLayout()
    inputHBoxLayout1.addWidget(qt.QLabel('Channels:'))
    self.inputChannels1 = qt.QRadioButton('1 CH')
    self.inputChannels2 = qt.QRadioButton('2 CH')
    self.inputChannels3 = qt.QRadioButton('3 CH')
    self.inputChannels2.checked = 1
    self.inputChannelsButtonGroup = qt.QButtonGroup()
    self.inputChannelsButtonGroup.addButton(self.inputChannels1)
    self.inputChannelsButtonGroup.addButton(self.inputChannels2)
    self.inputChannelsButtonGroup.addButton(self.inputChannels3)
    inputHBoxLayout1.addWidget(self.inputChannels1)
    inputHBoxLayout1.addWidget(self.inputChannels2)
    inputHBoxLayout1.addWidget(self.inputChannels3)
    # inputHBoxLayout1.addStretch(1)
    setupFormLayout.addRow(inputHBoxLayout1)
    
    self.modelFileSelector = qt.QComboBox()
    self.updateModelList()
    setupFormLayout.addRow('AI Model:',self.modelFileSelector)

    #### MRI Inputs ####
    sectionScannerInput = SeparatorWidget('MRI Inputs')
    setupFormLayout.addRow(sectionScannerInput)

    scannerInputHBoxLayout = qt.QHBoxLayout()
    
    # Select MRI Bridge OpenIGTLink connection
    bridgeConnectionLabel = qt.QLabel('IGTLServer MRI:')
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
    scannerInputHBoxLayout.addWidget(bridgeConnectionLabel)  
    scannerInputHBoxLayout.addWidget(self.bridgeConnectionSelector)  
    scannerInputHBoxLayout.addItem(hSpacer)

    # Scanner Mode check box
    scannerModeLabel = qt.QLabel('Mode:')
    self.scannerModeMagPhase = qt.QRadioButton('Mag/Phase')
    self.scannerModeRealImag = qt.QRadioButton('Real/Imag')
    self.scannerModeMagPhase.checked = 1
    self.scannerModeButtonGroup = qt.QButtonGroup()
    self.scannerModeButtonGroup.addButton(self.scannerModeMagPhase)
    self.scannerModeButtonGroup.addButton(self.scannerModeRealImag)    
  
    scannerInputHBoxLayout.addWidget(scannerModeLabel)
    scannerInputHBoxLayout.addWidget(self.scannerModeMagPhase)
    scannerInputHBoxLayout.addWidget(self.scannerModeRealImag)

    setupFormLayout.addRow(scannerInputHBoxLayout)

    #### MRI Scan Planes ####
    sectionScanPlane = SeparatorWidget('MRI Scan Planes')
    setupFormLayout.addRow(sectionScanPlane)

    # Create the main grid layout
    scanPlanesGridLayout = qt.QGridLayout()

    # Select which scan planes to use
    selectPlanesTitle = qt.QLabel('Select Scan Planes')
    selectPlanesTitle.setAlignment(qt.Qt.AlignCenter)  # Center-align the title
    # selectPlanesTitle.setStyleSheet('text-decoration: underline;')
    scanPlanesGridLayout.addWidget(selectPlanesTitle, 0, 0, 1, 2)  # Span 2 columns for centering

    # UpdateScanPlan check box
    autoUpdateLabel = qt.QLabel('Auto update')
    self.updateScanPlaneCheckBox = qt.QCheckBox()
    self.updateScanPlaneCheckBox.checked = False
    self.updateScanPlaneCheckBox.setToolTip('If checked, updates scan plane automatically with current tip position')

    # CenterAtTip check box
    centerAtTipLabel = qt.QLabel('Center at tip')
    self.centerAtTipCheckBox = qt.QCheckBox()
    self.centerAtTipCheckBox.checked = False
    self.centerAtTipCheckBox.setToolTip('If checked, centers scan plane at current tip position')    

    scanPlanesGridLayout.addWidget(autoUpdateLabel, 0, 9, 1, 3, qt.Qt.AlignRight)  
    scanPlanesGridLayout.addWidget(self.updateScanPlaneCheckBox, 0, 12, 1, 3, qt.Qt.AlignLeft)  
    scanPlanesGridLayout.addWidget(centerAtTipLabel, 0, 14)  
    scanPlanesGridLayout.addWidget(self.centerAtTipCheckBox, 0, 15)  

    plane0Label = qt.QLabel('PLANE_0 (COR):')
    plane0Label.setFixedWidth(105)
    self.usePlane0CheckBox = qt.QCheckBox()
    self.usePlane0CheckBox.checked = False
    self.usePlane0CheckBox.setToolTip('If checked, uses CORONAL scan plane')    
    plane1Label = qt.QLabel('PLANE_1 (SAG):')
    plane1Label.setFixedWidth(105)
    self.usePlane1CheckBox = qt.QCheckBox()
    self.usePlane1CheckBox.checked = False
    self.usePlane1CheckBox.setToolTip('If checked, uses SAGITTAL scan plane')    
    plane2Label = qt.QLabel('PLANE_2 (AX):')
    plane2Label.setFixedWidth(105)
    self.usePlane2CheckBox = qt.QCheckBox()
    self.usePlane2CheckBox.checked = False
    self.usePlane2CheckBox.setToolTip('If checked, uses AXIAL scan plane')    

    scanPlanesGridLayout.addWidget(plane0Label, 2, 0)
    scanPlanesGridLayout.addWidget(self.usePlane0CheckBox, 2, 1)
    scanPlanesGridLayout.addWidget(plane1Label, 3, 0)
    scanPlanesGridLayout.addWidget(self.usePlane1CheckBox, 3, 1)
    scanPlanesGridLayout.addWidget(plane2Label, 4, 0)
    scanPlanesGridLayout.addWidget(self.usePlane2CheckBox, 4, 1)

    # Add a vertical separator
    separator = qt.QFrame()
    separator.setFrameShape(qt.QFrame.VLine)  # Vertical line
    separator.setFrameShadow(qt.QFrame.Sunken)  # Sunken style for 3D effect
    scanPlanesGridLayout.addWidget(separator, 0, 2, 5, 1)  # Spans 3 rows

    # Select which scene view to initialize PLAN_0 and send to scanner
    setPlanesTitle = qt.QLabel('Update Scan Planes')
    setPlanesTitle.setAlignment(qt.Qt.AlignCenter)  # Center-align the title
    # setPlanesTitle.setStyleSheet('text-decoration: underline;')
    scanPlanesGridLayout.addWidget(setPlanesTitle, 0, 5, 1, 6)  # Span 2 columns for centering


    # Plane 0 Configuration
    setPlane0Label = qt.QLabel('Set PLANE_0:')
    setPlane0Label.setToolTip('Set position for PLANE_0')
    self.setPlane0Button_ras = qt.QRadioButton('RAS')
    self.setPlane0Button_view = qt.QRadioButton('Viewer')
    self.setPlane0Button_ras.checked = 1
    self.setPlane0ButtonGroup = qt.QButtonGroup()
    self.setPlane0ButtonGroup.addButton(self.setPlane0Button_ras)
    self.setPlane0ButtonGroup.addButton(self.setPlane0Button_view)
    self.rPlane0Textbox = qt.QLineEdit()
    self.aPlane0Textbox = qt.QLineEdit()
    self.sPlane0Textbox = qt.QLineEdit()
    self.rPlane0Textbox.setFixedWidth(50)
    self.aPlane0Textbox.setFixedWidth(50)
    self.sPlane0Textbox.setFixedWidth(50)
    self.rPlane0Textbox.setPlaceholderText('R')
    self.aPlane0Textbox.setPlaceholderText('A')
    self.sPlane0Textbox.setPlaceholderText('S')
    self.rPlane0Textbox.setReadOnly(False)
    self.aPlane0Textbox.setReadOnly(False)
    self.sPlane0Textbox.setReadOnly(False)
    self.rPlane0Textbox.setValidator(self.floatValidator)
    self.aPlane0Textbox.setValidator(self.floatValidator)
    self.sPlane0Textbox.setValidator(self.floatValidator)
    self.rPlane0Textbox.enabled = False
    self.aPlane0Textbox.enabled = False
    self.sPlane0Textbox.enabled = False
    self.scenePlane0Button_red = qt.QRadioButton('Red')
    self.scenePlane0Button_yellow = qt.QRadioButton('Yellow')
    self.scenePlane0Button_green = qt.QRadioButton('Green')
    self.scenePlane0Button_green.checked = 1
    self.scenePlane0ButtonGroup = qt.QButtonGroup()
    self.scenePlane0ButtonGroup.addButton(self.scenePlane0Button_red)
    self.scenePlane0ButtonGroup.addButton(self.scenePlane0Button_yellow)
    self.scenePlane0ButtonGroup.addButton(self.scenePlane0Button_green)
    self.scenePlane0Button_red.enabled = False
    self.scenePlane0Button_yellow.enabled = False
    self.scenePlane0Button_green.enabled = False
    self.sendPlane0Button = qt.QPushButton('Send PLANE_0')
    self.sendPlane0Button.toolTip = 'Send PLANE_0 to scanner'
    self.sendPlane0Button.setFixedWidth(100)
    self.sendPlane0Button.enabled = False

    scanPlanesGridLayout.addWidget(setPlane0Label, 2, 3)
    scanPlanesGridLayout.addWidget(self.setPlane0Button_ras, 2, 4)
    scanPlanesGridLayout.addWidget(self.setPlane0Button_view, 2, 5)
    scanPlanesGridLayout.addItem(hSpacer, 2, 6)
    scanPlanesGridLayout.addWidget(self.rPlane0Textbox, 2, 7)
    scanPlanesGridLayout.addWidget(self.aPlane0Textbox, 2, 8)
    scanPlanesGridLayout.addWidget(self.sPlane0Textbox, 2, 9)
    scanPlanesGridLayout.addWidget(self.scenePlane0Button_red, 2, 10)
    scanPlanesGridLayout.addWidget(self.scenePlane0Button_yellow, 2, 11)
    scanPlanesGridLayout.addWidget(self.scenePlane0Button_green, 2, 12)
    scanPlanesGridLayout.addItem(hSpacer, 2, 13)
    scanPlanesGridLayout.addWidget(self.sendPlane0Button, 2, 14, 1, 2)
 
    # Select which scene view to initialize PLAN_1 and send to scanner
    setPlane1Label = qt.QLabel('Set PLANE_1:')
    setPlane1Label.setToolTip('Set position for PLANE_1')
    self.setPlane1Button_ras = qt.QRadioButton('RAS')
    self.setPlane1Button_view = qt.QRadioButton('Viewer')
    self.setPlane1Button_ras.checked = 1
    self.setPlane1ButtonGroup = qt.QButtonGroup()
    self.setPlane1ButtonGroup.addButton(self.setPlane1Button_ras)
    self.setPlane1ButtonGroup.addButton(self.setPlane1Button_view)
    self.rPlane1Textbox = qt.QLineEdit()
    self.aPlane1Textbox = qt.QLineEdit()
    self.sPlane1Textbox = qt.QLineEdit()
    self.rPlane1Textbox.setFixedWidth(50)
    self.aPlane1Textbox.setFixedWidth(50)
    self.sPlane1Textbox.setFixedWidth(50)
    self.rPlane1Textbox.setPlaceholderText('R')
    self.aPlane1Textbox.setPlaceholderText('A')
    self.sPlane1Textbox.setPlaceholderText('S')
    self.rPlane1Textbox.setReadOnly(False)
    self.aPlane1Textbox.setReadOnly(False)
    self.sPlane1Textbox.setReadOnly(False)
    self.rPlane1Textbox.setValidator(self.floatValidator)
    self.aPlane1Textbox.setValidator(self.floatValidator)
    self.sPlane1Textbox.setValidator(self.floatValidator)
    self.rPlane1Textbox.enabled = False
    self.aPlane1Textbox.enabled = False
    self.sPlane1Textbox.enabled = False
    self.scenePlane1Button_red = qt.QRadioButton('Red')
    self.scenePlane1Button_yellow = qt.QRadioButton('Yellow')
    self.scenePlane1Button_green = qt.QRadioButton('Green')
    self.scenePlane1Button_green.checked = 1
    self.scenePlane1ButtonGroup = qt.QButtonGroup()
    self.scenePlane1ButtonGroup.addButton(self.scenePlane1Button_red)
    self.scenePlane1ButtonGroup.addButton(self.scenePlane1Button_yellow)
    self.scenePlane1ButtonGroup.addButton(self.scenePlane1Button_green)
    self.scenePlane1Button_red.enabled = False
    self.scenePlane1Button_yellow.enabled = False
    self.scenePlane1Button_green.enabled = False
    self.sendPlane1Button = qt.QPushButton('Send PLANE_1')
    self.sendPlane1Button.toolTip = 'Send PLANE_1 to scanner'
    self.sendPlane1Button.setFixedWidth(100)
    self.sendPlane1Button.enabled = False

    scanPlanesGridLayout.addWidget(setPlane1Label, 3, 3)
    scanPlanesGridLayout.addWidget(self.setPlane1Button_ras, 3, 4)
    scanPlanesGridLayout.addWidget(self.setPlane1Button_view, 3, 5)
    scanPlanesGridLayout.addItem(hSpacer, 3, 6)
    scanPlanesGridLayout.addWidget(self.rPlane1Textbox, 3, 7)
    scanPlanesGridLayout.addWidget(self.aPlane1Textbox, 3, 8)
    scanPlanesGridLayout.addWidget(self.sPlane1Textbox, 3, 9)
    scanPlanesGridLayout.addWidget(self.scenePlane1Button_red, 3, 10)
    scanPlanesGridLayout.addWidget(self.scenePlane1Button_yellow, 3, 11)
    scanPlanesGridLayout.addWidget(self.scenePlane1Button_green, 3, 12)
    scanPlanesGridLayout.addItem(hSpacer, 3, 13)
    scanPlanesGridLayout.addWidget(self.sendPlane1Button, 3, 14, 1, 2)

    # Select which scene view to initialize PLAN_2 and send to scanner
    setPlane2Label = qt.QLabel('Set PLANE_2:')
    setPlane2Label.setToolTip('Set position for PLANE_2')
    self.setPlane2Button_ras = qt.QRadioButton('RAS')
    self.setPlane2Button_view = qt.QRadioButton('Viewer')
    self.setPlane2Button_ras.checked = 1
    self.setPlane2ButtonGroup = qt.QButtonGroup()
    self.setPlane2ButtonGroup.addButton(self.setPlane2Button_ras)
    self.setPlane2ButtonGroup.addButton(self.setPlane2Button_view)
    self.rPlane2Textbox = qt.QLineEdit()
    self.aPlane2Textbox = qt.QLineEdit()
    self.sPlane2Textbox = qt.QLineEdit()
    self.rPlane2Textbox.setFixedWidth(50)
    self.aPlane2Textbox.setFixedWidth(50)
    self.sPlane2Textbox.setFixedWidth(50)
    self.rPlane2Textbox.setPlaceholderText('R')
    self.aPlane2Textbox.setPlaceholderText('A')
    self.sPlane2Textbox.setPlaceholderText('S')
    self.rPlane2Textbox.setReadOnly(False)
    self.aPlane2Textbox.setReadOnly(False)
    self.sPlane2Textbox.setReadOnly(False)
    self.rPlane2Textbox.setValidator(self.floatValidator)
    self.aPlane2Textbox.setValidator(self.floatValidator)
    self.sPlane2Textbox.setValidator(self.floatValidator)
    self.rPlane2Textbox.enabled = False
    self.aPlane2Textbox.enabled = False
    self.sPlane2Textbox.enabled = False
    self.scenePlane2Button_red = qt.QRadioButton('Red')
    self.scenePlane2Button_yellow = qt.QRadioButton('Yellow')
    self.scenePlane2Button_green = qt.QRadioButton('Green')
    self.scenePlane2Button_green.checked = 1
    self.scenePlane2ButtonGroup = qt.QButtonGroup()
    self.scenePlane2ButtonGroup.addButton(self.scenePlane2Button_red)
    self.scenePlane2ButtonGroup.addButton(self.scenePlane2Button_yellow)
    self.scenePlane2ButtonGroup.addButton(self.scenePlane2Button_green)
    self.scenePlane2Button_red.enabled = False
    self.scenePlane2Button_yellow.enabled = False
    self.scenePlane2Button_green.enabled = False
    self.sendPlane2Button = qt.QPushButton('Send PLANE_2')
    self.sendPlane2Button.toolTip = 'Send PLANE_2 to scanner'
    self.sendPlane2Button.setFixedWidth(100)
    self.sendPlane2Button.enabled = False

    scanPlanesGridLayout.addWidget(setPlane2Label, 4, 3)
    scanPlanesGridLayout.addWidget(self.setPlane2Button_ras, 4, 4)
    scanPlanesGridLayout.addWidget(self.setPlane2Button_view, 4, 5)
    scanPlanesGridLayout.addItem(hSpacer, 4, 6)
    scanPlanesGridLayout.addWidget(self.rPlane2Textbox, 4, 7)
    scanPlanesGridLayout.addWidget(self.aPlane2Textbox, 4, 8)
    scanPlanesGridLayout.addWidget(self.sPlane2Textbox, 4, 9)
    scanPlanesGridLayout.addWidget(self.scenePlane2Button_red, 4, 10)
    scanPlanesGridLayout.addWidget(self.scenePlane2Button_yellow, 4, 11)
    scanPlanesGridLayout.addWidget(self.scenePlane2Button_green, 4, 12)
    scanPlanesGridLayout.addItem(hSpacer, 4, 13)
    scanPlanesGridLayout.addWidget(self.sendPlane2Button, 4, 14, 1, 2)

    scanPlanesWidget = qt.QWidget()
    scanPlanesWidget.setLayout(scanPlanesGridLayout)
    setupFormLayout.addRow(scanPlanesWidget)
    
    ## Needle Tracking                
    ####################################

    trackingCollapsibleButton = ctk.ctkCollapsibleButton()
    trackingCollapsibleButton.text = 'Tracking'
    self.layout.addWidget(trackingCollapsibleButton)
    trackingFormLayout = qt.QFormLayout(trackingCollapsibleButton)

    sectionPlane0 = SeparatorWidget('PLANE_0 (COR)')
    trackingFormLayout.addRow(sectionPlane0)

    # Input magnitude/real volume (first volume)
    self.firstVolumePlane0Selector = slicer.qMRMLNodeComboBox()
    self.firstVolumePlane0Selector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.firstVolumePlane0Selector.selectNodeUponCreation = True
    self.firstVolumePlane0Selector.addEnabled = True
    self.firstVolumePlane0Selector.removeEnabled = True
    self.firstVolumePlane0Selector.noneEnabled = True
    self.firstVolumePlane0Selector.showHidden = False
    self.firstVolumePlane0Selector.showChildNodeTypes = False
    self.firstVolumePlane0Selector.setMRMLScene(slicer.mrmlScene)
    self.firstVolumePlane0Selector.setToolTip('Select the magnitude/real image')
    trackingFormLayout.addRow('Magnitude/Real: ', self.firstVolumePlane0Selector)

    # Input phase/imaginary volume (second volume)
    self.secondVolumePlane0Selector = slicer.qMRMLNodeComboBox()
    self.secondVolumePlane0Selector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.secondVolumePlane0Selector.selectNodeUponCreation = True
    self.secondVolumePlane0Selector.addEnabled = True
    self.secondVolumePlane0Selector.removeEnabled = True
    self.secondVolumePlane0Selector.noneEnabled = True
    self.secondVolumePlane0Selector.showHidden = False
    self.secondVolumePlane0Selector.showChildNodeTypes = False
    self.secondVolumePlane0Selector.setMRMLScene(slicer.mrmlScene)
    self.secondVolumePlane0Selector.setToolTip('Select the phase/imaginary image')
    trackingFormLayout.addRow('Phase/Imaginary: ', self.secondVolumePlane0Selector)
    
    # Select a segmentation for masking (optional)
    self.segmentationMaskPlane0Selector = slicer.qMRMLNodeComboBox()
    self.segmentationMaskPlane0Selector.nodeTypes = ['vtkMRMLSegmentationNode']
    self.segmentationMaskPlane0Selector.selectNodeUponCreation = True
    self.segmentationMaskPlane0Selector.noneEnabled = True
    self.segmentationMaskPlane0Selector.showChildNodeTypes = False
    self.segmentationMaskPlane0Selector.showHidden = False
    self.segmentationMaskPlane0Selector.setMRMLScene(slicer.mrmlScene)
    self.segmentationMaskPlane0Selector.setToolTip('Select segmentation for masking input images')
    trackingFormLayout.addRow('Mask (optional): ', self.segmentationMaskPlane0Selector)

    sectionPlane1 = SeparatorWidget('PLANE_1 (SAG)')
    trackingFormLayout.addRow(sectionPlane1)

    # Input magnitude/real volume (first volume)
    self.firstVolumePlane1Selector = slicer.qMRMLNodeComboBox()
    self.firstVolumePlane1Selector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.firstVolumePlane1Selector.selectNodeUponCreation = True
    self.firstVolumePlane1Selector.addEnabled = True
    self.firstVolumePlane1Selector.removeEnabled = True
    self.firstVolumePlane1Selector.noneEnabled = True
    self.firstVolumePlane1Selector.showHidden = False
    self.firstVolumePlane1Selector.showChildNodeTypes = False
    self.firstVolumePlane1Selector.setMRMLScene(slicer.mrmlScene)
    self.firstVolumePlane1Selector.setToolTip('Select the magnitude/real image')
    trackingFormLayout.addRow('Magnitude/Real: ', self.firstVolumePlane1Selector)

    # Input phase/imaginary volume (second volume)
    self.secondVolumePlane1Selector = slicer.qMRMLNodeComboBox()
    self.secondVolumePlane1Selector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.secondVolumePlane1Selector.selectNodeUponCreation = True
    self.secondVolumePlane1Selector.addEnabled = True
    self.secondVolumePlane1Selector.removeEnabled = True
    self.secondVolumePlane1Selector.noneEnabled = True
    self.secondVolumePlane1Selector.showHidden = False
    self.secondVolumePlane1Selector.showChildNodeTypes = False
    self.secondVolumePlane1Selector.setMRMLScene(slicer.mrmlScene)
    self.secondVolumePlane1Selector.setToolTip('Select the phase/imaginary image')
    trackingFormLayout.addRow('Phase/Imaginary: ', self.secondVolumePlane1Selector)
    
    # Select a segmentation for masking (optional)
    self.segmentationMaskPlane1Selector = slicer.qMRMLNodeComboBox()
    self.segmentationMaskPlane1Selector.nodeTypes = ['vtkMRMLSegmentationNode']
    self.segmentationMaskPlane1Selector.selectNodeUponCreation = True
    self.segmentationMaskPlane1Selector.noneEnabled = True
    self.segmentationMaskPlane1Selector.showChildNodeTypes = False
    self.segmentationMaskPlane1Selector.showHidden = False
    self.segmentationMaskPlane1Selector.setMRMLScene(slicer.mrmlScene)
    self.segmentationMaskPlane1Selector.setToolTip('Select segmentation for masking input images')
    trackingFormLayout.addRow('Mask (optional): ', self.segmentationMaskPlane1Selector)

    sectionPlane2 = SeparatorWidget('PLANE_2 (AX)')
    trackingFormLayout.addRow(sectionPlane2)

    # Input magnitude/real volume (first volume)
    self.firstVolumePlane2Selector = slicer.qMRMLNodeComboBox()
    self.firstVolumePlane2Selector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.firstVolumePlane2Selector.selectNodeUponCreation = True
    self.firstVolumePlane2Selector.addEnabled = True
    self.firstVolumePlane2Selector.removeEnabled = True
    self.firstVolumePlane2Selector.noneEnabled = True
    self.firstVolumePlane2Selector.showHidden = False
    self.firstVolumePlane2Selector.showChildNodeTypes = False
    self.firstVolumePlane2Selector.setMRMLScene(slicer.mrmlScene)
    self.firstVolumePlane2Selector.setToolTip('Select the magnitude/real image')
    trackingFormLayout.addRow('Magnitude/Real: ', self.firstVolumePlane2Selector)

    # Input phase/imaginary volume (second volume)
    self.secondVolumePlane2Selector = slicer.qMRMLNodeComboBox()
    self.secondVolumePlane2Selector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.secondVolumePlane2Selector.selectNodeUponCreation = True
    self.secondVolumePlane2Selector.addEnabled = True
    self.secondVolumePlane2Selector.removeEnabled = True
    self.secondVolumePlane2Selector.noneEnabled = True
    self.secondVolumePlane2Selector.showHidden = False
    self.secondVolumePlane2Selector.showChildNodeTypes = False
    self.secondVolumePlane2Selector.setMRMLScene(slicer.mrmlScene)
    self.secondVolumePlane2Selector.setToolTip('Select the phase/imaginary image')
    trackingFormLayout.addRow('Phase/Imaginary: ', self.secondVolumePlane2Selector)
    
    # Select a segmentation for masking (optional)
    self.segmentationMaskPlane2Selector = slicer.qMRMLNodeComboBox()
    self.segmentationMaskPlane2Selector.nodeTypes = ['vtkMRMLSegmentationNode']
    self.segmentationMaskPlane2Selector.selectNodeUponCreation = True
    self.segmentationMaskPlane2Selector.noneEnabled = True
    self.segmentationMaskPlane2Selector.showChildNodeTypes = False
    self.segmentationMaskPlane2Selector.showHidden = False
    self.segmentationMaskPlane2Selector.setMRMLScene(slicer.mrmlScene)
    self.segmentationMaskPlane2Selector.setToolTip('Select segmentation for masking input images')
    trackingFormLayout.addRow('Mask (optional): ', self.segmentationMaskPlane2Selector)
    
    trackingHBoxLayout = qt.QHBoxLayout()  
    sectionTrack = SeparatorWidget('', useLine=False)
    trackingFormLayout.addRow(sectionTrack)

    # Confidence threshold
    self.confidenceComboBox = qt.QComboBox()
    trackingHBoxLayout.addWidget(self.confidenceComboBox)
    self.confidenceLevelLabels = [('Low', 1), ('Medium Low', 2), ('Medium', 3), ('Medium High', 4), ('High', 5)]
    for level, value in self.confidenceLevelLabels:
      self.confidenceComboBox.addItem(level) 
    defaultConfidence = 'Medium'  
    index = self.confidenceComboBox.findText(defaultConfidence)
    if index != -1:
        self.confidenceComboBox.setCurrentIndex(index)
    trackingHBoxLayout.addItem(hSpacer)
    
    # Start/Stop tracking 
    self.startTrackingButton = qt.QPushButton('Start Tracking')
    self.startTrackingButton.toolTip = 'Start needle tracking in image sequence'
    self.startTrackingButton.enabled = False
    trackingHBoxLayout.addWidget(self.startTrackingButton)
    self.stopTrackingButton = qt.QPushButton('Stop Tracking')
    self.stopTrackingButton.toolTip = 'Stop the needle tracking'
    self.stopTrackingButton.enabled = False    
    trackingHBoxLayout.addWidget(self.stopTrackingButton)
    trackingFormLayout.addRow('Confidence level:', trackingHBoxLayout)

    ## Optional                
    ####################################
    
    optionalCollapsibleButton = ctk.ctkCollapsibleButton()
    optionalCollapsibleButton.collapsed = True
    optionalCollapsibleButton.text = 'Optional'
    self.layout.addWidget(optionalCollapsibleButton)
    optionalFormLayout = qt.QFormLayout(optionalCollapsibleButton)

    
    sectionTargetTip = SeparatorWidget('Target and Needle Tip')
    optionalFormLayout.addRow(sectionTargetTip)

    igtlHBoxLayout = qt.QHBoxLayout()   

    # Push target coordinates to robot
    '''
    self.pushTargetToRobotCheckBox = qt.QCheckBox()
    self.pushTargetToRobotCheckBox.checked = False
    self.pushTargetToRobotCheckBox.setToolTip('If checked, pushes target position to robot in zFrame coordinates')
    igtlHBoxLayout.addWidget(qt.QLabel('Push Target to Robot'))
    igtlHBoxLayout.addWidget(self.pushTargetToRobotCheckBox)
    igtlHBoxLayout.addItem(hSpacer)
    '''

    # Push tip coordinates to robot
    self.pushTipToRobotCheckBox = qt.QCheckBox()
    self.pushTipToRobotCheckBox.checked = False
    self.pushTipToRobotCheckBox.setToolTip('If checked, pushes current tip position to robot in scanner coordinates')
    igtlHBoxLayout.addWidget(qt.QLabel('Push Tip to Robot'))
    igtlHBoxLayout.addWidget(self.pushTipToRobotCheckBox)
    igtlHBoxLayout.addStretch()
    optionalFormLayout.addRow(igtlHBoxLayout)
    
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
    '''
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
    '''
    # Select target and send with OpenIGTLink server
    '''
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
    '''
    ## Advanced parameters            
    ####################################

    advancedCollapsibleButton = ctk.ctkCollapsibleButton()
    advancedCollapsibleButton.text = 'Advanced'
    advancedCollapsibleButton.collapsed=1
    self.layout.addWidget(advancedCollapsibleButton)
    advancedFormLayout = qt.QFormLayout(advancedCollapsibleButton)

    # Log mode check box (print log messages)
    logHBoxLayout = qt.QHBoxLayout()  
    logLabel = qt.QLabel('Screen log')
    logLabel.setFixedWidth(85)
    self.logFlagCheckBox = qt.QCheckBox()
    self.logFlagCheckBox.checked = False
    self.logFlagCheckBox.setToolTip('If checked, prints log messages at intermediate steps')
    logHBoxLayout.addWidget(logLabel)
    logHBoxLayout.addWidget(self.logFlagCheckBox)
    logHBoxLayout.addStretch()
    advancedFormLayout.addRow(logHBoxLayout)
        
    # Debug mode check box (output images at intermediate steps)
    saveHBoxLayout = qt.QHBoxLayout()   
    debugLabel = qt.QLabel('Save images')
    debugLabel.setFixedWidth(85)
    self.debugFlagCheckBox = qt.QCheckBox()
    self.debugFlagCheckBox.checked = False
    self.debugFlagCheckBox.setToolTip('If checked, output images at intermediate steps')
    self.debugNameTextbox = qt.QLineEdit()
    self.debugNameTextbox.setFixedWidth(250)
    self.debugNameTextbox.setPlaceholderText('Include optional subfolder name')
    self.debugNameTextbox.setReadOnly(False)
    self.debugNameTextbox.enabled = False
    saveHBoxLayout.addWidget(debugLabel)
    saveHBoxLayout.addWidget(self.debugFlagCheckBox)
    saveHBoxLayout.addItem(hSpacer)    
    saveHBoxLayout.addWidget(self.debugNameTextbox)  
    saveHBoxLayout.addStretch()  
    advancedFormLayout.addRow(saveHBoxLayout)    

    # Preprocessing options for the images
    preprocessingHBoxLayout = qt.QHBoxLayout()   
    phaseUnwrapLabel = qt.QLabel('Phase unwrap')
    phaseUnwrapLabel.setFixedWidth(85)
    self.phaseUnwrapCheckBox = qt.QCheckBox()
    self.phaseUnwrapCheckBox.checked = False
    self.phaseUnwrapCheckBox.setToolTip('If checked, phase images will be unwraped before needle segmentation')
    preprocessingHBoxLayout.addWidget(phaseUnwrapLabel)
    preprocessingHBoxLayout.addWidget(self.phaseUnwrapCheckBox)
    preprocessingHBoxLayout.addStretch()  
    advancedFormLayout.addRow(preprocessingHBoxLayout)
    
    # Window size for sliding window
    self.windowSizeWidget = ctk.ctkSliderWidget()
    self.windowSizeWidget.singleStep = 4
    self.windowSizeWidget.minimum = 32
    self.windowSizeWidget.maximum = 84
    self.windowSizeWidget.value = 84
    self.windowSizeWidget.setToolTip("Set window size (px) for the sliding window")
    advancedFormLayout.addRow("Sliding window size ", self.windowSizeWidget)

    # Min tip size for segmentation acceptance
    self.minTipSizeWidget = ctk.ctkSliderWidget()
    self.minTipSizeWidget.singleStep = 5
    self.minTipSizeWidget.minimum = 5
    self.minTipSizeWidget.maximum = 35
    self.minTipSizeWidget.value = 10
    self.minTipSizeWidget.setToolTip("Set minimum tip size (px) for accepting segmentation")
    advancedFormLayout.addRow("Minimum tip size ", self.minTipSizeWidget)

    # Min shaft size for segmentation acceptance
    self.minShaftSizeWidget = ctk.ctkSliderWidget()
    self.minShaftSizeWidget.singleStep = 5
    self.minShaftSizeWidget.minimum = 5
    self.minShaftSizeWidget.maximum = 50
    self.minShaftSizeWidget.value = 20
    self.minShaftSizeWidget.setToolTip("Set minimum shaft size (px) for accepting segmentation")
    advancedFormLayout.addRow("Minimum shaft size ", self.minShaftSizeWidget)

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
    self.inputVolume2D.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.inputVolume3D.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.inputChannels1.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.inputChannels2.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.inputChannels3.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.modelFileSelector.connect('currentIndexChanged(int)', self.updateParameterNodeFromGUI)

    self.scannerModeMagPhase.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.scannerModeRealImag.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.usePlane0CheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.usePlane1CheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.usePlane2CheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)

    self.updateScanPlaneCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.centerAtTipCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.confidenceComboBox.connect('currentIndexChanged(int)', self.updateParameterNodeFromGUI)
    self.bridgeConnectionSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)

    self.firstVolumePlane0Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.secondVolumePlane0Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.segmentationMaskPlane0Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.firstVolumePlane1Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.secondVolumePlane1Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.segmentationMaskPlane1Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.firstVolumePlane2Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.secondVolumePlane2Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.segmentationMaskPlane2Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    
    self.pushTipToRobotCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    #self.pushTargetToRobotCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    #self.transformSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    #self.targetSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    self.robotConnectionSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateParameterNodeFromGUI)
    
    self.logFlagCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.debugFlagCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.debugNameTextbox.connect("textChanged", self.updateParameterNodeFromGUI)
    self.phaseUnwrapCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.windowSizeWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.minTipSizeWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.minShaftSizeWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)


    # Connect UI buttons to event calls
    self.inputModeMagPhase.connect("toggled(bool)", self.updateModelList)
    self.inputModeRealImag.connect("toggled(bool)", self.updateModelList)
    self.inputVolume2D.connect("toggled(bool)", self.updateModelList)
    self.inputVolume3D.connect("toggled(bool)", self.updateModelList)
    self.inputChannels1.connect("toggled(bool)", self.updateModelList)    
    self.inputChannels2.connect("toggled(bool)", self.updateModelList)
    self.inputChannels3.connect("toggled(bool)", self.updateModelList)

    self.usePlane0CheckBox.connect("toggled(bool)", self.updateButtons)
    self.usePlane1CheckBox.connect("toggled(bool)", self.updateButtons)
    self.usePlane2CheckBox.connect("toggled(bool)", self.updateButtons)
    
    self.setPlane0Button_ras.connect("toggled(bool)", self.updateButtons)
    self.setPlane0Button_view.connect("toggled(bool)", self.updateButtons)
    self.setPlane1Button_ras.connect("toggled(bool)", self.updateButtons)
    self.setPlane1Button_view.connect("toggled(bool)", self.updateButtons)
    self.setPlane2Button_ras.connect("toggled(bool)", self.updateButtons)
    self.setPlane2Button_view.connect("toggled(bool)", self.updateButtons)

    self.rPlane0Textbox.textChanged.connect(lambda text, tb=self.rPlane0Textbox: self.validateFloat(text, tb))
    self.aPlane0Textbox.textChanged.connect(lambda text, tb=self.aPlane0Textbox: self.validateFloat(text, tb))
    self.sPlane0Textbox.textChanged.connect(lambda text, tb=self.sPlane0Textbox: self.validateFloat(text, tb))
    self.rPlane1Textbox.textChanged.connect(lambda text, tb=self.rPlane1Textbox: self.validateFloat(text, tb))
    self.aPlane1Textbox.textChanged.connect(lambda text, tb=self.aPlane1Textbox: self.validateFloat(text, tb))
    self.sPlane1Textbox.textChanged.connect(lambda text, tb=self.sPlane1Textbox: self.validateFloat(text, tb))
    self.rPlane2Textbox.textChanged.connect(lambda text, tb=self.rPlane2Textbox: self.validateFloat(text, tb))
    self.aPlane2Textbox.textChanged.connect(lambda text, tb=self.aPlane2Textbox: self.validateFloat(text, tb))
    self.sPlane2Textbox.textChanged.connect(lambda text, tb=self.sPlane2Textbox: self.validateFloat(text, tb))

    self.sendPlane0Button.connect('clicked(bool)', self.sendPlane0)
    self.sendPlane1Button.connect('clicked(bool)', self.sendPlane1)
    self.sendPlane2Button.connect('clicked(bool)', self.sendPlane2)
    self.updateScanPlaneCheckBox.connect("toggled(bool)", self.updateButtons)
    self.centerAtTipCheckBox.connect("toggled(bool)", self.updateButtons)
    self.bridgeConnectionSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)

    self.firstVolumePlane0Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.secondVolumePlane0Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.firstVolumePlane1Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.secondVolumePlane1Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.firstVolumePlane2Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.secondVolumePlane2Selector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.startTrackingButton.connect('clicked(bool)', self.startTracking)
    self.stopTrackingButton.connect('clicked(bool)', self.stopTracking)    

    self.pushTipToRobotCheckBox.connect("toggled(bool)", self.updateButtons)
    #self.pushTargetToRobotCheckBox.connect("toggled(bool)", self.updateButtons)
    #self.sendTargetButton.connect('clicked(bool)', self.sendTarget)
    #self.transformSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    #self.targetSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    self.robotConnectionSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.updateButtons)
    
    # Internal variables
    self.isTrackingOn = False
    self.inputMode = None
    self.inputVolume = None
    self.inputChannels = None

    self.scannerMode = None
    self.useScanPlanes = None

    self.updateScanPlane = None
    self.centerScanAtTip = None
    self.mrigtlBridgeServerNode = None

    self.firstVolumePlane0 = None
    self.secondVolumePlane0 = None
    self.firstVolumePlane1 = None
    self.secondVolumePlane1 = None
    self.firstVolumePlane2 = None
    self.secondVolumePlane2 = None

    self.pushTipToRobot = None
    #self.pushTargetToRobot = None
    self.robotIGTLClientNode = None
    #self.zFrameTransform = None

    self.debugFlag = None
    self.debugName = None
    self.windowSize = None
    self.minTipSize = None
    self.minShaftSize = None

    # Timestamp
    self.tracker = TimestampTracker(['image_received', 'image_prepared', 'needle_segmented', 'tip_tracked', 'plan_updated'])

    # Initialize module logic
    self.logic = AINeedleTrackingLogic()
    self.logic.tracker = self.tracker
    self.logic.setConfidenceLevelLabels(self.confidenceLevelLabels)
  
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
    self.inputModeMagPhase.checked = (self._parameterNode.GetParameter('InputMode') == 'MagPhase')
    self.inputModeRealImag.checked = (self._parameterNode.GetParameter('InputMode') == 'RealImag')
    self.inputVolume2D.checked = (self._parameterNode.GetParameter('InputVolume') == '2D')
    self.inputVolume3D.checked = (self._parameterNode.GetParameter('InputVolume') == '3D')
    self.inputChannels1.checked = (self._parameterNode.GetParameter('InputChannels') == '1CH')
    self.inputChannels2.checked = (self._parameterNode.GetParameter('InputChannels') == '2CH')
    self.inputChannels3.checked = (self._parameterNode.GetParameter('InputChannels') == '3CH')
    self.modelFileSelector.setCurrentIndex(int(self._parameterNode.GetParameter('Model')))

    self.scannerModeMagPhase.checked = (self._parameterNode.GetParameter('ScannerMode') == 'MagPhase')
    self.scannerModeRealImag.checked = (self._parameterNode.GetParameter('ScannerMode') == 'RealImag')
    self.usePlane0CheckBox.checked = (self._parameterNode.GetParameter('UseScanPlane0') == 'True')
    self.usePlane1CheckBox.checked = (self._parameterNode.GetParameter('UseScanPlane1') == 'True')
    self.usePlane2CheckBox.checked = (self._parameterNode.GetParameter('UseScanPlane2') == 'True')

    self.setPlane0Button_ras.checked = (self._parameterNode.GetParameter('SetPlane0RAS') == 'True')
    self.setPlane1Button_ras.checked = (self._parameterNode.GetParameter('SetPlane1RAS') == 'True')
    self.setPlane2Button_ras.checked = (self._parameterNode.GetParameter('SetPlane2RAS') == 'True')

    self.updateScanPlaneCheckBox.checked = (self._parameterNode.GetParameter('UpdateScanPlane') == 'True')
    self.centerAtTipCheckBox.checked = (self._parameterNode.GetParameter('CenterScanAtTip') == 'True')
    self.confidenceComboBox.setCurrentIndex(int(self._parameterNode.GetParameter('ConfidenceLevel')))

    self.bridgeConnectionSelector.setCurrentNode(self._parameterNode.GetNodeReference('mrigtlBridgeServer'))

    self.firstVolumePlane0Selector.setCurrentNode(self._parameterNode.GetNodeReference('FirstVolumePlane0'))
    self.secondVolumePlane0Selector.setCurrentNode(self._parameterNode.GetNodeReference('SecondVolumePlane0'))
    self.segmentationMaskPlane0Selector.setCurrentNode(self._parameterNode.GetNodeReference('MaskPlane0'))
    self.firstVolumePlane1Selector.setCurrentNode(self._parameterNode.GetNodeReference('FirstVolumePlane1'))
    self.secondVolumePlane1Selector.setCurrentNode(self._parameterNode.GetNodeReference('SecondVolumePlane1'))
    self.segmentationMaskPlane1Selector.setCurrentNode(self._parameterNode.GetNodeReference('MaskPlane1'))
    self.firstVolumePlane2Selector.setCurrentNode(self._parameterNode.GetNodeReference('FirstVolumePlane2'))
    self.secondVolumePlane2Selector.setCurrentNode(self._parameterNode.GetNodeReference('SecondVolumePlane2'))
    self.segmentationMaskPlane2Selector.setCurrentNode(self._parameterNode.GetNodeReference('MaskPlane2'))

    self.pushTipToRobotCheckBox.checked = (self._parameterNode.GetParameter('PushTipToRobot') == 'True')
    #self.pushTargetToRobotCheckBox.checked = (self._parameterNode.GetParameter('PushTargetToRobot') == 'True')
    #self.transformSelector.setCurrentNode(self._parameterNode.GetNodeReference('zFrame'))
    #self.targetSelector.setCurrentNode(self._parameterNode.GetNodeReference('Target'))
    self.robotConnectionSelector.setCurrentNode(self._parameterNode.GetNodeReference('RobotIGTLClient'))
    
    self.logFlagCheckBox.checked = (self._parameterNode.GetParameter('ScreenLog') == 'True')
    self.debugFlagCheckBox.checked = (self._parameterNode.GetParameter('Debug') == 'True')
    self.debugNameTextbox.setText(self._parameterNode.GetParameter('DebugName'))
    self.phaseUnwrapCheckBox.checked = (self._parameterNode.GetParameter('PhaseUnwrap') == 'True')
    self.windowSizeWidget.value = float(self._parameterNode.GetParameter('WindowSize'))
    self.minTipSizeWidget.value = float(self._parameterNode.GetParameter('MinTipSize'))
    self.minShaftSizeWidget.value = float(self._parameterNode.GetParameter('MinShaftSize'))

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
    self._parameterNode.SetParameter('InputMode', 'MagPhase' if self.inputModeMagPhase.checked else 'RealImag')
    self._parameterNode.SetParameter('InputVolume', '2D' if self.inputVolume2D.checked else '3D')
    self._parameterNode.SetParameter('InputChannels', '1CH' if self.inputChannels1.checked else '2CH' if self.inputChannels2.checked else '3CH')
    self._parameterNode.SetParameter('Model', str(self.modelFileSelector.currentIndex))

    self._parameterNode.SetParameter('ScannerMode', 'MagPhase' if self.scannerModeMagPhase.checked else 'RealImag')
    self._parameterNode.SetParameter('UseScanPlane0', 'True' if self.usePlane0CheckBox.checked else 'False')
    self._parameterNode.SetParameter('UseScanPlane1', 'True' if self.usePlane1CheckBox.checked else 'False')
    self._parameterNode.SetParameter('UseScanPlane2', 'True' if self.usePlane2CheckBox.checked else 'False')

    self._parameterNode.SetParameter('SetPlane0RAS', 'True' if self.setPlane0Button_ras.checked else 'False')
    self._parameterNode.SetParameter('SetPlane1RAS', 'True' if self.setPlane1Button_ras.checked else 'False')
    self._parameterNode.SetParameter('SetPlane2RAS', 'True' if self.setPlane2Button_ras.checked else 'False')

    self._parameterNode.SetParameter('UpdateScanPlane', 'True' if self.updateScanPlaneCheckBox.checked else 'False')
    self._parameterNode.SetParameter('CenterScanAtTip', 'True' if self.centerAtTipCheckBox.checked else 'False')
    self._parameterNode.SetParameter('ConfidenceLevel', str(self.confidenceComboBox.currentIndex))
    self._parameterNode.SetNodeReferenceID('mrigtlBridgeServer', self.bridgeConnectionSelector.currentNodeID)

    self._parameterNode.SetNodeReferenceID('FirstVolumePlane0', self.firstVolumePlane0Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('SecondVolumePlane0', self.secondVolumePlane0Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('MaskPlane0', self.segmentationMaskPlane0Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('FirstVolumePlane1', self.firstVolumePlane1Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('SecondVolumePlane1', self.secondVolumePlane1Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('MaskPlane1', self.segmentationMaskPlane1Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('FirstVolumePlane2', self.firstVolumePlane2Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('SecondVolumePlane2', self.secondVolumePlane2Selector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('MaskPlane2', self.segmentationMaskPlane2Selector.currentNodeID)

    self._parameterNode.SetParameter('PushTipToRobot', 'True' if self.pushTipToRobotCheckBox.checked else 'False')
    #self._parameterNode.SetParameter('PushTargetToRobot', 'True' if self.pushTargetToRobotCheckBox.checked else 'False')
    #self._parameterNode.SetNodeReferenceID('zFrame', self.transformSelector.currentNodeID)
    #self._parameterNode.SetNodeReferenceID('Target', self.targetSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID('RobotIGTLClient', self.robotConnectionSelector.currentNodeID)

    self._parameterNode.SetParameter('ScreenLog', 'True' if self.logFlagCheckBox.checked else 'False')
    self._parameterNode.SetParameter('Debug', 'True' if self.debugFlagCheckBox.checked else 'False')
    self._parameterNode.SetParameter('DebugName', self.debugNameTextbox.text.strip())    
    self._parameterNode.SetParameter('PhaseUnwrap', 'True' if self.phaseUnwrapCheckBox.checked else 'False')
    self._parameterNode.SetParameter('WindowSize', str(self.windowSizeWidget.value))
    self._parameterNode.SetParameter('MinTipSize', str(self.minTipSizeWidget.value))
    self._parameterNode.SetParameter('MinShaftSize', str(self.minShaftSizeWidget.value))
    self._parameterNode.EndModify(wasModified)

  # Validation of float input
  def validateFloat(self, text, textbox):
    validator = textbox.validator()
    if validator is None:
      print(f"No validator set for {textbox}. Skipping validation.")
      return
    # Skip validation if the text box is empty
    if not text.strip():  # `strip()` removes whitespace, ensuring only non-whitespace counts
        textbox.setStyleSheet("")  # Reset to default style
        self.updateButtons()  # Update buttons
        return
    # Perform validation and get the state
    state = validator.validate(text, 0)
    if state == qt.QValidator.Acceptable:
      textbox.setStyleSheet("")   # Reset to default style
    elif state == qt.QValidator.Intermediate:
      textbox.setStyleSheet("border: 1px solid orange;")  # Highlight intermediate input
    else:  # Invalid input
      textbox.setStyleSheet("border: 1px solid red;")  # Highlight invalid input
    self.updateButtons()       # Update buttons

  # Update button states
  def updateButtons(self):
    # Initialize requirements as all optional (True)
    serverDefined = True      # Not required
    clientDefined = True      # Not required
    #transformDefined = True   # Not required
    #targetDefined = True      # Not required 
    scanPlaneDefined = True if (self.usePlane0CheckBox.checked or self.usePlane1CheckBox.checked or self.usePlane2CheckBox.checked) else False
    modelDefined = False if (self.modelFileSelector.currentText == '') else True

    # Not tracking = ENABLE SELECTION
    if not self.isTrackingOn:
      # Logic for required variables selection: 
      self.inputModeMagPhase.enabled = True
      self.inputModeRealImag.enabled = True
      self.inputVolume2D.enabled = True
      self.inputVolume3D.enabled = True
      self.inputChannels1.enabled = True
      self.inputChannels2.enabled = True
      self.inputChannels3.enabled = True
      self.modelFileSelector.enabled = True
      self.scannerModeMagPhase.enabled = True
      self.scannerModeRealImag.enabled = True
      self.usePlane0CheckBox.enabled = True
      self.usePlane1CheckBox.enabled = True
      self.usePlane2CheckBox.enabled = True
      self.pushTipToRobotCheckBox.enabled = True
      #self.pushTargetToRobotCheckBox.enabled = True
      self.confidenceComboBox.enabled = True
      self.windowSizeWidget.setEnabled(True)
      self.minTipSizeWidget.setEnabled(True)
      self.minShaftSizeWidget.setEnabled(True)

      # Logic for optional variables selection
      # 1) Connection to Scanner
      if scanPlaneDefined:
        self.bridgeConnectionSelector.enabled = True
        serverDefined = True if (self.bridgeConnectionSelector.currentNode()) else False
      else:
        self.bridgeConnectionSelector.enabled = False
        serverDefined = True
      # 1.b) Update Scan Plane with Tip
      if scanPlaneDefined and serverDefined:
        self.updateScanPlaneCheckBox.enabled = True
      else:
        self.updateScanPlaneCheckBox.enabled = False
      #1.c) Center Scan At the Tip
      if  self.updateScanPlaneCheckBox.checked:
        self.centerAtTipCheckBox.enabled = True
      else:
        self.centerAtTipCheckBox.enabled = False
      # PLAN_0
      if self.usePlane0CheckBox.checked:
        self.firstVolumePlane0Selector.enabled = True
        self.secondVolumePlane0Selector.enabled = True
        self.segmentationMaskPlane0Selector.enabled = True
        if serverDefined:
          self.setPlane0Button_ras.enabled = True
          self.setPlane0Button_view.enabled = True
          if self.setPlane0Button_view.checked:
            self.scenePlane0Button_red.enabled = True
            self.scenePlane0Button_yellow.enabled = True
            self.scenePlane0Button_green.enabled = True    
            self.rPlane0Textbox.enabled = False
            self.aPlane0Textbox.enabled = False
            self.sPlane0Textbox.enabled = False
            self.sendPlane0Button.enabled = True
          else:
            self.scenePlane0Button_red.enabled = False
            self.scenePlane0Button_yellow.enabled = False
            self.scenePlane0Button_green.enabled = False    
            self.rPlane0Textbox.enabled = True
            self.aPlane0Textbox.enabled = True
            self.sPlane0Textbox.enabled = True
            rasDefinedPlane0 = (self.rPlane0Textbox.text.strip() and self.aPlane0Textbox.text.strip() and self.sPlane0Textbox.text.strip())
            if rasDefinedPlane0:
              self.sendPlane0Button.enabled = True
            else:
              self.sendPlane0Button.enabled = False
        else:
          self.setPlane0Button_ras.enabled = False
          self.setPlane0Button_view.enabled = False
          self.scenePlane0Button_red.enabled = False
          self.scenePlane0Button_yellow.enabled = False
          self.scenePlane0Button_green.enabled = False    
          self.rPlane0Textbox.enabled = False
          self.aPlane0Textbox.enabled = False
          self.sPlane0Textbox.enabled = False
          self.sendPlane0Button.enabled = False
      else:
        self.setPlane0Button_ras.enabled = False
        self.setPlane0Button_view.enabled = False
        self.rPlane0Textbox.enabled = False
        self.aPlane0Textbox.enabled = False
        self.sPlane0Textbox.enabled = False        
        self.scenePlane0Button_red.enabled = False
        self.scenePlane0Button_yellow.enabled = False
        self.scenePlane0Button_green.enabled = False
        self.firstVolumePlane0Selector.enabled = False
        self.secondVolumePlane0Selector.enabled = False
        self.segmentationMaskPlane0Selector.enabled = False
        self.sendPlane0Button.enabled = False

      # PLAN_1
      if self.usePlane1CheckBox.checked:
        self.firstVolumePlane1Selector.enabled = True
        self.secondVolumePlane1Selector.enabled = True
        self.segmentationMaskPlane1Selector.enabled = True
        if serverDefined:
          self.setPlane1Button_ras.enabled = True
          self.setPlane1Button_view.enabled = True
          if self.setPlane1Button_view.checked:
            self.scenePlane1Button_red.enabled = True
            self.scenePlane1Button_yellow.enabled = True
            self.scenePlane1Button_green.enabled = True    
            self.rPlane1Textbox.enabled = False
            self.aPlane1Textbox.enabled = False
            self.sPlane1Textbox.enabled = False
            self.sendPlane1Button.enabled = True
          else:
            self.scenePlane1Button_red.enabled = False
            self.scenePlane1Button_yellow.enabled = False
            self.scenePlane1Button_green.enabled = False    
            self.rPlane1Textbox.enabled = True
            self.aPlane1Textbox.enabled = True
            self.sPlane1Textbox.enabled = True
            rasDefinedPlane1 = (self.rPlane1Textbox.text.strip() and self.aPlane1Textbox.text.strip() and self.sPlane1Textbox.text.strip())
            if rasDefinedPlane1:
              self.sendPlane1Button.enabled = True
            else:
              self.sendPlane1Button.enabled = False
        else:
          self.setPlane1Button_ras.enabled = False
          self.setPlane1Button_view.enabled = False
          self.scenePlane1Button_red.enabled = False
          self.scenePlane1Button_yellow.enabled = False
          self.scenePlane1Button_green.enabled = False    
          self.rPlane1Textbox.enabled = False
          self.aPlane1Textbox.enabled = False
          self.sPlane1Textbox.enabled = False
          self.sendPlane1Button.enabled = False
      else:
        self.setPlane1Button_ras.enabled = False
        self.setPlane1Button_view.enabled = False
        self.rPlane1Textbox.enabled = False
        self.aPlane1Textbox.enabled = False
        self.sPlane1Textbox.enabled = False        
        self.scenePlane1Button_red.enabled = False
        self.scenePlane1Button_yellow.enabled = False
        self.scenePlane1Button_green.enabled = False
        self.firstVolumePlane1Selector.enabled = False
        self.secondVolumePlane1Selector.enabled = False
        self.segmentationMaskPlane1Selector.enabled = False
        self.sendPlane1Button.enabled = False

      # PLAN_2
      if self.usePlane2CheckBox.checked:
        self.firstVolumePlane2Selector.enabled = True
        self.secondVolumePlane2Selector.enabled = True
        self.segmentationMaskPlane2Selector.enabled = True
        if serverDefined:
          self.setPlane2Button_ras.enabled = True
          self.setPlane2Button_view.enabled = True
          if self.setPlane2Button_view.checked:
            self.scenePlane2Button_red.enabled = True
            self.scenePlane2Button_yellow.enabled = True
            self.scenePlane2Button_green.enabled = True    
            self.rPlane2Textbox.enabled = False
            self.aPlane2Textbox.enabled = False
            self.sPlane2Textbox.enabled = False
            self.sendPlane2Button.enabled = True
          else:
            self.scenePlane2Button_red.enabled = False
            self.scenePlane2Button_yellow.enabled = False
            self.scenePlane2Button_green.enabled = False    
            self.rPlane2Textbox.enabled = True
            self.aPlane2Textbox.enabled = True
            self.sPlane2Textbox.enabled = True
            rasDefinedPlane2 = (self.rPlane2Textbox.text.strip() and self.aPlane2Textbox.text.strip() and self.sPlane2Textbox.text.strip())
            if rasDefinedPlane2:
              self.sendPlane2Button.enabled = True
            else:
              self.sendPlane2Button.enabled = False
        else:
          self.setPlane2Button_ras.enabled = False
          self.setPlane2Button_view.enabled = False
          self.scenePlane2Button_red.enabled = False
          self.scenePlane2Button_yellow.enabled = False
          self.scenePlane2Button_green.enabled = False    
          self.rPlane2Textbox.enabled = False
          self.aPlane2Textbox.enabled = False
          self.sPlane2Textbox.enabled = False
          self.sendPlane2Button.enabled = False
      else:
        self.setPlane2Button_ras.enabled = False
        self.setPlane2Button_view.enabled = False
        self.rPlane2Textbox.enabled = False
        self.aPlane2Textbox.enabled = False
        self.sPlane2Textbox.enabled = False        
        self.scenePlane2Button_red.enabled = False
        self.scenePlane2Button_yellow.enabled = False
        self.scenePlane2Button_green.enabled = False
        self.firstVolumePlane2Selector.enabled = False
        self.secondVolumePlane2Selector.enabled = False
        self.segmentationMaskPlane2Selector.enabled = False
        self.sendPlane2Button.enabled = False

      # 2) Push target and tip (required for both)
      if self.pushTipToRobotCheckBox.checked: # or self.pushTargetToRobotCheckBox.checked:
        self.robotConnectionSelector.enabled = True
        #self.transformSelector.enabled = True
        clientDefined = self.robotConnectionSelector.currentNode()
        #transformDefined = self.transformSelector.currentNode()
      else:
        self.robotConnectionSelector.enabled = False
        #self.transformSelector.enabled = False
      # 3) Push target only
      '''
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
      '''
      # 3) Debug
      if self.debugFlagCheckBox.checked:
        self.debugNameTextbox.enabled = True
        # debugNameDefined = if self.debugNameTextbox.text.strip() 
      else:
        self.debugNameTextbox.enabled = False
      # 4) Phase unwrap
      if self.inputModeMagPhase.checked:
        self.phaseUnwrapCheckBox.enabled = True
      else:
        self.phaseUnwrapCheckBox.enabled = False
    # When tracking = DISABLE SELECTION
    else:
      #Setup
      self.inputModeMagPhase.enabled = False
      self.inputModeRealImag.enabled = False
      self.inputVolume2D.enabled = False
      self.inputVolume3D.enabled = False
      self.inputChannels1.enabled = False
      self.inputChannels2.enabled = False
      self.inputChannels3.enabled = False
      self.modelFileSelector.enabled = False
      self.scannerModeMagPhase.enabled = False
      self.scannerModeRealImag.enabled = False
      #Optional - MRI Scan Plane
      self.bridgeConnectionSelector.enabled = False
      self.usePlane0CheckBox.enabled = False
      self.usePlane1CheckBox.enabled = False
      self.usePlane2CheckBox.enabled = False
      self.setPlane0Button_ras.enabled = False
      self.setPlane0Button_view.enabled = False
      self.setPlane1Button_ras.enabled = False
      self.setPlane1Button_view.enabled = False
      self.setPlane2Button_ras.enabled = False
      self.setPlane2Button_view.enabled = False
      self.rPlane0Textbox.enabled = False
      self.aPlane0Textbox.enabled = False
      self.sPlane0Textbox.enabled = False
      self.rPlane1Textbox.enabled = False
      self.aPlane1Textbox.enabled = False
      self.sPlane1Textbox.enabled = False      
      self.rPlane2Textbox.enabled = False
      self.aPlane2Textbox.enabled = False
      self.sPlane2Textbox.enabled = False
      self.scenePlane0Button_red.enabled = False
      self.scenePlane0Button_yellow.enabled = False
      self.scenePlane0Button_green.enabled = False
      self.scenePlane1Button_red.enabled = False
      self.scenePlane1Button_yellow.enabled = False
      self.scenePlane1Button_green.enabled = False
      self.scenePlane2Button_red.enabled = False
      self.scenePlane2Button_yellow.enabled = False
      self.scenePlane2Button_green.enabled = False
      self.sendPlane0Button.enabled = False
      self.sendPlane1Button.enabled = False
      self.sendPlane2Button.enabled = False
      self.updateScanPlaneCheckBox.enabled = False
      self.centerAtTipCheckBox.enabled = False
      #Optional - Tip and Target
      self.pushTipToRobotCheckBox.enabled = False
      #self.pushTargetToRobotCheckBox.enabled = False
      self.robotConnectionSelector.enabled = False
      #self.transformSelector.enabled = False
      #self.targetSelector.enabled = False
      #self.sendTargetButton.enabled = False
      #Tracking
      self.confidenceComboBox.enabled = False
      self.firstVolumePlane0Selector.enabled = False
      self.secondVolumePlane0Selector.enabled = False
      self.segmentationMaskPlane0Selector.enabled = False
      self.firstVolumePlane1Selector.enabled = False
      self.secondVolumePlane1Selector.enabled = False
      self.segmentationMaskPlane1Selector.enabled = False
      self.firstVolumePlane2Selector.enabled = False
      self.secondVolumePlane2Selector.enabled = False
      self.segmentationMaskPlane2Selector.enabled = False
      #Debug
      self.windowSizeWidget.setEnabled(False)
      self.minTipSizeWidget.setEnabled(False)
      self.minShaftSizeWidget.setEnabled(False)      
    # Check if Tracking is enabled
    if self.inputChannels1.checked:
      rtNodesDefined = True if (self.firstVolumePlane0Selector.currentNode() is not None) else False
    else:
      rtNodesDefined = True if (self.firstVolumePlane0Selector.currentNode() is not None and self.secondVolumePlane0Selector.currentNode() is not None) else False
    # print('%s, %s, %s, %s, %s, %s' %(modelDefined, rtNodesDefined, serverDefined, clientDefined, transformDefined, targetDefined))
    #self.startTrackingButton.enabled = modelDefined and rtNodesDefined and serverDefined and clientDefined and transformDefined and targetDefined and not self.isTrackingOn
    self.startTrackingButton.enabled = modelDefined and rtNodesDefined and serverDefined and clientDefined and not self.isTrackingOn
    self.stopTrackingButton.enabled = self.isTrackingOn
    
  def updateModelList(self):
    # Clear combo box
    self.modelFileSelector.clear()
    # Set folder
    if self.inputModeRealImag.checked:
      inputMode = 'RealImag'
    else:
      inputMode = 'MagPhase'    
    if self.inputChannels1.checked:
      channels = '1'
    elif self.inputChannels2.checked:
      channels = '2'
    else:
      channels = '3'
    if self.inputVolume2D.checked:
      volume = '2'
    else:
      volume = '3'
    listPath = os.path.join(self.path, 'Models', inputMode, volume+'D-'+channels+'CH')
    # Check if the folder exists
    if os.path.exists(listPath) and os.path.isdir(listPath):
        # Get the list of files
        modelList = [f for f in os.listdir(listPath) if os.path.isfile(os.path.join(listPath, f))]
        modelList.sort()
    else:
        # If the folder doesn't exist or is not a directory, use an empty list
        modelList = []
    modelList.sort()
    self.modelFileSelector.addItems(modelList)
    
  # Get selected scene view for initializing scan plane (PLANE_0)
  def getSelectedView0(self):
    selectedView = None
    if (self.scenePlane0Button_red.checked == True):
      selectedView = ('Red')
    elif (self.scenePlane0Button_yellow.checked ==True):
      selectedView = ('Yellow')
    elif (self.scenePlane0Button_green.checked ==True):
      selectedView = ('Green')
    return selectedView
  # Get selected scene view for initializing scan plane (PLANE_1)
  def getSelectedView1(self):
    selectedView = None
    if (self.scenePlane1Button_red.checked == True):
      selectedView = ('Red')
    elif (self.scenePlane1Button_yellow.checked ==True):
      selectedView = ('Yellow')
    elif (self.scenePlane1Button_green.checked ==True):
      selectedView = ('Green')
    return selectedView
  # Get selected scene view for initializing scan plane (PLANE_2)
  def getSelectedView2(self):
    selectedView = None
    if (self.scenePlane2Button_red.checked == True):
      selectedView = ('Red')
    elif (self.scenePlane2Button_yellow.checked ==True):
      selectedView = ('Yellow')
    elif (self.scenePlane2Button_green.checked ==True):
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
    self.inputMode = 'MagPhase' if self.inputModeMagPhase.checked else 'RealImag'
    self.inputVolume = 2 if self.inputVolume2D.checked else 3
    self.inputChannels = 1 if self.inputChannels1.checked else 2 if self.inputChannels2.checked else 3
    self.model = self.modelFileSelector.currentText

    self.scannerMode = 'MagPhase' if self.scannerModeMagPhase.checked else 'RealImag'
    self.useScanPlanes = [self.usePlane0CheckBox.checked , self.usePlane1CheckBox.checked , self.usePlane2CheckBox.checked ]

    self.updateScanPlane = self.updateScanPlaneCheckBox.checked 
    self.centerScanAtTip = self.centerAtTipCheckBox.checked 
    self.confidenceLevel = (self.confidenceComboBox.currentIndex + 1)
    confidenceText = self.logic.getConfidenceText(self.confidenceLevel)
    print('inputMode = %s, inputVolume = %s, inputChannels = %s, model = %s' %(self.inputMode, self.inputVolume, self.inputChannels, self.model))
    print('Tracking with confidence = %s' %confidenceText)
    print('____________________')

    self.mrigtlBridgeServerNode = self.bridgeConnectionSelector.currentNode()

    self.firstVolumePlane0 = self.firstVolumePlane0Selector.currentNode()
    self.secondVolumePlane0 = self.secondVolumePlane0Selector.currentNode() 
    self.segmentationNodePlane0 = self.segmentationMaskPlane0Selector.currentNode()
    self.firstVolumePlane1 = self.firstVolumePlane1Selector.currentNode()
    self.secondVolumePlane1 = self.secondVolumePlane1Selector.currentNode() 
    self.segmentationNodePlane1 = self.segmentationMaskPlane1Selector.currentNode()
    self.firstVolumePlane2 = self.firstVolumePlane2Selector.currentNode()
    self.secondVolumePlane2 = self.secondVolumePlane2Selector.currentNode() 
    self.segmentationNodePlane2 = self.segmentationMaskPlane2Selector.currentNode()

    self.pushTipToRobot = self.pushTipToRobotCheckBox.checked 
    self.robotIGTLClientNode = self.robotConnectionSelector.currentNode()
    #self.zFrameTransform = self.transformSelector.currentNode()

    self.debugFlag = self.debugFlagCheckBox.checked
    self.debugName = self.debugNameTextbox.text.strip()
    self.windowSize = int(self.windowSizeWidget.value)
    self.minTipSize = int(self.minTipSizeWidget.value)
    self.minShaftSize = int(self.minShaftSizeWidget.value)

    # Check if folder exists
    if self.debugFlag:
      path = os.path.dirname(os.path.abspath(__file__))
      folder_path = os.path.join(path, 'Debug', self.debugName)
      if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
      else:
        print(f"Folder '{folder_path}' already exists.")

    # Define image conversion
    if self.inputMode != self.scannerMode:
      self.imageConvertion = self.inputMode
    else:
      self.imageConvertion = 'None'

    # Initialize tracking logic
    self.logic.initializeTracking(self.useScanPlanes)
    self.logic.initializeModel(self.inputMode, self.inputVolume, self.inputChannels, self.model)
    
    # Create listener to image sequence node (considering phase image comes after magnitude)
    if self.useScanPlanes[0] is True:
      self.tracker.mark('image_received')
      self.getNeedle('COR', self.firstVolumePlane0, self.secondVolumePlane0, self.segmentationNodePlane0) 
      if self.inputChannels==1:
        self.addObserver(self.firstVolumePlane0, self.firstVolumePlane0.ImageDataModifiedEvent, self.receivedImagePlane0)
      else:
        self.addObserver(self.secondVolumePlane0, self.secondVolumePlane0.ImageDataModifiedEvent, self.receivedImagePlane0)
    if self.useScanPlanes[1] is True:
      self.tracker.mark('image_received')
      self.getNeedle('SAG', self.firstVolumePlane1, self.secondVolumePlane1, self.segmentationNodePlane1) 
      if self.inputChannels==1:
        self.addObserver(self.firstVolumePlane1, self.firstVolumePlane1.ImageDataModifiedEvent, self.receivedImagePlane1)
      else:
        self.addObserver(self.secondVolumePlane1, self.secondVolumePlane1.ImageDataModifiedEvent, self.receivedImagePlane1)
    if self.useScanPlanes[2] is True:
      self.tracker.mark('image_received')
      self.getNeedle('AX', self.firstVolumePlane2, self.secondVolumePlane2, self.segmentationNodePlane2) 
      if self.inputChannels==1:
        self.addObserver(self.firstVolumePlane2, self.firstVolumePlane2.ImageDataModifiedEvent, self.receivedImagePlane2)
      else:
        self.addObserver(self.secondVolumePlane2, self.secondVolumePlane2.ImageDataModifiedEvent, self.receivedImagePlane2)

  
  def stopTracking(self):
    self.isTrackingOn = False
    self.updateButtons()
    self.tracker.print_pairwise_durations([
      ('image_received', 'image_prepared'),
      ('image_prepared', 'needle_segmented'),
      ('needle_segmented', 'tip_tracked'),
      ('tip_tracked', 'plan_updated')
    ])
    self.tracker.clear()
    #TODO: Define what should to be refreshed
    print('UI: stopTracking()')
    if self.useScanPlanes[0] is True:
      if self.inputChannels==1:
        self.removeObserver(self.firstVolumePlane0, self.firstVolumePlane0.ImageDataModifiedEvent, self.receivedImagePlane0)
      else:
        self.removeObserver(self.secondVolumePlane0, self.secondVolumePlane0.ImageDataModifiedEvent, self.receivedImagePlane0)
    if self.useScanPlanes[1] is True:
      if self.inputChannels==1:
        self.removeObserver(self.firstVolumePlane1, self.firstVolumePlane1.ImageDataModifiedEvent, self.receivedImagePlane1)
      else:
        self.removeObserver(self.secondVolumePlane1, self.secondVolumePlane1.ImageDataModifiedEvent, self.receivedImagePlane1)
    if self.useScanPlanes[2] is True:
      if self.inputChannels==1:
        self.removeObserver(self.firstVolumePlane2, self.firstVolumePlane2.ImageDataModifiedEvent, self.receivedImagePlane2)
      else:
        self.removeObserver(self.secondVolumePlane2, self.secondVolumePlane2.ImageDataModifiedEvent, self.receivedImagePlane2)
    #print('Finished removing observers')
  
  # Send PLAN_0 though OpenIGTLink MRI Server - Value at selected view slice
  def sendPlane0(self):
    self.mrigtlBridgeServerNode = self.bridgeConnectionSelector.currentNode()
    if self.setPlane0Button_ras.checked:
      r = float(self.rPlane0Textbox.text.strip())
      a = float(self.aPlane0Textbox.text.strip())
      s = float(self.sPlane0Textbox.text.strip())
      if r and a and s:
        viewCoordinates = (r, a, s)
      else:
        print('ERROR in sendPlane0: Invalid RAS coordinates')
        return
    else:
      viewCoordinates = self.getSelectetViewCenterCoordinates(self.getSelectedView0())
    # Set  and push PLANE_0
    print('sendPlane0: %s' %str(viewCoordinates))
    self.logic.initializeScanPlane(coordinates=viewCoordinates, plane='COR') 
    self.logic.pushScanPlaneToIGTLink(self.mrigtlBridgeServerNode, plane='COR')
    
  # Send PLAN_1 though OpenIGTLink MRI Server - Value at selected view slice
  def sendPlane1(self):
    self.mrigtlBridgeServerNode = self.bridgeConnectionSelector.currentNode()
    if self.setPlane1Button_ras.checked:
      r = float(self.rPlane1Textbox.text.strip())
      a = float(self.aPlane1Textbox.text.strip())
      s = float(self.sPlane1Textbox.text.strip())
      if r and a and s:
        viewCoordinates = (r, a, s)
      else:
        print('ERROR in sendPlane1: Invalid RAS coordinates')
        return
    else:
      viewCoordinates = self.getSelectetViewCenterCoordinates(self.getSelectedView1())
    # Set  and push PLANE_1
    print('sendPlane1: %s' %str(viewCoordinates))
    self.logic.initializeScanPlane(coordinates=viewCoordinates, plane='SAG') 
    self.logic.pushScanPlaneToIGTLink(self.mrigtlBridgeServerNode, plane='SAG')

  # Send PLAN_2 though OpenIGTLink MRI Server - Value at selected view slice
  def sendPlane2(self):
    # Get parameters
    self.mrigtlBridgeServerNode = self.bridgeConnectionSelector.currentNode()
    if self.setPlane2Button_ras.checked:
      r = float(self.rPlane2Textbox.text.strip())
      a = float(self.aPlane2Textbox.text.strip())
      s = float(self.sPlane2Textbox.text.strip())
      if r and a and s:
        viewCoordinates = (r, a, s)
      else:
        print('ERROR in sendPlane2: Invalid RAS coordinates')
        return
    else:
      viewCoordinates = self.getSelectetViewCenterCoordinates(self.getSelectedView2())
    # Set  and push PLANE_2
    print('sendPlane2: %s' %str(viewCoordinates))
    self.logic.initializeScanPlane(coordinates=viewCoordinates, plane='AX')
    self.logic.pushScanPlaneToIGTLink(self.mrigtlBridgeServerNode, plane='AX')

  '''
  def sendTarget(self):
    print('UI: sendTarget()')
    # Get parameters
    self.target = self.targetSelector.currentNode()
    self.robotIGTLClientNode = self.robotConnectionSelector.currentNode()
    self.zFrameTransform = self.transformSelector.currentNode()
    # Set zFrame transformation
    self.logic.initializeZFrame(self.zFrameTransform)
    # Push target
    self.logic.pushTargetToIGTLink(self.robotIGTLClientNode, self.target)
  '''

  #TODO: Make a generic version that checks which plane is responsible for the callback
  def receivedImagePlane0(self, caller=None, event=None):
    self.tracker.mark('image_received')
    print(caller.GetName())
    if self.useScanPlanes[0]:
      self.getNeedle('COR',self.firstVolumePlane0, self.secondVolumePlane0, self.segmentationNodePlane0)

  def receivedImagePlane1(self, caller=None, event=None):
    self.tracker.mark('image_received')
    print(caller.GetName())
    if self.useScanPlanes[1]:
      self.getNeedle('SAG',self.firstVolumePlane1, self.secondVolumePlane1, self.segmentationNodePlane1)
    
  def receivedImagePlane2(self, caller=None, event=None):
    self.tracker.mark('image_received')
    print(caller.GetName())
    if self.useScanPlanes[2]:
      self.getNeedle('AX',self.firstVolumePlane2, self.secondVolumePlane2, self.segmentationNodePlane2)
      
  def getNeedle(self, plane, firstVolume, secondVolume, segMask):
    print('PLANE = %s' %plane)
    # Execute one tracking cycle
    if self.isTrackingOn:
      logFlag = self.logFlagCheckBox.checked
      phaseUnwrap = self.phaseUnwrapCheckBox.checked
      # Get needle tip
      confidence = self.logic.getNeedle(plane, firstVolume, secondVolume, segMask, phaseUnwrap, self.imageConvertion, self.inputVolume, confidenceLevel=self.confidenceLevel, windowSize=self.windowSize, in_channels=self.inputChannels, minTip=self.minTipSize, minShaft=self.minShaftSize, logFlag=logFlag, debugFlag=self.debugFlag, debugName=self.debugName) 
      if confidence is None:
        print('Tracking failed')
        self.tracker.mark('plan_updated') # Consider the scan "update" is to keep current position
      else:
        confidenceText = self.logic.getConfidenceText(confidence)
        print('Tracked with %s confidence' %confidenceText)          
        if self.updateScanPlane is True:   
          if confidence >= self.confidenceLevel:
            if plane=='COR':
              if self.useScanPlanes[1] is True:
                self.logic.updateScanPlane(plane='SAG', sliceOnly=not self.centerScanAtTip, logFlag=logFlag)
                self.logic.pushScanPlaneToIGTLink(self.mrigtlBridgeServerNode, plane='SAG')
              if self.useScanPlanes[2] is True:
                self.logic.updateScanPlane(plane='AX', sliceOnly=not self.centerScanAtTip, logFlag=logFlag)
                self.logic.pushScanPlaneToIGTLink(self.mrigtlBridgeServerNode, plane='AX')
            if plane=='SAG': 
              if self.useScanPlanes[0] is True:
                self.logic.updateScanPlane(plane='COR', sliceOnly=not self.centerScanAtTip, logFlag=logFlag)
                self.logic.pushScanPlaneToIGTLink(self.mrigtlBridgeServerNode, plane='COR')
              if self.useScanPlanes[2] is True:
                self.logic.updateScanPlane(plane='AX', sliceOnly=not self.centerScanAtTip, logFlag=logFlag)
                self.logic.pushScanPlaneToIGTLink(self.mrigtlBridgeServerNode, plane='AX')   
            if plane=='AX':
              if self.useScanPlanes[0] is True:
                self.logic.updateScanPlane(plane='COR', sliceOnly=not self.centerScanAtTip, logFlag=logFlag)
                self.logic.pushScanPlaneToIGTLink(self.mrigtlBridgeServerNode, plane='COR')
              if self.useScanPlanes[1] is True:
                self.logic.updateScanPlane(plane='SAG', sliceOnly=not self.centerScanAtTip, logFlag=logFlag)
                self.logic.pushScanPlaneToIGTLink(self.mrigtlBridgeServerNode, plane='SAG')
          else:
            print('Scan plane NOT updated - No confidence on needle tracking')
            self.tracker.mark('plan_updated') # Consider the scan "update" is to keep current position

        if self.pushTipToRobot is True:
          self.logic.pushTipToIGTLink(self.robotIGTLClientNode)
          self.logic.pushTipConfidenceToIGTLink(self.robotIGTLClientNode)
          print('Tip pushed to robot')
      print('____________________')

################################################################################################################################################
# Logic Class
################################################################################################################################################

class AINeedleTrackingLogic(ScriptedLoadableModuleLogic):

  def __init__(self):
    ScriptedLoadableModuleLogic.__init__(self)
    self.cliParamNode = None

    # Timestamp
    self.tracker = None

    # Image file writer
    self.path = os.path.dirname(os.path.abspath(__file__))
    self.debug_path = os.path.join(self.path,'Debug')
    self.fileWriter = sitk.ImageFileWriter()
    
    # Phase rescaling filter
    self.phaseRescaleFilter = sitk.RescaleIntensityImageFilter()
    self.phaseRescaleFilter.SetOutputMaximum(2*np.pi)
    self.phaseRescaleFilter.SetOutputMinimum(0)    
    
    # Input image masking
    self.sitk_mask0 = None
    self.sitk_mask1 = None
    self.sitk_mask2 = None
    
    # Previous tip detection in each scan plane
    self.prevDetection = [None, None, None]

    # Confidence level labels
    self.confidenceLevelLabels = None
    
    # Check if PLANE_0 node exists, if not, create a new one
    self.scanPlane0TransformNode = slicer.util.getFirstNodeByClassByName('vtkMRMLLinearTransformNode', 'PLANE_0')
    if self.scanPlane0TransformNode is None:
        self.scanPlane0TransformNode = slicer.vtkMRMLLinearTransformNode()
        self.scanPlane0TransformNode.SetName('PLANE_0')
        slicer.mrmlScene.AddNode(self.scanPlane0TransformNode)
    self.initializeScanPlane(plane='COR')
    # Check if PLANE_1 node exists, if not, create a new one
    self.scanPlane1TransformNode = slicer.util.getFirstNodeByClassByName('vtkMRMLLinearTransformNode','PLANE_1')
    if self.scanPlane1TransformNode is None:
        self.scanPlane1TransformNode = slicer.vtkMRMLLinearTransformNode()
        self.scanPlane1TransformNode.SetName('PLANE_1')
        slicer.mrmlScene.AddNode(self.scanPlane1TransformNode)
    self.initializeScanPlane(plane='SAG')
    # Check if PLANE_2 node exists, if not, create a new one
    self.scanPlane2TransformNode = slicer.util.getFirstNodeByClassByName('vtkMRMLLinearTransformNode','PLANE_2')
    if self.scanPlane2TransformNode is None:
        self.scanPlane2TransformNode = slicer.vtkMRMLLinearTransformNode()
        self.scanPlane2TransformNode.SetName('PLANE_2')
        slicer.mrmlScene.AddNode(self.scanPlane2TransformNode)
    self.initializeScanPlane(plane='AX')
    # Check if needleLabelMap nodes exists, if not, create a new one
    self.needleLabelMapNodes = []
    for i in range(3):
      name = f'NeedleLabelMap_{i}'
      labelMapNode = slicer.util.getFirstNodeByClassByName('vtkMRMLLabelMapVolumeNode', name)
      if labelMapNode is None:
        labelMapNode = slicer.vtkMRMLLabelMapVolumeNode()
        labelMapNode.SetName(name)
        slicer.mrmlScene.AddNode(labelMapNode)
        colorTableNode = self.createColorTable()
        labelMapNode.CreateDefaultDisplayNodes()
        labelMapNode.GetDisplayNode().SetAndObserveColorNodeID(colorTableNode.GetID())
      self.needleLabelMapNodes.append(labelMapNode) 
    # Check if text node exists, if not, create a new one
    self.needleConfidenceNode = slicer.util.getFirstNodeByClassByName('vtkMRMLTextNode','CurrentTipConfidence')
    if self.needleConfidenceNode is None:
        self.needleConfidenceNode = slicer.vtkMRMLTextNode()
        self.needleConfidenceNode.SetName('CurrentTipConfidence')
        slicer.mrmlScene.AddNode(self.needleConfidenceNode)
    # Check if segmented tip node exists, if not, create a new one
    self.tipSegmNode = slicer.util.getFirstNodeByClassByName('vtkMRMLLinearTransformNode','SegmentedTip')
    if self.tipSegmNode is None:
        self.tipSegmNode = slicer.vtkMRMLLinearTransformNode()
        self.tipSegmNode.SetName('SegmentedTip')
        slicer.mrmlScene.AddNode(self.tipSegmNode)
    # Check if tracked tip node exists, if not, create a new one
    self.tipTrackedNode = slicer.util.getFirstNodeByClassByName('vtkMRMLLinearTransformNode','CurrentTrackedTip')
    if self.tipTrackedNode is None:
        self.tipTrackedNode = slicer.vtkMRMLLinearTransformNode()
        self.tipTrackedNode.SetName('CurrentTrackedTip')
        slicer.mrmlScene.AddNode(self.tipTrackedNode)

    # Check if Tip point list node exists, if not, create a new one
    self.tipMarkupsNode = slicer.util.getFirstNodeByClassByName('vtkMRMLMarkupsFiducialNode', 'NeedleTip' )
    if self.tipMarkupsNode is None:
        self.tipMarkupsNode = slicer.vtkMRMLMarkupsFiducialNode()
        self.tipMarkupsNode.SetName('NeedleTip')
        slicer.mrmlScene.AddNode(self.tipMarkupsNode)
        self.initializeTipMarkup()

  # Initialize parameter node with default settings
  def setDefaultParameters(self, parameterNode):
    if not parameterNode.GetParameter('InputMode'):
      parameterNode.SetParameter('InputMode', 'Mag/Phase')  
    if not parameterNode.GetParameter('InputVolume'):
      parameterNode.SetParameter('InputVolume', '2D')               
    if not parameterNode.GetParameter('InputChannels'):
      parameterNode.SetParameter('InputChannels', '2CH')               
    if not parameterNode.GetParameter('Model'): 
      parameterNode.SetParameter('Model', '0')    # Index of selected option

    if not parameterNode.GetParameter('ScannerMode'):
      parameterNode.SetParameter('ScannerMode', 'Mag/Phase')  
    if not parameterNode.GetParameter('UseScanPlane0'): 
      parameterNode.SetParameter('UseScanPlane0', 'False')  
    if not parameterNode.GetParameter('UseScanPlane1'): 
      parameterNode.SetParameter('UseScanPlane1', 'False')  
    if not parameterNode.GetParameter('UseScanPlane2'): 
      parameterNode.SetParameter('UseScanPlane2', 'False')              

    if not parameterNode.GetParameter('SetPlane0RAS'): 
      parameterNode.SetParameter('SetPlane0RAS', 'False')  
    if not parameterNode.GetParameter('SetPlane1RAS'): 
      parameterNode.SetParameter('SetPlane1RAS', 'False')  
    if not parameterNode.GetParameter('SetPlane2RAS'): 
      parameterNode.SetParameter('SetPlane2RAS', 'False')  

    if not parameterNode.GetParameter('UpdateScanPlane'): 
      parameterNode.SetParameter('UpdateScanPlane', 'False')   
    if not parameterNode.GetParameter('CenterScanAtTip'): 
      parameterNode.SetParameter('CenterScanAtTip', 'False')     
    if not parameterNode.GetParameter('ConfidenceLevel'): 
      parameterNode.SetParameter('ConfidenceLevel', '2')   # Index of selected option
    if not parameterNode.GetParameter('PushTipToRobot'): 
      parameterNode.SetParameter('PushTipToRobot', 'False')  
    if not parameterNode.GetParameter('ScreenLog'):
      parameterNode.SetParameter('ScreenLog', 'False')     
    if not parameterNode.GetParameter('Debug'):
      parameterNode.SetParameter('Debug', 'False') 
    if not parameterNode.GetParameter('DebugName'): 
      parameterNode.SetParameter('DebugName', '')
    if not parameterNode.GetParameter('WindowSize'):
      parameterNode.SetParameter('WindowSize', '84') 
    if not parameterNode.GetParameter('MinTipSize'):
      parameterNode.SetParameter('MinTipSize', '10')     
    if not parameterNode.GetParameter('MinShaftSize'):
      parameterNode.SetParameter('MinShaftSize', '30') 

  def initializeTipMarkup(self):
    # Ensure there is only one control point
    if self.tipMarkupsNode.GetNumberOfControlPoints() > 1:
        self.tipMarkupsNode.RemoveAllControlPoints()
    # If no control point exists, add one at (0,0,0)
    if self.tipMarkupsNode.GetNumberOfControlPoints() == 0:
        self.tipMarkupsNode.AddControlPoint(vtk.vtkVector3d(0, 0, 0), "T")
    # Ensure tip is labeled, locked and one-point only
    self.tipMarkupsNode.SetNthControlPointLabel(0, "T")
    self.tipMarkupsNode.SetLocked(True)
    self.tipMarkupsNode.SetFixedNumberOfControlPoints(True)           
    displayNode = self.tipMarkupsNode.GetDisplayNode()
    if displayNode:
        displayNode.SetGlyphScale(1)  # 1% glyph size
    else:
        # If the display node does not exist, create it and set the glyph size
        self.tipMarkupsNode.CreateDefaultDisplayNodes()
        self.tipMarkupsNode.GetDisplayNode().SetGlyphScale(1)
    self.setTipMarkupColor()
    # Set the parent transform to self.tipTrackedNode
    if self.tipTrackedNode:
        self.tipMarkupsNode.SetAndObserveTransformNodeID(self.tipTrackedNode.GetID())

  # Set TipMarkup Color according to tracking status
  def setTipMarkupColor(self, tipTracked:bool = None):
    displayNode = self.tipMarkupsNode.GetDisplayNode()
    if tipTracked is None:
      displayNode.SetSelectedColor(1.0, 0.502, 0.502) #PINK (default)
    elif tipTracked is True:
      displayNode.SetSelectedColor(0.667, 1.0, 0.498) #GREEN (default)
    else:
      displayNode.SetSelectedColor(1.0, 1.0, 0.498) #YELLOW (default)

  # Create a ColorTable for the LabelMapNode
  # Lack of ColorTable was generating vtk error messages in the log for Slicer when running in my Mac
  def createColorTable(self):
    colorTableName = 'NeedleColorMap'
    # Check if the color table already exists
    existingNode = slicer.util.getFirstNodeByClassByName("vtkMRMLColorTableNode", colorTableName)
    if existingNode:
        return existingNode
    # If non existant, create a new one
    label_list = [("shaft", 1, 0.2, 0.5, 0.8), ('tip', 2, 1.0, 0.8, 0.7)]
    colorTableNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLColorTableNode")
    colorTableNode.SetHideFromEditors(True)
    colorTableNode.SetTypeToUser()
    colorTableNode.SetName(colorTableName)
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
  # Volume node can be an object volume node (user already created node) or 
  # a string with node name (checks for node with this name and if non existant, creates it with provided type)
  def pushSitkToSlicerVolume(self, sitk_image, node: slicer.vtkMRMLScalarVolumeNode or slicer.vtkMRMLLabelMapVolumeNode or str, type='vtkMRMLScalarVolumeNode'):
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

  # Check if two binary images have pixels close to each other by a given distance (default = 3px)
  def checkIfAdjacent(self, sitk_tip, sitk_shaft, distance=3):
    sitk_dilated_tip = sitk.BinaryDilate(sitk_tip, (distance, distance, distance))
    intersection = sitk_dilated_tip & sitk_shaft
    intersection_stats = sitk.StatisticsImageFilter()
    intersection_stats.Execute(intersection)
    # Check if there are any non-zero pixels in the intersection
    if intersection_stats.GetSum() > 0:
      return True
    else:
      return False

  # Given an sitk_label image, return the labeled separated components and a dictionary with the stats sorted in descending order by centroid size
  def separateComponents(self, sitk_label):
    if sitk.GetArrayFromImage(sitk_label).sum() > 0:
      # Separate in components
      sitk_components = sitk.ConnectedComponent(sitk_label)
      stats = sitk.LabelShapeStatisticsImageFilter()
      stats.Execute(sitk_components)
      # Get labels sizes and centroid physical coordinates
      labels = stats.GetLabels()
      labels_size = []
      labels_centroid = []
      for l in labels:
        number_pixels = stats.GetNumberOfPixels(l)
        centroid = stats.GetCentroid(l)
        labels_size.append(number_pixels)
        labels_centroid.append(centroid)    
      # Combine the lists into a dictionary and sort by size in descending order
      dict_components = [{'label': label, 'size': size, 'centroid': centroid} for label, size, centroid in zip(labels, labels_size, labels_centroid)]
      dict_components = sorted(dict_components, key=lambda x: x['size'], reverse=True)
      return (sitk_components, dict_components)
    else:
      return (None, None)
  
  # Close segmentation gaps in the
  def connectShaftGaps(self, sitk_image, gap_direction=[0, 3, 0]):
    # Apply a binary closing operation (dilation followed by erosion)
    sitk_dilated = sitk.BinaryDilate(sitk_image, gap_direction)
    return sitk.BinaryErode(sitk_dilated, gap_direction)

  # Given a binary skeleton image (single pixel-wide), find the physical coordinates of extremity closer to the image center
  def getShaftTipCoordinates(self, sitk_line):
    # Get the coordinates of all non-zero pixels in the binary image
    nonzero_coords = np.argwhere(sitk.GetArrayFromImage(sitk_line) == 1)
    # Calculate the distance of each non-zero pixel to all others
    distances = np.linalg.norm(nonzero_coords[:, None, :] - nonzero_coords[None, :, :], axis=-1)
    print(distances.shape)
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

  def setConfidenceLevelLabels(self, labels):
    self.confidenceLevelLabels = labels
  
  def getConfidenceText(self, confidenceLevel):
    # Search for the number in the list and return the corresponding text
    for text, value in self.confidenceLevelLabels:
        if value == confidenceLevel:
            return text
    return None
    
  # Return string with the image direction name
  def getDirectionName(self, sitk_image):
    AX_DIR = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    COR_DIR = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0) 
    SAG_DIR = (0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    direction = sitk_image.GetDirection()
    if direction == AX_DIR:
        return 'AX'
    elif direction == SAG_DIR:
        return 'SAG'
    elif direction == COR_DIR:
        return 'COR'
    else:
        return 'Reformat'
      
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
  
  # Pull the real/imaginary volumes from the MRML scene and convert them to magnitude/phase volumes
  def realImagToMagPhase(self, realVolume, imagVolume):
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

  # Pull the magnitude/phase volumes from the MRML scene and convert them to real/imaginary volumes
  def magPhaseToRealImag(self, magVolume, phaseVolume):
    sitk_mag = sitkUtils.PullVolumeFromSlicer(magVolume)
    sitk_phase = sitkUtils.PullVolumeFromSlicer(phaseVolume)
    numpy_mag = sitk.GetArrayFromImage(sitk_mag)
    numpy_phase = sitk.GetArrayFromImage(sitk_phase)
    # Scaling
    p_max = np.max(numpy_phase)
    p_min = np.min(numpy_phase)
    if p_min<0:
      if p_max>0:
        numpy_phase = numpy_phase / p_max*np.pi
    elif p_min >=0:
      if p_max>0:
        numpy_phase = numpy_phase / p_max*2.*np.pi
    numpy_comp = numpy_mag * np.cos(numpy_phase) + 1j* numpy_mag * np.sin(numpy_phase)
    numpy_real = numpy_comp.real
    numpy_imag = numpy_comp.imag
    sitk_real = self.numpyToitk(numpy_real, sitk_mag, type=sitk.sitkFloat32)
    sitk_imag = self.numpyToitk(numpy_imag, sitk_mag, type=sitk.sitkFloat32)
    return (sitk_real, sitk_imag)

  # Unwrap the phase sitk image
  # Using code from https://github.com/maribernardes/SimpleNeedleTracking-3DSlicer/blob/master/SimpleNeedleTracking/SimpleNeedleTracking.py
  def phaseUnwrapItk(self, sitk_phase):
    # Rescale phase
    sitk_phase = self.phaseRescaleFilter.Execute(sitk_phase)
    # Unwrapped base phase
    numpy_base_p = sitk.GetArrayFromImage(sitk_phase)
    if numpy_base_p.shape[0] == 1: # 2D image in a 3D array: make it 2D array for improved performance
        array_p_unwraped = np.ma.copy(numpy_base_p)  # Initialize unwraped array as the original
        array_p_unwraped[0,:,:] = unwrap_phase(numpy_base_p[0,:,:], wrap_around=(False,False))               
    else:
        array_p_unwraped = unwrap_phase(numpy_base_p, wrap_around=(False,False,False))   
    # Put in sitk image
    sitk_unwrapped_phase = sitk.GetImageFromArray(array_p_unwraped)
    sitk_unwrapped_phase.CopyInformation(sitk_phase)
    return sitk_unwrapped_phase
  
  # Build a sitk mask volume from a segmentation node
  def getMaskFromSegmentation(self, segmentationNode, referenceVolumeNode):
    if segmentationNode is not None and referenceVolumeNode is not None:
      # Create a temporary labelmap node
      maskLabelMapNode = slicer.vtkMRMLLabelMapVolumeNode()
      slicer.mrmlScene.AddNode(maskLabelMapNode)
      # Create mask from segmentation
      slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, maskLabelMapNode, referenceVolumeNode)
      sitk_mask = sitkUtils.PullVolumeFromSlicer(maskLabelMapNode)
      sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
      # Remove temporary labelmap node
      slicer.mrmlScene.RemoveNode(maskLabelMapNode)
      return sitk_mask
    else:
      return None
  
  def setupUNet(self, inputVolume, in_channels, model, out_channels=3):
    # Setup UNet model
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
    self.model.load_state_dict(torch.load(model, weights_only=True, map_location=device))
    ## Setup transforms
    if inputVolume == '2':
      pixel_dim = (6, 1.171875, 1.171875)
    else:
      pixel_dim = (3.6, 1.171875, 1.171875)
    # Define pre-inference transforms
    if in_channels==2:
      pre_array = [
        # 2-channel input
        LoadSitkImaged(keys=['image_1', 'image_2']),
        EnsureChannelFirstd(keys=['image_1', 'image_2']), 
        ConcatItemsd(keys=['image_1', 'image_2'], name='image'),
      ]     
    elif in_channels==3:
      pre_array = [
        # 3-channel input
        LoadSitkImaged(keys=['image_1', 'image_2', 'image_3']),
        EnsureChannelFirstd(keys=['image_1', 'image_2', 'image_3']), 
        ConcatItemsd(keys=['image_1', 'image_2', 'image_3'], name='image'),
      ]          
    else:
      pre_array = [
        # 1-channel input
        LoadSitkImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
      ]
    pre_array.append(ScaleIntensityd(keys=['image'], minv=0, maxv=1, channel_wise=True))
    # Separate COR / SAG / AX
    pre_array_cor = copy.deepcopy(pre_array)
    pre_array_sag = copy.deepcopy(pre_array)
    pre_array_ax = copy.deepcopy(pre_array)
    pre_array_cor.append(Orientationd(keys=['image'], axcodes='PIL')) #PIL
    pre_array_sag.append(Orientationd(keys=['image'], axcodes='RIP')) #LIP
    pre_array_ax.append(Orientationd(keys=['image'], axcodes='SPL'))  #AX
    
    pre_array_cor.append(Spacingd(keys=['image'], pixdim=pixel_dim, mode=('bilinear')))
    pre_array_sag.append(Spacingd(keys=['image'], pixdim=pixel_dim, mode=('bilinear')))
    pre_array_ax.append(Spacingd(keys=['image'], pixdim=pixel_dim, mode=('bilinear')))
    
    self.pre_transforms_cor = Compose(pre_array_cor)
    self.pre_transforms_sag = Compose(pre_array_sag)
    self.pre_transforms_ax = Compose(pre_array_ax)
    
    # Define post-inference transforms
    self.post_transforms = Compose([ AsDiscreted(keys=['pred'], argmax=True, num_classes=3),
                                     PushSitkImaged(keys=['pred'], resample=True, print_log=False)
                                  ])  

  # Reset tracking values
  def initializeTracking(self, useScanPlanes):
    self.prevDetection = [False if plane else None for plane in useScanPlanes]
    # Reset tip transform nodes
    identityMatrix = vtk.vtkMatrix4x4()
    identityMatrix.Identity()
    self.tipSegmNode.SetMatrixTransformToParent(identityMatrix)    
    self.tipTrackedNode.SetMatrixTransformToParent(identityMatrix)    

  # Initialize AI model
  def initializeModel(self, inputMode, inputVolume, in_channels, modelName):
    modelFilePath = os.path.join(self.path, 'Models', inputMode, str(inputVolume)+'D-'+str(in_channels)+'CH', modelName)
    self.setupUNet(inputVolume, in_channels, modelFilePath) # Setup UNet

  # Initialize masks
  def initializeMasks(self, segmentationNodePlane0, firstVolumePlane0, segmentationNodePlane1, firstVolumePlane1, segmentationNodePlane2, firstVolumePlane2):
    self.sitk_mask0 = self.getMaskFromSegmentation(segmentationNodePlane0, firstVolumePlane0)    # Update mask (None if nothing in segmentationNode or firstVolume)
    self.sitk_mask1 = self.getMaskFromSegmentation(segmentationNodePlane1, firstVolumePlane1)    # Update mask (None if nothing in segmentationNode)
    self.sitk_mask2 = self.getMaskFromSegmentation(segmentationNodePlane2, firstVolumePlane2)    # Update mask (None if nothing in segmentationNode)

  '''
  def initializeZFrame(self, zFrameToWorld):
    # Get world to ZFrame transformations
    worldToZFrame = vtk.vtkMatrix4x4()
    zFrameToWorld.GetMatrixTransformFromWorld(worldToZFrame)
    # Set it to worldToZFrameNode
    self.worldToZFrameNode.SetMatrixTransformToParent(worldToZFrame)
  '''

  # Set Scan Plane Orientation
  # Default position is (0,0,0), unless center is specified 
  # If sliceOnly, will set position of the slice only (keep the other coordinates as defined by previous values)
  def initializeScanPlane(self, coordinates=(0,0,0), plane='COR', sliceOnly=False):
    m = vtk.vtkMatrix4x4()
    if plane == 'COR':
      # Set rotation
      m.SetElement(0, 0, 1); m.SetElement(0, 1, 0); m.SetElement(0, 2, 0)
      m.SetElement(1, 0, 0); m.SetElement(1, 1, 0); m.SetElement(1, 2, -1)
      m.SetElement(2, 0, 0); m.SetElement(2, 1, 1); m.SetElement(2, 2, 0)
      # Set translation
      if sliceOnly:
        m.SetElement(1, 3, coordinates[1])
      else:
        m.SetElement(0, 3, coordinates[0]); m.SetElement(1, 3, coordinates[1]); m.SetElement(2, 3, coordinates[2])
      self.scanPlane0TransformNode.SetMatrixTransformToParent(m)
    elif plane == 'SAG':
      # Set rotation
      m.SetElement(0, 0, 0); m.SetElement(0, 1, 0); m.SetElement(0, 2, 1)
      m.SetElement(1, 0, 0); m.SetElement(1, 1, 1); m.SetElement(1, 2, 0)
      m.SetElement(2, 0, -1); m.SetElement(2, 1, 0); m.SetElement(2, 2, 0)
      # Set translation
      if sliceOnly:
        m.SetElement(0, 3, coordinates[0])
      else:
        m.SetElement(0, 3, coordinates[0]); m.SetElement(1, 3, coordinates[1]); m.SetElement(2, 3, coordinates[2])
      self.scanPlane1TransformNode.SetMatrixTransformToParent(m)
    elif plane == 'AX':
      # Set rotation
      m.SetElement(0, 0, 1); m.SetElement(0, 1, 0); m.SetElement(0, 2, 0)
      m.SetElement(1, 0, 0); m.SetElement(1, 1, 1); m.SetElement(1, 2, 0)
      m.SetElement(2, 0, 0); m.SetElement(2, 1, 0); m.SetElement(2, 2, 1)
      # Set translation
      if sliceOnly:
        m.SetElement(2, 3, coordinates[2])
      else:
        m.SetElement(0, 3, coordinates[0]); m.SetElement(1, 3, coordinates[1]); m.SetElement(2, 3, coordinates[2])
      self.scanPlane2TransformNode.SetMatrixTransformToParent(m)
    else: #Other - Still not supported
      print('Invalid plane option')

  # Chooses which scan to update
  def updateScanPlane(self, plane='COR', sliceOnly=False, logFlag=False):
    # Get current scan plane
    plane_matrix = vtk.vtkMatrix4x4()
    if plane == 'COR':    # PLAN_0
      self.scanPlane0TransformNode.GetMatrixTransformToParent(plane_matrix) 
    elif plane == 'SAG':  # PLAN_1
      self.scanPlane1TransformNode.GetMatrixTransformToParent(plane_matrix)
    elif plane == 'AX':   # PLAN_2
      self.scanPlane2TransformNode.GetMatrixTransformToParent(plane_matrix)
    else:
      print('Invalid scan plane')
      return
    # Get current tip transform
    tip_matrix = vtk.vtkMatrix4x4()
    self.tipTrackedNode.GetMatrixTransformToParent(tip_matrix)
    # Set matrix with current tip
    if (sliceOnly is False): # Update all coordinates (center at needle tip)
      print('Update all coordinates of slice')
      plane_matrix.SetElement(0, 3, tip_matrix.GetElement(0, 3))
      plane_matrix.SetElement(1, 3, tip_matrix.GetElement(1, 3))
      plane_matrix.SetElement(2, 3, tip_matrix.GetElement(2, 3))      
    else:                   # Update only slice coordinate (not centering at needle tip)
      if plane == 'COR':
        plane_matrix.SetElement(1, 3, tip_matrix.GetElement(1, 3))
        self.scanPlane0TransformNode.SetMatrixTransformToParent(plane_matrix)
      elif plane == 'SAG':
        plane_matrix.SetElement(0, 3, tip_matrix.GetElement(0, 3))
        self.scanPlane1TransformNode.SetMatrixTransformToParent(plane_matrix)
      elif plane == 'AX':
        plane_matrix.SetElement(2, 3, tip_matrix.GetElement(2, 3))
        self.scanPlane2TransformNode.SetMatrixTransformToParent(plane_matrix)
    # Update plane transform node
    if plane == 'COR':    # PLAN_0
      self.scanPlane0TransformNode.SetMatrixTransformToParent(plane_matrix) 
    elif plane == 'SAG':  # PLAN_1
      self.scanPlane1TransformNode.SetMatrixTransformToParent(plane_matrix)
    elif plane == 'AX':   # PLAN_2
      self.scanPlane2TransformNode.SetMatrixTransformToParent(plane_matrix)          

    if logFlag:
      scanPlaneCenter = [plane_matrix.GetElement(0,3), plane_matrix.GetElement(1,3), plane_matrix.GetElement(2,3)]
      if plane=='COR':
        print('PLAN_0 = %s' %scanPlaneCenter)
      elif plane=='SAG':
        print('PLAN_1 = %s' %scanPlaneCenter)
      elif plane=='AX':
        print('PLAN_2 = %s' %scanPlaneCenter)
      
    return

  def pushScanPlaneToIGTLink(self, connectionNode, plane='COR'):
    if plane=='COR':    # PLAN_0
      connectionNode.RegisterOutgoingMRMLNode(self.scanPlane0TransformNode)
      connectionNode.PushNode(self.scanPlane0TransformNode)
      self.tracker.mark('plan_updated')
      connectionNode.UnregisterOutgoingMRMLNode(self.scanPlane0TransformNode)
    elif plane=='SAG':  # PLAN_1
      connectionNode.RegisterOutgoingMRMLNode(self.scanPlane1TransformNode)
      connectionNode.PushNode(self.scanPlane1TransformNode)
      self.tracker.mark('plan_updated')
      connectionNode.UnregisterOutgoingMRMLNode(self.scanPlane1TransformNode)
    if plane=='AX':     # PLAN_2
      connectionNode.RegisterOutgoingMRMLNode(self.scanPlane2TransformNode)
      connectionNode.PushNode(self.scanPlane2TransformNode)
      self.tracker.mark('plan_updated')
      connectionNode.UnregisterOutgoingMRMLNode(self.scanPlane2TransformNode)

  '''
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
  '''

  def pushTipToIGTLink(self, connectionNode):
    '''
    # Apply zTransform to currentTip
    self.tipTrackedZNode.CopyContent(self.tipTrackedNode)
    self.tipTrackedZNode.SetAndObserveTransformNodeID(self.worldToZFrameNode.GetID())
    self.tipTrackedZNode.HardenTransform()
    #  Push to IGTLink:
    # zFrame
    connectionNode.RegisterOutgoingMRMLNode(self.tipTrackedZNode)
    connectionNode.PushNode(self.tipTrackedZNode)
    connectionNode.UnregisterOutgoingMRMLNode(self.tipTrackedZNode)
    '''
    # scanner frame
    connectionNode.RegisterOutgoingMRMLNode(self.tipTrackedNode)
    connectionNode.PushNode(self.tipTrackedNode)
    connectionNode.UnregisterOutgoingMRMLNode(self.tipTrackedNode)

  def pushTipConfidenceToIGTLink(self, connectionNode):
    # needle tracking confidence
    connectionNode.RegisterOutgoingMRMLNode(self.needleConfidenceNode)
    connectionNode.PushNode(self.needleConfidenceNode)
    connectionNode.UnregisterOutgoingMRMLNode(self.needleConfidenceNode)

  def getNeedle(self, plane, firstVolume, secondVolume, segMask, phaseUnwrap, imageConversion, inputVolume, confidenceLevel=3, windowSize=84, in_channels=2, out_channels=3, minTip=10, minShaft=30, logFlag=False, debugFlag=False, debugName=''):    
    # Increment tracking counter
    image_count = len(self.tracker.timestamps['image_received'])
    print('Image #%i' %(image_count))

    ######################################
    ##                                  ##
    ## Step 0: Set input images         ##
    ##                                  ##
    ######################################

    # Update mask
    sitk_mask = self.getMaskFromSegmentation(segMask, firstVolume)    # Update mask (None if nothing in segmentationNode or firstVolume)

    # Get sitk images from MRML volume nodes 
    if (imageConversion == 'RealImag'): # Convert to magnitude/phase
      (sitk_img_m, sitk_img_p) = self.magPhaseToRealImag(firstVolume, secondVolume)
    elif (imageConversion == 'MagPhase'):
      (sitk_img_m, sitk_img_p) = self.realImagToMagPhase(firstVolume, secondVolume)
    else:                         # Conversion is None
      sitk_img_m = sitkUtils.PullVolumeFromSlicer(firstVolume)
      if (in_channels!=1):
        sitk_img_p = sitkUtils.PullVolumeFromSlicer(secondVolume)
    # Cast it to 32Float
    sitk_img_m = sitk.Cast(sitk_img_m, sitk.sitkFloat32)
    if (in_channels!=1):
      sitk_img_p = sitk.Cast(sitk_img_p, sitk.sitkFloat32)
    # 3-channels input
    if in_channels == 3:
      (sitk_img_a, _) = self.realImagToMagPhase(firstVolume, secondVolume)
      sitk_img_a = sitk.Cast(sitk_img_a, sitk.sitkFloat32) #Cast it to 32Float
    # Phase unwrap
    if (phaseUnwrap is True) and (imageConversion != 'RealImag'):
      sitk_img_p = self.phaseUnwrapItk(sitk_img_p)

    # plane = self.getDirectionName(sitk_img_m)
    #TODO: Check getDirectionName function

    ######################################
    ##                                  ##
    ## Step 0b: Mask inputs             ##
    ##                                  ##
    ######################################

    # Apply segmentation mask (optional)
    if sitk_mask is not None:
      sitk_img_m = sitk.Mask(sitk_img_m, sitk_mask)
      if (in_channels!=1):
        sitk_img_p = sitk.Mask(sitk_img_p, sitk_mask)
      if in_channels == 3:
        sitk_img_a = sitk.Mask(sitk_img_a, sitk_mask)

    # Push debug images to Slicer     
    if debugFlag:
      self.pushSitkToSlicerVolume(sitk_img_m, 'debug_img_m')
      self.saveSitkImage(sitk_img_m, name='debug_img_m_'+str(image_count), path=os.path.join(self.path, 'Debug', debugName))
      if (in_channels!=1):
        self.pushSitkToSlicerVolume(sitk_img_p, 'debug_img_p')
        self.saveSitkImage(sitk_img_p, name='debug_img_p_'+str(image_count), path=os.path.join(self.path, 'Debug', debugName))
      if in_channels == 3:
        self.pushSitkToSlicerVolume(sitk_img_a, 'debug_img_a')
        self.saveSitkImage(sitk_img_a, name='debug_img_a_'+str(image_count), path=os.path.join(self.path, 'Debug', debugName))
      if sitk_mask is not None:
        self.pushSitkToSlicerVolume(sitk_mask, 'debug_mask')
        self.saveSitkImage(sitk_mask, name='debug_mask_'+str(image_count), path=os.path.join(self.path, 'Debug', debugName))

    ######################################
    ##                                  ##
    ## Step 1: Set input dictionary     ##
    ##                                  ##
    ######################################
    # Set input dictionary
    if in_channels==2:
      input_dict = {'image_1': sitk_img_m, 'image_2': sitk_img_p}
    elif in_channels==3:
      input_dict = {'image_1': sitk_img_m, 'image_2': sitk_img_p, 'image_3': sitk_img_a}    
    else:
      input_dict = {'image':sitk_img_m}
      
    # Adjust window_size to input volume
    if inputVolume == 2:
      window_size = (1, windowSize, windowSize)
    else:
      window_size = (3, windowSize, windowSize)      

    ######################################
    ##                                  ##
    ## Step 2: Inference                ##
    ##                                  ##
    ######################################
    self.tracker.mark('image_prepared')
    # Apply pre_transforms
    if plane == 'AX':
      pre_transforms = self.pre_transforms_ax
    elif plane == 'SAG':
      pre_transforms = self.pre_transforms_sag
    else:
      pre_transforms = self.pre_transforms_cor
    data = pre_transforms(input_dict)
    # Evaluate model
    self.model.eval()
    with torch.no_grad():
      batch_input = data['image'].unsqueeze(0)
      val_inputs = batch_input.to(torch.device('cpu'))
      val_outputs = sliding_window_inference(val_inputs, window_size, 1, self.model)
      data['pred'] = val_outputs[0]
    # Apply post-transform
    data = self.post_transforms(data)
    sitk_output = data['pred']
    self.tracker.mark('needle_segmented')

    # Push segmentation to Slicer
    if plane == 'AX':
      self.pushSitkToSlicerVolume(sitk_output, self.needleLabelMapNodes[2])
    elif plane == 'SAG':
      self.pushSitkToSlicerVolume(sitk_output, self.needleLabelMapNodes[1])
    else:
      self.pushSitkToSlicerVolume(sitk_output, self.needleLabelMapNodes[0])
    
    if debugFlag:
      self.saveSitkImage(sitk_output, name='debug_labelmap_'+str(image_count), path=os.path.join(self.path, 'Debug', debugName), is_label=True)

    ######################################
    ##                                  ##
    ## Step 3: Separate tip elements    ##
    ##                                  ##
    ######################################    

    # Separate labels
    sitk_tip = (sitk_output==2)
    sitk_shaft = (sitk_output==1)

    if debugFlag:
      self.saveSitkImage(sitk_tip, name='debug_shaft_'+str(image_count), path=os.path.join(self.path, 'Debug', debugName), is_label=True)
      self.saveSitkImage(sitk_shaft, name='debug_shaft_'+str(image_count), path=os.path.join(self.path, 'Debug', debugName), is_label=True)

    # Separate tip from segmentation
    (sitk_tip_components, tip_dict) = self.separateComponents(sitk_tip)
    # if debugFlag:
      # self.pushSitkToSlicerVolume(sitk_tip, 'debug_tip')
    #if logFlag:
    #  if tip_dict is not None:
    #    for element in tip_dict:
    #      print('Tip Label %s: -> Size: %s, Center: %s' %(element['label'], element['size'], element['centroid']))
    #  else:
    #    print('No tip segmentation')

    ######################################
    ##                                  ##
    ## Step 4: Separate shaft elements  ##
    ##                                  ##
    ######################################        
        
    # Close segmentation gaps    
    sitk_shaft = self.connectShaftGaps(sitk_shaft)
    # Separate shaft from segmentation
    (sitk_shaft_components, shaft_dict) = self.separateComponents(sitk_shaft)
    # if debugFlag:
      # self.pushSitkToSlicerVolume(sitk_shaft, 'debug_shaft')
    #if logFlag:
    #  if shaft_dict is not None:
    #    for element in shaft_dict:
    #      print('Shaft Label %s: -> Size: %s, Center: %s' %(element['label'], element['size'], element['centroid']))
    #  else:
    #    print('No shaft segmentation')    

    ######################################
    ##                                  ##
    ## Step 5: Select tip/shaft labels  ##
    ##                                  ##
    ######################################    

    # Initialize selected labels
    tip_label = None
    tip_label2 = None
    shaft_label = None
    shaft_label2 = None
        
    # Selected largest shaft
    if shaft_dict is not None:
      shaft_label = shaft_dict[0]['label']
      shaft_size = shaft_dict[0]['size']
      sitk_selected_shaft = sitk.BinaryThreshold(sitk_shaft_components, lowerThreshold=shaft_label, upperThreshold=shaft_label, insideValue=1, outsideValue=0)
      # Is 2nd largest a candidate?
      if len(shaft_dict)>1:
        shaft_size2 = shaft_dict[1]['size']
        if shaft_size2 >= minShaft:
          shaft_label2 = shaft_dict[1]['label']
      if debugFlag:
        self.saveSitkImage(sitk_selected_shaft, name='debug_selected_shaft_'+str(image_count), path=os.path.join(self.path, 'Debug', debugName), is_label=True)
        self.pushSitkToSlicerVolume(sitk_selected_shaft, 'debug_selected_shaft')
    
    # Select largest tip
    if tip_dict is not None: 
      tip_label = tip_dict[0]['label']
      tip_size = tip_dict[0]['size']
      tip_center = tip_dict[0]['centroid']
      sitk_selected_tip = sitk.BinaryThreshold(sitk_tip_components, lowerThreshold=tip_label, upperThreshold=tip_label, insideValue=1, outsideValue=0)
      # Is 2nd largest a candidate?
      if len(tip_dict)>1:
        tip_size2 = tip_dict[1]['size']
        if tip_size2 >= minTip:
          tip_label2 = tip_dict[1]['label']
          tip_center2 = tip_dict[1]['centroid']
      if debugFlag:
        self.saveSitkImage(sitk_selected_tip, name='debug_selected_tip_'+str(image_count), path=os.path.join(self.path, 'Debug', debugName), is_label=True)
        self.pushSitkToSlicerVolume(sitk_selected_tip, 'debug_selected_tip')

    # Check tip and shaft connection
    connected = False
    if (shaft_label is not None) and (tip_label is not None):
        connected = self.checkIfAdjacent(sitk_selected_tip, sitk_selected_shaft) # S1T1
        if (connected is False):
          if (tip_label2 is not None): #Tip1 not connected to shaft1 - Check Tip2
            sitk_selected_tip2 = sitk.BinaryThreshold(sitk_tip_components, lowerThreshold=tip_label2, upperThreshold=tip_label2, insideValue=1, outsideValue=0)         
            connected = self.checkIfAdjacent(sitk_selected_tip2, sitk_selected_shaft) #S1T2
            if connected is True: #Change selection to tip2
              tip_label = tip_label2
              tip_center = tip_center2
              tip_size = tip_size2
              sitk_selected_tip = sitk_selected_tip2
            elif (shaft_label2 is not None): #Tip2 not connected to shaft1 - Check shaft2
              sitk_selected_shaft2 = sitk.BinaryThreshold(sitk_shaft_components, lowerThreshold=shaft_label2, upperThreshold=shaft_label2, insideValue=1, outsideValue=0)
              connected = self.checkIfAdjacent(sitk_selected_tip, sitk_selected_shaft2) #S2T1
              if (connected is True): #Change selection to shaft2
                shaft_label = shaft_label2
                shaft_size = shaft_size2
                sitk_selected_shaft = sitk_selected_shaft2
              elif (tip_label2 is not None): #Tip1 not connected to shaft2 - Check Tip2
                connected = self.checkIfAdjacent(sitk_selected_tip2, sitk_selected_shaft2) #S2T2
                if (connected is True): #Change selection to tip2 and shaft2
                  tip_label = tip_label2
                  tip_center = tip_center2
                  tip_size = tip_size2
                  sitk_selected_tip = sitk_selected_tip2
                  shaft_label = shaft_label2
                  shaft_size = shaft_size2
                  sitk_selected_shaft = sitk_selected_shaft2                
          elif (shaft_label2 is not None): #Tip1 not connected to shaft1 and NO Tip2 - Check shaft2
            sitk_selected_shaft2 = sitk.BinaryThreshold(sitk_shaft_components, lowerThreshold=shaft_label2, upperThreshold=shaft_label2, insideValue=1, outsideValue=0)
            connected = self.checkIfAdjacent(sitk_selected_tip, sitk_selected_shaft2) #S2T1
            if (connected is True): #Change selection to shaft2
              shaft_label = shaft_label2
              shaft_size = shaft_size2
              sitk_selected_shaft = sitk_selected_shaft2  
            
    #if logFlag:
    #  print('SELECTED: tip = %s, shaft = %s, connected = %s ' %(tip_label, shaft_label, connected))  

    ######################################
    ##                                  ##
    ## Step 6: Set confidence for tip   ##
    ##                                  ##
    ######################################
    
    # Define confidence on tip estimate
    # CONFIDENCE LEVELS
    # Low = 1
    # Medium Low = 2
    # Medium = 3
    # Medium High = 4
    # High = 5
    # None = no segmentation

    # NO TIP
    if tip_label is None: 
      # NO TIP and NO SHAFT
      if shaft_label is None:
        if logFlag:
          print('Segmented tip = None')
          # Does not update anything
        self.setTipMarkupColor(tipTracked=False)  
        self.tracker.mark('tip_tracked') # Consider no tip to be the tracked tip result
        return None  # NONE (no tip, no shaft)
      # NO TIP WITH SHAFT
      else:
        sitk_skeleton = sitk.BinaryThinning(sitk_selected_shaft)
        shaft_tip = self.getShaftTipCoordinates(sitk_skeleton)
        tip_center = shaft_tip          
        if shaft_size >= minShaft:
          confidence = 2      # MEDIUM LOW (no tip, big shaft) - Use shaft tip
        else:
          confidence = 1      # LOW (no tip, small shaft) - Use shaft tip
    # WITH TIP and NO SHAFT
    elif shaft_label is None:
      if tip_size >= minTip: 
        confidence = 3   # MEDIUM (big tip, no shaft) - Use tip center
      else:
        confidence = 1      # LOW (small tip, no shaft) - Use tip center
    # WITH TIP and SHAFT <CONNECTED>
    elif connected:
        if tip_size >= minTip: 
          confidence = 5       # HIGH (big tip and shaft connected) - Use tip center           
        elif shaft_size >= minShaft:
          confidence = 4  # MEDIUM HIGH (small tip and big shaft connected) - Use tip center
        else:
          confidence = 3       # MEDIUM (small tip and small shaft connected) - Use tip center
    # WITH TIP and SHAFT <NOT CONNECTED>
    else:
      if tip_size >= minTip: 
        confidence = 3  # MEDIUM LOW (big tip NOT connected)
      elif shaft_size >= minShaft:
        sitk_skeleton = sitk.BinaryThinning(sitk_selected_shaft)
        if debugFlag:
          self.saveSitkImage(sitk_skeleton, name='debug_skeleton_'+str(image_count), path=os.path.join(self.path, 'Debug', debugName), is_label=True)
        shaft_tip = self.getShaftTipCoordinates(sitk_skeleton)
        tip_center = shaft_tip          
        confidence = 2   # MEDIUM LOW (small tip, big shaft) - Use shaft tip
      else:
        confidence = 1          # LOW (small tip and small shaft) - Use tip center 
    
    ####################################
    ##                                ##
    ## Step 7: Update tipSegmNode     ##
    ##                                ##
    ####################################

    # Convert to 3D Slicer coordinates (RAS)
    centerRAS = [-tip_center[0], -tip_center[1], tip_center[2]]
    # # Plot
    if logFlag:
      #print('Segmented tip = %s' %centerRAS)
      print("Segmented tip = [%.4f, %.4f, %.4f]" % tuple(centerRAS))

    # Push coordinates to tip Node
    segmTipMatrix = vtk.vtkMatrix4x4()
    segmTipMatrix.SetElement(0,3, centerRAS[0])
    segmTipMatrix.SetElement(1,3, centerRAS[1])
    segmTipMatrix.SetElement(2,3, centerRAS[2])
    self.tipSegmNode.SetMatrixTransformToParent(segmTipMatrix)

    ####################################
    ##                                ##
    ## Step 8: Push to tipTrackedNode ##
    ##                                ##
    ####################################

    #if sitk_img_m.GetDepth()>1 or : # Update slice coordinates if first estimate or if 3D
    #  updateSlice = True
    #else:
    #  updateSlice = False

    if (confidence >= confidenceLevel): 
      # Get current tip transform
      trackTipMatrix = vtk.vtkMatrix4x4()
      self.tipTrackedNode.GetMatrixTransformToParent(trackTipMatrix)
      if plane == 'COR':
      # Update tracked tip L/R and I/S coordinates
        trackTipMatrix.SetElement(0,3, centerRAS[0])
        trackTipMatrix.SetElement(2,3, centerRAS[2])
        if not (self.prevDetection[1] is True or self.prevDetection[2] is True): # No tip in other planes
          trackTipMatrix.SetElement(1,3, centerRAS[1])                 # Update COR slice position
        self.prevDetection[0] = True
      elif plane == 'SAG':
      # Update tracked tip A/P and I/S coordinates
        trackTipMatrix.SetElement(1,3, centerRAS[1])
        trackTipMatrix.SetElement(2,3, centerRAS[2])      
        if not (self.prevDetection[0] is True or self.prevDetection[2] is True): # No tip in other planes
          trackTipMatrix.SetElement(0,3, centerRAS[0])                  # Update SAG slice position
        self.prevDetection[1] = True
      elif plane == 'AX':
      # Update tracked tip L/R and A/P coordinates
        trackTipMatrix.SetElement(0,3, centerRAS[0])
        trackTipMatrix.SetElement(1,3, centerRAS[1])      
        if not (self.prevDetection[0] is True or self.prevDetection[1] is True): # No tip in other planes
          trackTipMatrix.SetElement(2,3, centerRAS[2])                  # Update AX slice position
        self.prevDetection[2] = True
      # Set new tip value
      self.tipTrackedNode.SetMatrixTransformToParent(trackTipMatrix)
      self.setTipMarkupColor(tipTracked=True)
      if logFlag:
        tracked = [trackTipMatrix.GetElement(0,3), trackTipMatrix.GetElement(1,3), trackTipMatrix.GetElement(2,3)]
        #print('Tracked tip = %s' %tracked)
        print("Tracked tip = [%.4f, %.4f, %.4f]" % tuple(tracked))
    else:
      self.setTipMarkupColor(tipTracked=False)
      print('Tracked tip not updated (not enough confidence)')
    # Push confidence to Node
    self.needleConfidenceNode.SetText(str(time.time()) + '; ' + self.getConfidenceText(confidence) + '; '+ str(confidence)) 
    self.tracker.mark('tip_tracked')
    return confidence
  
  ############################################
