
from monai.utils import first, set_determinism
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, MetaTensor, decollate_batch, ITKReader
from monai.config import print_config
from monai.apps import download_and_extract

import numpy as np
import torch
import tempfile
import shutil
import os
import glob
import sys
import argparse

from configparser import ConfigParser

import torch
from configparser import ConfigParser
from monai.transforms import (
    Activationsd,
    AdjustContrastd,
    AsDiscrete,
    AsDiscreted,
    # AddChanneld, # Mariana: deprecated. Replace with EnsureChannelFirst
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandAdjustContrastd,
    RandFlipd,
    RandZoomd,
    RandScaleIntensityd,
    RandStdShiftIntensityd,
    RemoveSmallObjectsd,
    SaveImaged,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,    
    Spacingd,
    ToTensord,
)

from monai.utils import first, set_determinism
from monai.networks.nets import UNet
from monai.networks.layers import Norm

import numpy as np
import glob
import os
import shutil
import SimpleITK as sitk
import itk

from monai.data import CacheDataset, DataLoader, Dataset

class Param():
    def __init__(self, root_dir, pixel_dim, window_size, orientation, in_channels, input_type, out_channels, label_type, device_name, min_size_object):
        self.root_dir = root_dir
        self.pixel_dim = pixel_dim
        self.window_size = window_size
        self.axcodes = orientation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_type = input_type
        self.label_type = label_type
        self.device_name = device_name
        self.min_size_object = min_size_object

#--------------------------------------------------------------------------------
# Model
#--------------------------------------------------------------------------------

def setupModel(param):

    model_unet = UNet(
        spatial_dims=3,
        in_channels=param.in_channels,
        out_channels=param.out_channels,
        channels=[16, 32, 64, 128], 
        strides=[(1, 2, 2), (1, 2, 2), (1, 1, 1)], 
        num_res_units=2,
        norm=Norm.BATCH,
    )
    
    post_pred = AsDiscrete(argmax=True, to_onehot=param.out_channels, n_classes=param.out_channels)
    post_label = AsDiscrete(to_onehot=param.out_channels, n_classes=param.out_channels)
    return (model_unet, post_pred, post_label)


def loadInferenceTransforms(param, output_path):
    if param.in_channels==2:
        pre_array = [
            # 2-channel input
            # LoadImaged(keys=["image_1", "image_2"]),
            EnsureChannelFirstd(keys=["image_1", "image_2"]), 
            ConcatItemsd(keys=["image_1", "image_2"], name="image"),
        ]        
    else:
        pre_array = [
            # 1-channel input
            # LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
        ]
    pre_array.append(ScaleIntensityd(keys=["image"], minv=0, maxv=1, channel_wise=True))
    pre_array.append(Orientationd(keys=["image"], axcodes=param.axcodes))
    pre_array.append(Spacingd(keys=["image"], pixdim=param.pixel_dim, mode=("bilinear")))
    pre_transforms = Compose(pre_array)

    # define post transforms
    post_transforms = Compose([
        # EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=pre_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                                # then invert `pred` based on this information. we can use same info
                                # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
                                             # for example, may need the `affine` to invert `Spacingd` transform,
                                             # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
                                           # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
                                           # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                                   # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", argmax=True, num_classes=param.out_channels),
        RemoveSmallObjectsd(keys="pred", min_size=int(param.min_size_object), connectivity=1, independent_channels=False),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_path, output_postfix="seg", resample=False, output_dtype=np.uint16, separate_folder=False),
    ])    
    return (pre_transforms, post_transforms)

def generateFileList(param, input_path):
    print('Reading images from: ' + param.root_dir)
    images_m = sorted(glob.glob(os.path.join(param.root_dir, input_path, "*_M.nii.gz")))
    images_p = sorted(glob.glob(os.path.join(param.root_dir, input_path, "*_P.nii.gz")))
    
    # Use two types of images combined
    if param.in_channels==2:
        data_dicts = [
            {"image_1": image_m_name, "image_2": image_p_name}
            for image_m_name, image_p_name in zip(images_m, images_p)
        ]    
    else:
        # Use phase images
        if param.input_type=='P':
            data_dicts = [
                {"image": image_name} for image_name in images_p
            ]
        # Use magnitude images
        else:
            data_dicts = [
                {"image": image_name} for image_name in images_m
            ]
    return data_dicts
def sitkToMetaTensor(sitk_image):
    array=sitk.GetArrayFromImage(sitk_image)
    tensor = torch.from_numpy(array)
    affine = None
    meta = {
        'size': sitk_image.GetSize(),# Convert sitk image to a NumPy array
        'spacing': sitk_image.GetSpacing(),
        'origin': sitk_image.GetOrigin(),
        'direction': sitk_image.GetDirection(),
    }
    m = MetaTensor(t, affine=affine, meta=meta)

    return 
def itkToDataset(itk_image_1, itk_image_2):
    # Get array from image
    array_1=sitk.GetArrayFromImage(itk_image_1)
    array_2=sitk.GetArrayFromImage(sitk_image_2)
    # Make channel first
    array_1 = array_1[np.newaxis, ...]
    array_2 = array_2[np.newaxis, ...]
    # Concatenate both channels
    array = np.vstack((array_1, array_2))

    # Get metadata from sitk image
    metadata = {
        'size': sitk_image_1.GetSize(),# Convert sitk image to a NumPy array
        'spacing': sitk_image_1.GetSpacing(),
        'origin': sitk_image_1.GetOrigin(),
        'direction': sitk_image_1.GetDirection(),
    }
    return [{"image": array, "metadata": metadata}]

def inference():
    path = os.path.dirname(os.path.abspath(__file__))
    param = Param(path, (3.6, 1.171875, 1.171875), (3, 48, 48), 'PIL', 2, 'MP', 3, 'multi','cpu',50)
    input_path = 'test_data'
    output_path = 'inference_output'
    model_file = 'model_needle_tracking.pth'
    device = torch.device(param.device_name)

    # Load inputs
    val_files = generateFileList(param, input_path)
    n_files = len(val_files)
    print('# of images: ' + str(n_files))
    print(val_files)

    
    itkReader = ITKReader(channel_dim=None)
    itk_image_1 = itk.imread(val_files[0]['image_1']).astype(itk.F)
    itk_image_2 = itk.imread(val_files[0]['image_2']).astype(itk.F)

    (array_1, metadata_1) = itkReader.get_data(img=itk_image_1)
    (array_2, medatada_2) = itkReader.get_data(img=itk_image_2)

    print(array_1)
    print(metadata_1)
    # data_list = [{"image_1": metaTensor1, "image_2": metaTensor2}]

    print('Loading dataset')
    (pre_transforms, post_transforms) =  loadInferenceTransforms(param, output_path)
    
    # Initialize the ArrayDataset
    # dataset = ArrayDataset(data=data_list, transform=None, image_only=False)

    # Create a data loader for batch processing
    val_ds = CacheDataset(data=val_files, transform=pre_transforms, cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    # val_ds = CacheDataset(data=data_list, transform=pre_transforms, cache_rate=1.0, num_workers=0)
    # val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)




    #--------------------------------------------------------------------------------
    # Model
    #--------------------------------------------------------------------------------
    print('Setting UNet')
    (model_unet, post_pred, post_label) = setupModel(param)
    model = model_unet.to(device)
    model.load_state_dict(torch.load(os.path.join(param.root_dir, model_file), map_location=device))

    #--------------------------------------------------------------------------------
    # Validate
    #--------------------------------------------------------------------------------
    print('Evaluate model')
    model.eval()
    



    with torch.no_grad():
        metric_sum = 0.0
        metric_count = 0

        for i, val_data in enumerate(val_loader):
            roi_size = param.window_size
            sw_batch_size = 4
            
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs = from_engine(["pred"])(val_data)
            #val_output_label = torch.argmax(val_outputs, d    run(param, output_path, files, model_file)im=1, keepdim=True)
            #saver.save_batch(val_output_label, val_data['image_meta_dict'])            


inference()