# CREATE NEW CUSTOM ITK OBJECT LOADER
import SimpleITK as sitk
import numpy as np
import torch
from monai.data import MetaTensor, ImageReader
from monai.data.utils import orientation_ras_lps, is_no_channel
from monai.config import DtypeLike
from monai.utils import convert_to_dst_type, ensure_tuple, MetaKeys, SpaceKeys, TraceKeys
from monai.utils import ImageMetaKey as Key
from torch.utils.data._utils.collate import np_str_obj_array_pattern
from monai.transforms import Transform

class sitkReader():
    def __init__(
            self,
            series_name: str = "",
            reverse_indexing: bool = False,
            series_meta: bool = False,
            affine_lps_to_ras: bool = True,
            **kwargs,
        ):
        super().__init__()
        self.kwargs = kwargs
        self.series_name = series_name
        self.reverse_indexing = reverse_indexing
        self.series_meta = series_meta
        self.affine_lps_to_ras = affine_lps_to_ras
    
    def get_data(self, img) -> tuple[np.ndarray, dict]:
        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}
        data = self._get_array_data(img)
        img_array.append(data)
        header = self._get_meta_dict(img)
        header[MetaKeys.ORIGINAL_AFFINE] = self._get_affine(img, self.affine_lps_to_ras)
        header[MetaKeys.SPACE] = SpaceKeys.RAS if self.affine_lps_to_ras else SpaceKeys.LPS
        header[MetaKeys.AFFINE] = header[MetaKeys.ORIGINAL_AFFINE].copy()
        header[MetaKeys.SPATIAL_SHAPE] = self._get_spatial_shape(img)
        # default to "no_channel" or -1
        header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (float("nan") if len(data.shape) == len(header[MetaKeys.SPATIAL_SHAPE]) else -1)
        self._copy_compatible_dict(header, compatible_meta)
        return self._stack_images(img_array, compatible_meta), compatible_meta
        
    def _get_meta_dict(self, img) -> dict:
        img_meta_dict = img.GetMetaDataKeys()
        meta_dict: dict = {}
        for key in img_meta_dict:
            if key.startswith("ITK_"):
                continue
            val = img.GetMetaData(key)
            meta_dict[key] = np.asarray(val) if type(val).__name__.startswith("itk") else val
        meta_dict["spacing"] = np.asarray(img.GetSpacing())
        return dict(meta_dict)

    def _get_affine(self, img, lps_to_ras: bool = True):
        dir_array = img.GetDirection()
        direction = np.array([dir_array[0:3],dir_array[3:6],dir_array[6:9]])
        spacing = np.asarray(img.GetSpacing())
        origin = np.asarray(img.GetOrigin())
        sr = min(max(direction.shape[0], 1), 3)
        affine: np.ndarray = np.eye(sr + 1)
        affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
        affine[:sr, -1] = origin[:sr]
        if lps_to_ras:
            affine = orientation_ras_lps(affine)
        return affine

    def _get_spatial_shape(self, img):
        ## Not handling multichannel images with SimpleITK
        dir_array = img.GetDirection()
        sr = np.array([dir_array[0:3],dir_array[3:6],dir_array[6:9]]).shape[0]
        sr = max(min(sr, 3), 1)
        _size = list(img.GetSize())
        return np.asarray(_size[:sr])

    def _get_array_data(self, img):
        ## Not handling multichannel images with SimpleITK
        np_img = sitk.GetArrayFromImage(img)
        return np_img if self.reverse_indexing else np_img.T
    
    def _stack_images(self, image_list: list, meta_dict: dict):
        if len(image_list) <= 1:
            return image_list[0]
        if not is_no_channel(meta_dict.get(MetaKeys.ORIGINAL_CHANNEL_DIM, None)):
            channel_dim = int(meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM])
            return np.concatenate(image_list, axis=channel_dim)
        # stack at a new first dim as the channel dim, if `'original_channel_dim'` is unspecified
        meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM] = 0
        return np.stack(image_list, axis=0)

    def _copy_compatible_dict(self, from_dict: dict, to_dict: dict):
        if not isinstance(to_dict, dict):
            raise ValueError(f"to_dict must be a Dict, got {type(to_dict)}.")
        if not to_dict:
            for key in from_dict:
                datum = from_dict[key]
                if isinstance(datum, np.ndarray) and np_str_obj_array_pattern.search(datum.dtype.str) is not None:
                    Object
                to_dict[key] = str(TraceKeys.NONE) if datum is None else datum  # NoneType to string for default_collate
        else:
            affine_key, shape_key = MetaKeys.AFFINE, MetaKeys.SPATIAL_SHAPE
            if affine_key in from_dict and not np.allclose(from_dict[affine_key], to_dict[affine_key]):
                raise RuntimeError(
                    "affine matrix of all images should be the same for channel-wise concatenation. "
                    f"Got {from_dict[affine_key]} and {to_dict[affine_key]}."
                )
            if shape_key in from_dict and not np.allclose(from_dict[shape_key], to_dict[shape_key]):
                raise RuntimeError(
                    "spatial_shape of all images should be the same for channel-wise concatenation. "
                    f"Got {from_dict[shape_key]} and {to_dict[shape_key]}."
            )
                
class LoadSitkImage(Transform):
    def __init__(self,
            image_only: bool = False,
            dtype: DtypeLike or None = np.float32,
            ensure_channel_first: bool = False,
            simple_keys: bool = False,
            prune_meta_pattern: str or None = None,
            prune_meta_sep: str = ".",
            expanduser: bool = True,         
        ) -> None:
        self.reader = sitkReader()
        self.image_only = image_only
        self.ensure_channel_first = ensure_channel_first
        self.dtype = dtype
        self.simple_keys = simple_keys
        self.pattern = prune_meta_pattern
        self.sep = prune_meta_sep
        self.expanduser = expanduser

    def __call__(self, img):
        if not isinstance(img, sitk.SimpleITK.Image):
            raise RuntimeError(f"{self.__class__.__name__} The input image is not an ITK object.\n")    
        img_array, meta_data = self.reader.get_data(img)
        img_array = convert_to_dst_type(img_array, dst=img_array, dtype=self.dtype)[0]
        if not isinstance(meta_data, dict):
            raise ValueError(f"`meta_data` must be a dict, got type {type(meta_data)}.")
        # Here I changed from original LoadImage to use tensor instead of numpy array (img_array) 
        # so the result is similar to loading the nifti file with LoadImage
        img = MetaTensor.ensure_torch_and_prune_meta(
            torch.from_numpy(img_array), meta_data, self.simple_keys, pattern=self.pattern, sep=self.sep
        )
        if self.ensure_channel_first:
            img = EnsureChannelFirst()(img)
        if self.image_only:
            return img
        return img, img.meta if isinstance(img, MetaTensor) else meta_data