'''dataset read file'''

import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, fields, field
from typing import Optional, List, Dict, Any
from PIL import Image
import cv2
from jpeg4py import JPEG
import pandas as pd
import numpy as np
import os
import ast
import warnings
from datasets.chars import CHARS_DICT, CBL_LP_CLASS_DICT
from datasets.PreprocFns import PreprocFuns
from models.box_ops import validate_xyxy_bbox,box_xyxy_to_cxcywh


def build_str_delayed(LPs:torch.Tensor):
    '''autoregressive model need delayed LPs as tgt input
    LPs: [B,8] or [8]
    return LPs_delay: [B,8] or [8], the last char is ignored, the first char is set to 0
    '''
    LPs_delay = LPs.clone()
    LPs_delay[..., 1:] = LPs[..., :-1]
    LPs_delay[..., 0] = 0
    return LPs_delay

'''
函数的输出也要可重塑，且能被解释器检索。
'''
# 定义通用的数据集输出类（覆盖所有可能的 ablation 输出）

@dataclass
class BatchBox:
    """
        images[3,y,x],
        plateType[1],
        boxes[1,4], # in [0,1] format xyxy
        LPs[8],
        verteces[8], # in [-1, 1] 左上开始顺时针4个点
        theta[6],
    """
    # 基础必选字段（所有实验都需要）
    images: torch.Tensor
    LPs: Optional[torch.Tensor] = None 
    plateType: Optional[torch.Tensor] = None 
    bboxes: Optional[torch.Tensor] = None 
    # 可选字段（不同 ablation 实验按需赋值，默认 None）
    LPs_delay: Optional[torch.Tensor] = None  # 延迟标签（自回归模型needed）
    verteces: Optional[torch.Tensor] = None  # 顶点坐标（STN模型needed）
    theta: Optional[torch.Tensor] = None      # 变换矩阵（STN模型needed）
    # lr_LP_img: Optional[torch.Tensor] = None #LRLPR 不使用，直接装在images
    hr_LP_img: Optional[torch.Tensor] = None #LRLPR
    # 其他扩展字段
    meta_info: Optional[Dict[str, Any]] = None  # 元信息（如样本路径、标注类型）
    weights: Optional[List[float]] = None       # 样本权重（仅部分实验需要）
    extra: Dict[str, Any] = field(default_factory=dict)  # 兜底：临时扩展字段

    def mv_to_device(self, device, non_blocking=True):
        # mv all tensor fields to device
        # non_blocking=True: 异步 DMA（需要 DataLoader pin_memory=True）
        # 使用 fields(self) 迭代所有定义的字段
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(device, non_blocking=non_blocking))
        return
    def __repr__(self):
        res = [f"{self.__class__.__name__}:"]
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                # 只打印 shape
                content = f"Tensor(shape={list(value.shape)})"
            else:
                content = repr(value)
            res.append(f"    {field.name}: {content}")
        return "\n".join(res)
    pass



class LRLPR_base(Dataset):
    defaultOutImgSize = (128, 512)  # (H, W)
    
    def __init__(
        self,
        csvFile,
        lpr_max_len=8,
        preprocFun=None,
        shuffle=False,
        imgSize=defaultOutImgSize,
        img_align=True
    ):
        '''adj_img_align让lp和img对齐而不是和trace对齐'''
        self.df = pd.read_csv(csvFile)
        self.lrlpr_dir = os.path.dirname(csvFile)
        self.lpr_max_len = lpr_max_len
        self.img_size = imgSize
        self.img_align=img_align
        if img_align:
            os.environ['LRLPR_img_align'] = 'True'
        else:
            os.environ['LRLPR_img_align'] = 'False'
        
        # 获取列的位置
        keys = ["trace_id", "trace_dir", "plate_text"]
        self.col_indexes = [self.df.columns.get_loc(key) for key in keys]
        
        # shuffle self.anno_csv
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        
        # PreprocFun. default resize img only. define other fun if need data augment.
        self.PreprocFun = (
            PreprocFuns.resize_cv2(imgSize) if preprocFun is None else preprocFun(imgSize)
        )
        
        return
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        r"""
        返回一个trace下的所有图片对（lr和hr）
        lr_imgs: [5, 3, H, W]
        hr_imgs: [5, 3, H, W]
        LPs: [8]
        LPs_delay: [8]
        """
        trace_id, trace_dir, plate_text = self.df.iloc[index, self.col_indexes]
        
        # 读取lr和hr图像
        trace_full_path = os.path.join(self.lrlpr_dir, trace_dir)
        lr_imgs = self._read_image_sequence(trace_full_path, 'lr', 5)
        hr_imgs = self._read_image_sequence(trace_full_path, 'hr', 5)
        
        # 读取车牌号
        LPs = self.read_LPs(plate_text)
        LPs_delay = build_str_delayed(LPs)
        return (
            lr_imgs,
            hr_imgs,
            LPs,
            LPs_delay,
        )
    
    def _read_image_sequence(self, trace_path, prefix, num_images):
        """
        读取一个序列的图像（lr或hr）
        prefix: 'lr' 或 'hr'
        num_images: 要读取的图像数量
        返回: [num_images, 3, H, W]
        """
        imgs = []
        for i in range(1, num_images + 1):
            # 尝试png或jpg
            img_name_png = f"{prefix}-{i:03d}.png"
            img_name_jpg = f"{prefix}-{i:03d}.jpg"
            
            img_path_png = os.path.join(trace_path, img_name_png)
            img_path_jpg = os.path.join(trace_path, img_name_jpg)
            
            if os.path.exists(img_path_png):
                img = self._load_and_process_image(img_path_png)
            elif os.path.exists(img_path_jpg):
                img = self._load_and_process_image(img_path_jpg)
            else:
                raise FileNotFoundError(
                    f"Image not found: {img_path_png} or {img_path_jpg}"
                )
            
            imgs.append(img)
        
        return torch.stack(imgs, dim=0)
    
    def _load_and_process_image(self, filePath):
        """
        加载和处理单个图像
        返回: [3, H, W]
        """
        # 用 cv2 加载图像（BGR）
        img_bgr = cv2.imread(filePath)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {filePath}")
        
        # 转换为 RGB（cv2 默认是 BGR）
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 使用 PreprocFun 处理（包括resize）
        img_tensor = self.PreprocFun(image=img_rgb)['image']
        
        return img_tensor
    
    def read_LPs(self, license_plate, allow_warning=False):
        """
        转换车牌字符为索引
        license_plate: 车牌字符串
        返回: [lpr_max_len]
        """
        license_plate = license_plate.ljust(self.lpr_max_len, "-")
        try:
            LPs = torch.tensor([CHARS_DICT[c] for c in license_plate], dtype=torch.int64)
        except KeyError as e:
            import warnings
            
            warnings.warn(
                f"Character {e.args[0]} not found in CHARS_DICT. Assigning default value 0."
            ) if allow_warning else None
            LPs = torch.tensor(
                [CHARS_DICT.get(c, 0) for c in license_plate],
                dtype=torch.int64
            )
        
        return LPs
    
    @staticmethod
    def collate_fn(batch):
        """
        batch 中每个元素是 (lr_imgs, hr_imgs, LPs, LPs_delay)
        返回 BatchBox，extra 中包含 lr_lp_img 和 hr_lp_img
        """
        lr_imgs_list, hr_imgs_list, LPs_list, LPs_delay_list = zip(*batch)
        
        # 栈化为 [B, 5, 3, H, W] 的张量
        lr_imgs = torch.stack(lr_imgs_list, dim=0)
        hr_imgs = torch.stack(hr_imgs_list, dim=0)
        LPs = torch.stack(LPs_list, dim=0)
        LPs_delay = torch.stack(LPs_delay_list, dim=0)
        batchData=BatchBox(
            images=lr_imgs,  
            LPs=LPs,
            LPs_delay=LPs_delay,
            # lr_LP_img=lr_imgs,
            hr_LP_img=hr_imgs,
        )
        if os.environ.get('LRLPR_img_align', 'False').lower() == 'true':
            batchData = LRLPR_base.adj_img_align(batchData)
        return batchData
    @staticmethod
    def adj_img_align(batchData:BatchBox):
        B, T, C, H, W = batchData.images.shape

        # 1. 展开图片数据 [32, 5, 3, 128, 512] -> [160, 3, 128, 512]
        batchData.images = batchData.images.view(B * T, C, H, W)
        batchData.hr_LP_img = batchData.hr_LP_img.view(B * T, C, H, W)

        # 2. 对齐标签数据 [32, 8] -> [32, 5, 8] -> [160, 8]
        # 使用 unsqueeze + repeat 确保每个 trace 的标签被复制给对应的 5 张图
        batchData.LPs = batchData.LPs.unsqueeze(1).repeat(1, T, 1).view(B * T, -1)
        batchData.LPs_delay = batchData.LPs_delay.unsqueeze(1).repeat(1, T, 1).view(B * T, -1)
        return batchData

class CCPD_base(Dataset):
    r"""
    load CCPD img and tgt, return list:  
    imgs_tensor[3,y,x],
    labels[1],
    boxes[1,4],
    LPs[8]
    """
    ImgSize=(720,1160) # used to calcu relative pixel location

    @staticmethod
    def rescale(img_int, size):
        # 更改一下，先检查img的size是否等于目标size，若相等则不resize
        if img_int.size == size:
            return img_int
        return img_int.resize(size)

    def __init__(
        self,
        csvFile,
        lpr_max_len=8,
        preprocFun=None,
        shuffle=False,
        imgSize_car=(1160,720),
        imgSize_lp=(128, 512),
    ):
        self.df = pd.read_csv(csvFile)
        self.batch_name_space = ["imgs", "labels", "boxes", "LPs"]
        # 获取列的位置

        keys = [
            "filename",
            "CCPD_path",
            "license_plate",
            "bounding_box_1_x",
            "bounding_box_1_y",
            "bounding_box_2_x",
            "bounding_box_2_y",
            "vertex_1_x",
            "vertex_1_y",
            "vertex_2_x",
            "vertex_2_y",
            "vertex_3_x",
            "vertex_3_y",
            "vertex_4_x",
            "vertex_4_y",
        ]
        self.col_indexes = [self.df.columns.get_loc(key) for key in keys]
        # get dirpath of ccpd
        self.CCPD_dir = os.path.dirname(csvFile)
        # shuffle self.anno_csv
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.lp_max_len = lpr_max_len
        self.img_size_car = imgSize_car
        self.img_size_lp = imgSize_lp
        # PreprocFun. default resize_cv2_A to manage bbox and keypoints transforms.
        self.PreprocFun = (
            PreprocFuns.resize_cv2_A(imgSize_car) if preprocFun is None else preprocFun(imgSize_car)
        )
        self.PreprocFun_lp = PreprocFuns.resize_cv2(imgSize_lp)

        return

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        r"""
        imgs_tensor[3,y,x],
        labels[1],
        boxes[1,4],
        LPs[8],
        LPs_delay[8],
        verteces[8],
        hr_lp_img[3,128,512]
        """
        (
            filename,
            CCPD_path,
            license_plate,
            bbox1x,
            bbox1y,
            bbox2x,
            bbox2y,
            # CCPD 顶点标注从右下开始,牛逼
            v3x,
            v3y,
            v4x,
            v4y,
            v1x,
            v1y,
            v2x,
            v2y,
        ) = self.df.iloc[index, self.col_indexes]
        
        # Prepare keypoints for A package (4 vertices in original image coordinate)
        # keypoints format: [(x, y), ...] in pixel coordinates
        keypoints = [
            (v1x, v1y),
            (v2x, v2y),
            (v3x, v3y),
            (v4x, v4y),
        ]
        
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"
        image = JPEG(filePath).decode()

        # crop hr lp img
        pts_src = np.float32(keypoints)
        pts_dst = np.float32([[512, 128], [0, 128], [0, 0], [512, 0]])
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        lp_img = cv2.warpPerspective(image, M, (512, 128))
        hr_lp_img = self.PreprocFun_lp(image=lp_img)['image']
        
        result = self.PreprocFun(image=image, keypoints=keypoints)
        imgs_tensor = result['image']
        keypoints_transformed = result['keypoints']
        
        LPs = self.read_LPs(license_plate)
        LPs_delay = build_str_delayed(LPs)
        labels = torch.tensor(1, dtype=torch.int)
        
        # Generate bbox and vertices from transformed keypoints
        # keypoints_transformed: list of (x, y) tuples
        xs = [kp[0] for kp in keypoints_transformed]
        ys = [kp[1] for kp in keypoints_transformed]
        
        # Find min/max for bounding box (in normalized coordinates)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Normalize to [0, 1]
        img_h, img_w = self.img_size_car[0], self.img_size_car[1]
        boxes = torch.tensor(
            [min_x / img_w, min_y / img_h, max_x / img_w, max_y / img_h],
            dtype=torch.float32,
        )
        _, boxes = validate_xyxy_bbox(boxes)
        
        # 将顶点转换到 [-1, 1] 范围
        verteces = torch.tensor([
            [kp[0] / img_w * 2.0 - 1.0, kp[1] / img_h * 2.0 - 1.0] 
            for kp in keypoints_transformed
        ], dtype=torch.float32).view(-1)  # Flatten to [8]
        
        return (
            imgs_tensor,
            labels,
            boxes,
            LPs,
            LPs_delay,
            verteces,
            hr_lp_img,
        )
    @staticmethod
    def collate_fn(batch):
        imgs,labels,boxes,LPs,LPs_delay,vertecies,hr_lp_imgs=[torch.stack(tensor,0) for tensor in zip(*batch)]
        return BatchBox(images=imgs,plateType=labels,bboxes=boxes,LPs=LPs,LPs_delay=LPs_delay,verteces=vertecies,hr_LP_img=hr_lp_imgs)
        # return{"imgs":imgs,"labels":labels,'boxes':boxes,'LPs':LPs} # dict output

    @staticmethod
    def gen_bbox(bbox1x, bbox1y, bbox2x, bbox2y):
        boxes = torch.tensor(
            (
                bbox1x / CCPD_base.ImgSize[0],
                bbox1y / CCPD_base.ImgSize[1],
                bbox2x / CCPD_base.ImgSize[0],
                bbox2y / CCPD_base.ImgSize[1],
            ),
            dtype=torch.float32,
        )
        _, boxes = validate_xyxy_bbox(boxes)
        return boxes
    @staticmethod
    def gen_verteces(v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y):
        # 1. 首先计算 [0, 1] 范围内的归一化坐标
        verteces = torch.tensor((
            v1x/CCPD_base.ImgSize[0], v1y/CCPD_base.ImgSize[1],
            v2x/CCPD_base.ImgSize[0], v2y/CCPD_base.ImgSize[1],
            v3x/CCPD_base.ImgSize[0], v3y/CCPD_base.ImgSize[1],
            v4x/CCPD_base.ImgSize[0], v4y/CCPD_base.ImgSize[1],
        ), dtype=torch.float32)
        
        # 2. 映射到 [-1, 1] 范围
        # 公式： new = old * 2 - 1
        verteces_rescaled = verteces * 2.0 - 1.0
        
        return verteces_rescaled

    def read_LPs(self, license_plate, allow_worning=False):
        license_plate = license_plate.ljust(
            self.lp_max_len, "-"
        )  # license_plate.len = 7 or 8
        try:
            LPs = torch.tensor([CHARS_DICT[c] for c in license_plate])
        except KeyError as e:
            import warnings

            warnings.warn(
                f"Character {e.args[0]} not found in CHARS_DICT. Assigning default value 0."
            ) if allow_worning else None
            LPs = torch.tensor([CHARS_DICT.get(c, 0) for c in license_plate])
            pass
        return LPs

    def read_imgs_tensor_PIL(self, filename, CCPD_path):
        '''use torch preprocess'''
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"  # os.path.join(self.CCPD_dir, CCPD_path, filename)
        img_int = Image.open(filePath)
        imgs_tensor = self.PreprocFun(img_int)
        return imgs_tensor
    def read_imgs_tensor_cv2(self, filename, CCPD_path):
        '''use ablumentations preprocess'''
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"
        # image = JPEG('your_image.jpg').decode()
        # 用 cv2 加载图像（BGR）
        img_bgr = cv2.imread(filePath)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {filePath}")

        # 转换为 RGB（cv2 默认是 BGR）
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # # 如果你的 PreprocFun 是基于 PIL 的，这里要转回 Image
        # img_rgb_pil = Image.fromarray(img_rgb)

        imgs_tensor = self.PreprocFun(image=img_rgb)['image']
        return imgs_tensor
    def read_imgs_tensor_jpeg4py(self, filename, CCPD_path):
        '''jpeg4py load, use ablumentations preprocess'''
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"
        image = JPEG(filePath).decode()

        imgs_tensor = self.PreprocFun(image=image)['image']
        return imgs_tensor
    
    def read_imgs_tensor_with_keypoints(self, filename, CCPD_path, keypoints):
        """
        Load image and apply preprocessing with keypoint transformation.
        keypoints: list of (x, y) tuples in original image coordinates
        Returns: (imgs_tensor, keypoints_transformed)
        """
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"
        image = JPEG(filePath).decode()
        
        # Apply preprocessing with keypoint transformation
        result = self.PreprocFun(image=image, keypoints=keypoints)
        imgs_tensor = result['image']
        keypoints_transformed = result['keypoints']  # list of (x, y) in resized coordinates
        
        return imgs_tensor, keypoints_transformed

    pass

class CTPFSD_base(Dataset):
    '''
    load CTPFSD img and tgt.
    compatible with CCPD_base
    '''
    defaltOutImgSize_car = (540, 960)
    # (1080, 1920)  # y,x
    # (540, 960)  # y,x (H, W)
    defaltOutImgSize_lp = (128, 512)  # y,x

    def __init__(
        self,
        csvFile,
        lpr_max_len=8,
        preprocFun=None,
        shuffle=False,
        imgSize_car=defaltOutImgSize_car,
        imgSize_lp=defaltOutImgSize_lp
    ):
        self.df = pd.read_csv(csvFile)
        self.root_dir = os.path.dirname(csvFile)
        
        keys = [
            "license_plate",
            "detect_image_path",
            "recognize_image_path",
            "vertices_xy",
        ]
        self.col_indexes = [self.df.columns.get_loc(key) for key in keys]

        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.lp_max_len = lpr_max_len
        self.img_size_car = imgSize_car # (H, W)
        self.img_size_lp = imgSize_lp

        # PreprocFun for car image (with keypoints)
        self.PreprocFun_car = (
            PreprocFuns.resize_cv2_A(imgSize_car) if preprocFun is None else preprocFun(imgSize_car)
        )
        # PreprocFun for LP image (simple resize)
        self.PreprocFun_lp = PreprocFuns.resize_cv2(imgSize_lp)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        license_plate, detect_path, recognize_path, vertices_str = self.df.iloc[index, self.col_indexes]

        # Parse vertices
        try:
            keypoints = ast.literal_eval(vertices_str)
        except:
            keypoints = []
        
        # Read car image and transform
        imgs_tensor, keypoints_transformed = self.read_imgs_tensor_with_keypoints(
            detect_path, keypoints
        )

        # Read HR LP image
        hr_lp_img = self.read_imgs_tensor_lp(recognize_path)

        # LPs
        LPs = self.read_LPs(license_plate)
        LPs_delay = build_str_delayed(LPs)
        labels = torch.tensor(1, dtype=torch.int)

        # Generate bbox and vertices from transformed keypoints
        if len(keypoints_transformed) > 0:
            xs = [kp[0] for kp in keypoints_transformed]
            ys = [kp[1] for kp in keypoints_transformed]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
        else:
            min_x, max_x, min_y, max_y = 0, 0, 0, 0

        img_h, img_w = self.img_size_car[0], self.img_size_car[1]
        
        # Normalize bbox to [0, 1]
        boxes = torch.tensor(
            [min_x / img_w, min_y / img_h, max_x / img_w, max_y / img_h],
            dtype=torch.float32,
        )
        _, boxes = validate_xyxy_bbox(boxes)

        # Normalize vertices to [-1, 1]
        verteces_list = []
        for kp in keypoints_transformed:
             verteces_list.append(kp[0] / img_w * 2.0 - 1.0)
             verteces_list.append(kp[1] / img_h * 2.0 - 1.0)
             
        verteces = torch.tensor(verteces_list, dtype=torch.float32).view(-1)

        return (
            imgs_tensor,
            labels,
            boxes,
            LPs,
            LPs_delay,
            verteces,
            hr_lp_img,
        )

    def read_imgs_tensor_with_keypoints(self, img_rel_path, keypoints):
        filePath = os.path.join(self.root_dir, img_rel_path)
        img_bgr = cv2.imread(filePath)
        if img_bgr is None:
             raise FileNotFoundError(f"Image not found: {filePath}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self.PreprocFun_car(image=img_rgb, keypoints=keypoints)
        return result['image'], result['keypoints']

    def read_imgs_tensor_lp(self, img_rel_path):
        filePath = os.path.join(self.root_dir, img_rel_path)
        img_bgr = cv2.imread(filePath)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {filePath}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.PreprocFun_lp(image=img_rgb)['image']

    def read_LPs(self, license_plate):
        license_plate = license_plate.ljust(self.lp_max_len, "-")
        try:
            LPs = torch.tensor([CHARS_DICT[c] for c in license_plate])
        except KeyError:
            LPs = torch.tensor([CHARS_DICT.get(c, 0) for c in license_plate])
        return LPs

    @staticmethod
    def collate_fn(batch):
        imgs, labels, boxes, LPs, LPs_delay, vertecies, hr_lp_imgs = [
            torch.stack(tensor, 0) for tensor in zip(*batch)
        ]
        return BatchBox(
            images=imgs,
            plateType=labels,
            bboxes=boxes,
            LPs=LPs,
            LPs_delay=LPs_delay,
            verteces=vertecies,
            hr_LP_img=hr_lp_imgs,
        )

class CTPFSD_bbox(CTPFSD_base):
    # 返回bbox only here
    @staticmethod
    def collate_fn(batch):
        imgs, labels, boxes, LPs, LPs_delay, vertecies, hr_lp_imgs = [
            torch.stack(tensor, 0) for tensor in zip(*batch)
        ]
        return BatchBox(
            images=imgs,
            plateType=labels,
            bboxes=boxes,
            LPs=LPs,
            LPs_delay=LPs_delay,
            hr_LP_img=hr_lp_imgs,
        )

    pass
class CTPFSD_theta(CTPFSD_base):
    # 返回theta以供监督here
    def __getitem__(self, index):
        # 1. 直接调用父类的 __getitem__ 获取所有已经处理好的基础数据
        base_outputs = super().__getitem__(index)
        
        # 按照父类的返回顺序解包
        imgs_tensor, labels, boxes, LPs, LPs_delay, verteces, hr_lp_img = base_outputs
        
        # 2. 从 verteces 计算 6 DoF 仿射变换矩阵 theta
        # 确保 verteces 包含了有效的 4 个顶点 (8个坐标值)
        if verteces.numel() == 8:
            # 还原为 4x2 矩阵以便于提取
            pts = verteces.view(4, 2)
            
            # 顺序：左上(TL), 右上(TR), 右下(BR), 左下(BL)
            x_tl, y_tl = pts[0, 0], pts[0, 1]
            x_tr, y_tr = pts[1, 0], pts[1, 1]
            x_br, y_br = pts[2, 0], pts[2, 1]
            x_bl, y_bl = pts[3, 0], pts[3, 1]
            
            # 使用解析解直接计算 theta [2, 3]
            theta = torch.tensor([
                [
                    0.25 * (-x_tl + x_tr + x_br - x_bl), # theta_11
                    0.25 * (-x_tl - x_tr + x_br + x_bl), # theta_12
                    0.25 * ( x_tl + x_tr + x_br + x_bl)  # theta_13 (X轴平移)
                ],
                [
                    0.25 * (-y_tl + y_tr + y_br - y_bl), # theta_21
                    0.25 * (-y_tl - y_tr + y_br + y_bl), # theta_22
                    0.25 * ( y_tl + y_tr + y_br + y_bl)  # theta_23 (Y轴平移)
                ]
            ], dtype=torch.float32)
        else:
            # 异常/无关键点 fallback: 返回单位矩阵 (Identity Matrix)，表示不进行任何形变
            theta = torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ], dtype=torch.float32)

        # 3. 将计算好的 theta 拼接到返回的 tuple 中
        return (
            imgs_tensor,
            labels,
            boxes,
            LPs,
            LPs_delay,
            verteces,
            hr_lp_img,
            theta,  # <--- 你的新特性监督信号
        )
    @staticmethod
    def collate_fn(batch):
        imgs, labels, boxes, LPs, LPs_delay, vertecies, hr_lp_imgs,theta = [
            torch.stack(tensor, 0) for tensor in zip(*batch)
        ]
        return BatchBox(
            images=imgs,
            plateType=labels,
            # bboxes=boxes,
            LPs=LPs,
            LPs_delay=LPs_delay,
            hr_LP_img=hr_lp_imgs,
            theta=theta
        )
    pass

class Dataset_rand(Dataset):
    r'''
    random data for model test
    length: dataset length
    imgSize: (W,H)
    '''

    def __init__(self, *args, length=10,imgSize=(720,1160), **kwargs):
        self.length = length
        self.imgSize = imgSize
        self.batch_name_space = ["imgs", "labels", "boxes", "LPs"]
        super().__init__()
        return

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img, tgt = self.batch_rand(batchSize=1, imgSize=self.imgSize)
        return (
            img.squeeze(0),
            tgt["plateType"].squeeze(0),
            tgt["boxes"].squeeze(0),
            tgt["LPs"].squeeze(0),
            tgt["LPs_delay"].squeeze(0),
            tgt["verteces"].squeeze(0),
            tgt["theta"].squeeze(0),
        )

    defalt_imgSize = (720, 1160)

    @staticmethod
    def img_randn(B:int, C:int=3, W:int=defalt_imgSize[0], H:int=defalt_imgSize[1]):
        img = torch.randn(B, C, W, H)
        return img

    @staticmethod
    def gen_label(B: int):
        '''0: nothing;1: vehical'''
        labels = torch.ones((B, 1), dtype=int)
        return labels

    @staticmethod
    def bbox_rand(B: int, n_points: int = 4):
        bbox = torch.rand(B, 2 * n_points)
        return bbox

    @staticmethod
    def category_rand(B: int, nClass: int):
        l_class = torch.randint(0, nClass, (B, 1))
        return l_class

    @staticmethod
    def LP_rand(B: int, nChars: int, N: int = 8):
        """
        nChars: num of chars in diction
        N: LP length
        """
        lp = torch.randint(0, nChars, (B, N))
        return lp

    @staticmethod
    def batch_rand(batchSize=2,imgSize=defalt_imgSize):
        """return imgs and targets for LPD model forward"""
        imgs = Dataset_rand.img_randn(B=batchSize, W=imgSize[0], H=imgSize[1])
        labels = Dataset_rand.gen_label(batchSize)
        bboxes = Dataset_rand.bbox_rand(batchSize, 2)
        LPs = Dataset_rand.LP_rand(batchSize, nChars=75, N=8)
        LPs_delay = build_str_delayed(LPs)
        verteces = torch.rand(batchSize, 8)
        theta = torch.rand(batchSize, 6)
        targets = {"plateType": labels, "boxes": bboxes, "LPs": LPs, "LPs_delay": LPs_delay, "verteces": verteces, "theta": theta}
        # targets=[{"labels":labels[i],"boxes":bboxes[i],"LP":LPs[i]} for i in range(batchSize)]
        return imgs, targets

    

    @staticmethod
    def collate_fn(batch):
        imgs, labels, boxes, LPs, LPs_delay, verteces, theta = [torch.stack(tensor, 0) for tensor in zip(*batch)]
        return BatchBox(images=imgs, plateType=labels, bboxes=boxes, LPs=LPs, LPs_delay=LPs_delay, verteces=verteces, theta=theta)

    pass


'''
assembling functions. here are funcs that combine datasets with different augment or process methods.
'''
def CCPD_strong_augmented(root, transform=None):
    # Return a CCPD dataset instance with strong augmentations applied
    return


def _worker_init_fn(worker_id):
    """每个 DataLoader worker 启动时调用。
    cv2 默认会在每个 worker 内部再开多线程，与其他 worker 抢核，
    限制为单线程后各 worker 互不干扰，整体吞吐反而更高。
    """
    import cv2
    cv2.setNumThreads(1)


# def dataset2loader(
#     dataset: Dataset,
#     batch_size=16,
#     shuffle=True,
#     num_workers=8,
#     collate_fn=None,
#     pin_memory=False,
# ):
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         collate_fn=collate_fn if collate_fn is not None else dataset.collate_fn,
#         pin_memory=pin_memory,
#         worker_init_fn=_worker_init_fn,
#         persistent_workers=num_workers > 0,
#     )
