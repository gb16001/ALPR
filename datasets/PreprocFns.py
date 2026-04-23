import albumentations as A
from torchvision import transforms

class PreprocFuns:
    @staticmethod
    def resize(imgSize):
        '''imgSize:(y,x)
        '''
        return transforms.Compose(
                [
                    transforms.Resize(imgSize),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化图像
                ]
            )
    @staticmethod
    def resize_cv2(img_size):
        """img_size: (height, width)"""
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=0, std=1),
            A.pytorch.ToTensorV2(),
        ])

    @staticmethod
    def resize_jit_norm_cv2(img_size):
        """img_size: (height, width)"""
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            
            # 1. 先进行空间和颜色增强
            A.ColorJitter(p=0.5),
            A.ToGray(p=0.05), 
            
            # 2. 最后进行像素归一化
            # 这里的 mean 和 std 通常建议使用 ImageNet 的标准值，或者你数据集的统计值
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            
            # 3. 转为 PyTorch Tensor
            A.pytorch.ToTensorV2(),
        ])
    @staticmethod
    def resize_cv2_A(img_size):
        """img_size: (height, width)"""
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=0, std=1),
                A.pytorch.ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    @staticmethod
    def resize_cv2_norm(img_size):
        """img_size: (height, width)
        norm as imgnet standard"""
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=mean, std=std),
                A.pytorch.ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    @staticmethod
    def img_Resize_jit_affine_norm_A(imgSize):
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        strong_augment = A.Compose(
            [
                A.Resize(height=imgSize[0], width=imgSize[1]),
                A.ColorJitter(),
                A.ToGray(p=0.01),
                A.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-30, 30),
                    shear=(-30, 30),
                    translate_percent=(-0.3, 0.3),
                    p=0.5,
                ),
                A.Normalize(
                    mean=mean,  
                    std=std,
                ),
                A.pytorch.ToTensorV2(),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        return strong_augment
