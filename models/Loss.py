from typing import Optional
from dataclasses import dataclass, field
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment
from pytorch_msssim import SSIM
# torchvision generalized_box_iou_loss 内含 boolean-mask scatter (nonzero)，inductor 无法编译，
# 改用自实现的纯算子版本
from .box_ops import giou_loss as GIoU_loss, generalized_box_iou_compilable

from .box_ops import box_cxcywh_to_xyxy
import models.fullModel
import datasets
from models.fullModel import InferBox
from datasets import BatchBox


@dataclass
class LossBox:
    total_loss: Optional[torch.Tensor]= None
    main_loss: Optional[torch.Tensor]= None       # string CE (exDETR infer matched)
    denoise_loss: Optional[torch.Tensor]= None    # CDN denoise total (exDETR)
    prior_loss: Optional[torch.Tensor]= None
    verteces_loss: Optional[torch.Tensor]= None
    bbox_loss: Optional[torch.Tensor]= None
    img_recon_loss: Optional[torch.Tensor]= None
    theta_loss: Optional[torch.Tensor]= None
    bbox_cls_loss: Optional[torch.Tensor]= None   # LP/void class CE (exDETR infer)
    pass
@dataclass
class EvalBox:
    LP_total: int = 0
    LP_correct: int = 0
    LP_acc: float = 0.0
    LP_acc_denoise: Optional[float] = None
    LP_acc_prior: Optional[float] = None
    LP_err: Optional[float] = None
    char_acc: Optional[float] = None
    char_without_first_acc: Optional[float] = None
    recon_mse: Optional[float] = None
    recon_psnr: Optional[float] = None
    verteces_mse: Optional[float] = None
    verteces_nme: Optional[float] = None
    bbox_iou: Optional[float] = None
    bbox_ap70: Optional[float] = None
    bbox_ap50: Optional[float] = None
    pass

def mpdiou_loss(pred_bboxes: torch.Tensor, tgt_bboxes: torch.Tensor) -> torch.Tensor:
    """
    独立且可复用的 MPDIoU 损失计算函数。
    
    参数:
        pred_bboxes: 预测的边界框, 形状为 [B, 4], 格式为 (x1, y1, x2, y2)
        tgt_bboxes: 真实的边界框, 形状为 [B, 4], 格式为 (x1, y1, x2, y2)
    
    注意:
        当前计算逻辑中，由于除以 2.0 作为规范化因子，
        默认输入坐标已被归一化至 [0, 1] 的尺度 (对角线长度平方 w^2 + h^2 = 1^2 + 1^2 = 2)。
    """
    # 确保预测框不越界或者造成计算异常
    pred_bboxes = pred_bboxes.clamp(min=0.0, max=1.0)
    
    lt = torch.max(pred_bboxes[:, :2], tgt_bboxes[:, :2])
    rb = torch.min(pred_bboxes[:, 2:], tgt_bboxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    
    area_pred = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
    area_tgt = (tgt_bboxes[:, 2] - tgt_bboxes[:, 0]) * (tgt_bboxes[:, 3] - tgt_bboxes[:, 1])
    union = area_pred + area_tgt - inter
    iou = inter / (union + 1e-6)
    
    # 左上角距离平方和与右下角距离平方和
    d1_2 = (pred_bboxes[:, 0] - tgt_bboxes[:, 0])**2 + (pred_bboxes[:, 1] - tgt_bboxes[:, 1])**2
    d2_2 = (pred_bboxes[:, 2] - tgt_bboxes[:, 2])**2 + (pred_bboxes[:, 3] - tgt_bboxes[:, 3])**2
    
    # 计算MPDIoU损失
    loss_mpdiou = (1 - iou + d1_2 / 2.0 + d2_2 / 2.0).mean()
    return loss_mpdiou


class Poly1Loss(nn.Module):
    def __init__(self, epsilon=2.0, reduction='mean', weight=None, ignore_index=None):
        """
        PolyLoss-1 (ICLR 2022) 兼容版实现
        
        Args:
            epsilon: 调节因子 (默认 2.0)
            reduction: 'mean', 'sum', 'none'
            weight: 类别权重 Tensor
            ignore_index: 忽略的类别索引 (默认 -100, 常见于序列 Padding)
        """
        super(Poly1Loss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        # self.register_buffer('weight', weight) # 注册为 buffer 以便自动处理 device
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        Args:
            logits: [Batch, Num_Classes, ...] 
                    例如: (B, C), (B, C, N) 序列, (B, C, H, W) 图像
            target: [Batch, ...] 
                    例如: (B), (B, N) 序列, (B, H, W) 图像
        """
        # 1. 计算基础 CrossEntropy (reduction='none' 也就是保留维度)
        # F.cross_entropy 内部已经处理了 softmax 和 ignore_index 的逻辑
        ce_loss = F.cross_entropy(
            logits, target, 
            weight=None, 
            ignore_index=self.ignore_index if self.ignore_index is not None else -100, 
            reduction='none'
        )

        # 2. 计算 Pt (预测为真实类别的概率)
        pt = F.softmax(logits, dim=1)

        # target: [B, N] -> [B, 1, N]
        target_unsqueezed = target.unsqueeze(1)
        
        # 处理 ignore_index 防止 gather 越界 crash
        # 如果 target 中含有 -100，gather 会报错，所以临时替换成 0
        # (后续我们会用 mask 把这些位置的 loss 设为 0)
        if self.ignore_index is not None:
            # 创建 mask: 标记哪些位置是有效的
            valid_mask = target != self.ignore_index
            # 只有有效的位置才进行 gather，无效位置填 0 (或者任何合法 index)
            target_for_gather = target_unsqueezed.clone()
            target_for_gather[target_for_gather == self.ignore_index] = 0
        else:
            valid_mask = None
            target_for_gather = target_unsqueezed

        # 获取 target 对应的概率值
        # gather 输入: [B, C, N], index: [B, 1, N] -> 输出: [B, 1, N]
        p_true = pt.gather(1, target_for_gather).squeeze(1)
        # --- 关键修改结束 ---

        # 3. 计算 Poly1 调节项
        poly1 = self.epsilon * (1 - p_true)

        # 如果有 ignore_index，需要把 Poly 项对应位置清零
        if valid_mask is not None:
            poly1 = poly1 * valid_mask.float()

        # 4. 最终 Loss
        loss = ce_loss + poly1

        # 5. Reduction
        if self.reduction == 'mean':
            # 标准 CrossEntropyLoss 的 mean 是：总 Loss / 有效像素数(token数)
            if valid_mask is not None:
                return loss.sum() / valid_mask.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CELoss(nn.Module):
    '''support wraped IO, wrap nn.CrossEntropyLoss '''
    def __init__(self, args=None):
        weight=getattr(args,'weight',None)
        reduction=getattr(args,'reduction','mean')
        super().__init__()
        self.CEloss=nn.CrossEntropyLoss(weight=weight,reduction=reduction)
        return
    def forward(self,inputs:models.fullModel.InferBox,targets:datasets.BatchBox,*args, **kwargs)->LossBox:
        '''
        inputs: [B,C,d_1,...]
        targets: [B,d_1,...] long
        '''
        LP_tgt=targets.LPs
        loss=self.CEloss.forward(inputs.LPs_logits,LP_tgt)
        # return loss
        return LossBox(total_loss=loss)
    pass

class CE_3task(nn.Module):
    '''给 main, denoise, prior 三个任务都计算 CE loss'''
    def __init__(self, args=None):
        weight=getattr(args,'weight',None)
        reduction=getattr(args,'reduction','mean')
        self.k_main=getattr(args,'lambda_main',0.7)
        self.k_denoise=getattr(args,'lambda_denoise',0.2)
        self.k_prior=getattr(args,'lambda_prior',0.1)

        super().__init__()
        self.CEloss=nn.CrossEntropyLoss(weight=weight,reduction=reduction)
        return
    def forward(self,inputs:models.fullModel.InferBox,targets:datasets.BatchBox,*args, **kwargs)->LossBox:
        '''
        inputs: InferBox
        targets: BatchBox
        '''
        LP_tgt=targets.LPs
        loss_main=self.CEloss.forward(inputs.LPs_logits,LP_tgt)
        loss_denoise=self.CEloss.forward(inputs.denoise_LPs_logits,LP_tgt)
        loss_prior=self.CEloss.forward(inputs.prior_LPs_logits,LP_tgt)
        loss=self.k_main*loss_main+self.k_denoise*loss_denoise+self.k_prior*loss_prior
        # return loss
        return LossBox(total_loss=loss, main_loss=loss_main, denoise_loss=loss_denoise, prior_loss=loss_prior)
    pass

class CE_3task_vertecesMSE(nn.Module):
    def __init__(self, args=None):
        weight=getattr(args,'weight',None)
        reduction=getattr(args,'reduction','mean')
        self.k_main=getattr(args,'lambda_main',0.6)
        self.k_denoise=getattr(args,'lambda_denoise',0.2)
        self.k_prior=getattr(args,'lambda_prior',0.1)
        self.k_verteces=getattr(args,'lambda_verteces',1)

        super().__init__()
        self.CEloss=nn.CrossEntropyLoss(weight=weight,reduction=reduction)
        self.MSEloss=nn.MSELoss(reduction='mean')
        return

    def forward(
        self,
        inputs: models.fullModel.InferBox,
        targets: datasets.BatchBox,
        *args,
        **kwargs
    ) -> LossBox:
        """
        inputs: InferBox
        targets: BatchBox
        """
        LP_tgt = targets.LPs
        loss_main = self.CEloss.forward(inputs.LPs_logits, LP_tgt)
        loss_denoise = self.CEloss.forward(inputs.denoise_LPs_logits, LP_tgt)
        loss_prior = self.CEloss.forward(inputs.prior_LPs_logits, LP_tgt)
        loss_verteces = self.MSEloss.forward(inputs.STN_verteces.flatten(1), targets.verteces)
        loss = (
            self.k_main * loss_main
            + self.k_denoise * loss_denoise
            + self.k_prior * loss_prior
            + self.k_verteces * loss_verteces
        )
        # return loss
        return LossBox(
            total_loss=loss,
            main_loss=loss_main,
            denoise_loss=loss_denoise,
            prior_loss=loss_prior,
            verteces_loss=loss_verteces,
        )

class UniversalLoss(nn.Module):
    """
    通用自适应Loss计算器，自动检测输入中的各种任务并计算对应的损失：
    - CE任务的主预测（必需）
    - 2个额外的CE任务（denoise和prior，可选）
    - 顶点坐标预测MSE损失（可选）
    - 图像重建损失（MSE或其他，可选）
    """
    def __init__(self, args=None):
        super().__init__()
        weight = getattr(args, 'weight', None)
        reduction = getattr(args, 'reduction', 'mean')
        
        # CE任务系数（如果任务存在则使用）
        self.k_main = getattr(args, 'lambda_main', 0.6)
        self.k_denoise = getattr(args, 'lambda_denoise', 0.2)
        self.k_prior = getattr(args, 'lambda_prior', 0.1)
        
        # 顶点坐标系数
        self.k_verteces = getattr(args, 'lambda_verteces', 0.1)
        self.k_mpdiou = getattr(args, 'lambda_mpdiou', 0.5)
        
        # 图像重建系数
        self.k_img_recon_mse = getattr(args, 'lambda_img_recon', 0.5)
        # self.k_img_recon_ssim = getattr(args, 'lambda_img_recon', 0.25)
        
        
        # 损失函数定义
        self.CEloss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.MSEloss = nn.MSELoss(reduction='mean')
        # 重建图像评估
        # self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=3).cuda()
        return
    
    def forward(
        self,
        inputs: models.fullModel.InferBox,
        targets: datasets.BatchBox,
        *args,
        **kwargs
    ) -> LossBox:
        """
        自动检测输入中存在的任务，计算对应的损失
        
        inputs: InferBox (包含各种可选的预测输出)
        targets: BatchBox (包含各种可选的目标值)
        
        returns: LossBox (包含所有计算的损失分量)
        """
        total_loss = 0.0
        loss_main = None
        loss_denoise = None
        loss_prior = None
        loss_verteces = None
        loss_img_recon_mse = None
        loss_mpdiou = None
        
        
        # ===== 1. 主CE任务（必需） =====
        if inputs.LPs_logits is not None and targets.LPs is not None:
            loss_main = self.CEloss.forward(inputs.LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_main * loss_main
        else:
            # raise ValueError("主预测LPs_logits和目标LPs不能为空")
            pass
        
        # ===== 2. 额外任务1：Denoise CE任务 =====
        if inputs.denoise_LPs_logits is not None and targets.LPs is not None:
            loss_denoise = self.CEloss.forward(inputs.denoise_LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_denoise * loss_denoise
        
        # ===== 3. 额外任务2：Prior CE任务 =====
        if inputs.prior_LPs_logits is not None and targets.LPs is not None:
            loss_prior = self.CEloss.forward(inputs.prior_LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_prior * loss_prior
        
        # ===== 4. 顶点坐标MSE损失 =====
        if inputs.STN_verteces is not None and targets.verteces is not None:
            loss_verteces = self.MSEloss.forward(
                inputs.STN_verteces.flatten(1), 
                targets.verteces
            )
            total_loss = total_loss + self.k_verteces * loss_verteces
            # ===== 4. bbix iou 损失 =====
            if getattr(targets, 'bboxes', None) is not None:
                # inputs.STN_verteces: [B, 4, 2] usually
                pred_verts = inputs.STN_verteces.view(-1, 4, 2)
                # Normalize from [-1, 1] to [0, 1]
                pred_verts_01 = (pred_verts + 1.0) / 2.0
                
                min_xy, _ = pred_verts_01.min(dim=1)
                max_xy, _ = pred_verts_01.max(dim=1)
                pred_bboxes = torch.cat([min_xy, max_xy], dim=1)
                
                tgt_bboxes = targets.bboxes
                
                # MPDIoU calculation
                lt = torch.max(pred_bboxes[:, :2], tgt_bboxes[:, :2])
                rb = torch.min(pred_bboxes[:, 2:], tgt_bboxes[:, 2:])
                wh = (rb - lt).clamp(min=0)
                inter = wh[:, 0] * wh[:, 1]
                area_pred = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
                area_tgt = (tgt_bboxes[:, 2] - tgt_bboxes[:, 0]) * (tgt_bboxes[:, 3] - tgt_bboxes[:, 1])
                union = area_pred + area_tgt - inter
                iou = inter / (union + 1e-6)
                
                d1_2 = (pred_bboxes[:, 0] - tgt_bboxes[:, 0])**2 + (pred_bboxes[:, 1] - tgt_bboxes[:, 1])**2
                d2_2 = (pred_bboxes[:, 2] - tgt_bboxes[:, 2])**2 + (pred_bboxes[:, 3] - tgt_bboxes[:, 3])**2
                
                # w^2 + h^2 = 1^2 + 1^2 = 2 (normalized coords)
                loss_mpdiou = (1 - iou + d1_2 / 2.0 + d2_2 / 2.0).mean()
                total_loss = total_loss + self.k_mpdiou * loss_mpdiou
        
        # ===== 5. 图像重建损失 =====
        if inputs.reconstructed_img is not None and targets.images is not None:
            # 确保维度匹配后再计算损失
            # reconstructed_img 和 images 都应该是 [B, C, H, W] 格式
            loss_img_recon_mse = self.MSEloss.forward(
                inputs.reconstructed_img, 
                targets.hr_LP_img
            )
            # loss_ssim = 1 - self.ssim_module(inputs.reconstructed_img, targets.hr_LP_img)
            total_loss = total_loss + self.k_img_recon_mse * loss_img_recon_mse#+self.k_img_recon_ssim*loss_ssim
        
        return LossBox(
            total_loss=total_loss,
            main_loss=loss_main,
            denoise_loss=loss_denoise,
            prior_loss=loss_prior,
            verteces_loss=loss_verteces,
            img_recon_loss=loss_img_recon_mse,
            bbox_loss=loss_mpdiou,
        )


class UniversalLossV2(nn.Module):
    """
    通用自适应Loss计算器 V2
    自动检测输入中的各种任务并计算对应的损失：
    - CE任务的主预测（必需）
    - 2个额外的CE任务（denoise和prior，可选）
    - 顶点坐标预测MSE损失（可选）
    - Bbox直接预测的MPDIoU损失（可选）
    - 图像重建损失（MSE或其他，可选）
    """
    def __init__(self, args=None):
        super().__init__()
        weight = getattr(args, 'weight', None)
        reduction = getattr(args, 'reduction', 'mean')
        
        # CE任务系数
        self.k_main = getattr(args, 'lambda_main', 0.6)
        self.k_denoise = getattr(args, 'lambda_denoise', 0.2)
        self.k_prior = getattr(args, 'lambda_prior', 0.1)
        
        # 几何坐标系数
        self.k_verteces = getattr(args, 'lambda_verteces', 0.1)
        self.k_mpdiou = getattr(args, 'lambda_mpdiou', 0.5)
        
        # 图像重建系数
        self.k_img_recon_mse = getattr(args, 'lambda_img_recon', 0.5)
        # self.k_img_recon_ssim = getattr(args, 'lambda_img_recon', 0.25)
        
        # 损失函数定义
        self.CEloss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.MSEloss = nn.MSELoss(reduction='mean')
    
    def forward(self, inputs:InferBox, targets:BatchBox, *args, **kwargs):
        """
        自动检测输入中存在的任务，计算对应的损失
        """
        total_loss = 0.0
        loss_main = None
        loss_denoise = None
        loss_prior = None
        loss_verteces = None
        loss_mpdiou = None
        loss_img_recon_mse = None
        
        # ===== 1. 主CE任务（必需） =====
        if getattr(inputs, 'LPs_logits', None) is not None and getattr(targets, 'LPs', None) is not None:
            loss_main = self.CEloss(inputs.LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_main * loss_main
        
        # ===== 2. 额外任务1：Denoise CE任务 =====
        if getattr(inputs, 'denoise_LPs_logits', None) is not None and getattr(targets, 'LPs', None) is not None:
            loss_denoise = self.CEloss(inputs.denoise_LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_denoise * loss_denoise
        
        # ===== 3. 额外任务2：Prior CE任务 =====
        if getattr(inputs, 'prior_LPs_logits', None) is not None and getattr(targets, 'LPs', None) is not None:
            loss_prior = self.CEloss(inputs.prior_LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_prior * loss_prior
        
        # ===== 4. 顶点坐标MSE损失 =====
        if getattr(inputs, 'verteces', None) is not None and getattr(targets, 'verteces', None) is not None:
            loss_verteces = self.MSEloss(
                inputs.STN_verteces.flatten(1), 
                targets.verteces
            )
            total_loss = total_loss + self.k_verteces * loss_verteces
            
        # ===== 5. BBox MPDIoU 损失 =====
        if getattr(inputs, 'bboxes_hat', None) is not None and getattr(targets, 'bboxes', None) is not None:
            loss_mpdiou = mpdiou_loss(inputs.bboxes_hat, targets.bboxes)
            total_loss = total_loss + self.k_mpdiou * loss_mpdiou
        
        # ===== 6. 图像重建损失 =====
        if getattr(inputs, 'reconstructed_img', None) is not None and getattr(targets, 'hr_LP_img', None) is not None:
            loss_img_recon_mse = self.MSEloss(
                inputs.reconstructed_img, 
                targets.hr_LP_img
            )
            total_loss = total_loss + self.k_img_recon_mse * loss_img_recon_mse
        
        # 返回结果 (需确保LossBox正确导入)
        return LossBox(
            total_loss=total_loss,
            main_loss=loss_main,
            denoise_loss=loss_denoise,
            prior_loss=loss_prior,
            verteces_loss=loss_verteces,
            img_recon_loss=loss_img_recon_mse,
            bbox_loss=loss_mpdiou,
        )

class UniversalLossV2_theta(UniversalLossV2):
    def __init__(self, args=None):
        super().__init__(args)
        self.k_theta = getattr(args, 'lambda_theta', 0.1)
        return
    def forward(self, inputs:InferBox, targets:BatchBox, *args, **kwargs):
        """
        自动检测输入中存在的任务，计算对应的损失
        """
        total_loss = 0.0
        loss_main = None
        loss_denoise = None
        loss_prior = None
        loss_verteces = None
        loss_mpdiou = None
        loss_img_recon_mse = None
        
        # ===== 1. 主CE任务（必需） =====
        if getattr(inputs, 'LPs_logits', None) is not None and getattr(targets, 'LPs', None) is not None:
            loss_main = self.CEloss(inputs.LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_main * loss_main
        
        # ===== 2. 额外任务1：Denoise CE任务 =====
        if getattr(inputs, 'denoise_LPs_logits', None) is not None and getattr(targets, 'LPs', None) is not None:
            loss_denoise = self.CEloss(inputs.denoise_LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_denoise * loss_denoise
        
        # ===== 3. 额外任务2：Prior CE任务 =====
        if getattr(inputs, 'prior_LPs_logits', None) is not None and getattr(targets, 'LPs', None) is not None:
            loss_prior = self.CEloss(inputs.prior_LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_prior * loss_prior
        
        # ===== affine theta 预测loss =====
        if inputs.theta is not None and targets.theta is not None:
            loss_theta=self.MSEloss(inputs.theta,targets.theta)
            total_loss+=self.k_theta*loss_theta
        
        # ===== 6. 图像重建损失 =====
        if getattr(inputs, 'reconstructed_img', None) is not None and getattr(targets, 'hr_LP_img', None) is not None:
            loss_img_recon_mse = self.MSEloss(
                inputs.reconstructed_img, 
                targets.hr_LP_img
            )
            total_loss = total_loss + self.k_img_recon_mse * loss_img_recon_mse
        
        # 返回结果 (需确保LossBox正确导入)
        return LossBox(
            total_loss=total_loss,
            main_loss=loss_main,
            denoise_loss=loss_denoise,
            prior_loss=loss_prior,
            # verteces_loss=loss_verteces,
            img_recon_loss=loss_img_recon_mse,
            # bbox_loss=loss_mpdiou,
            theta_loss=loss_theta
        )

class Universal_polyLoss(nn.Module):
    """
    通用自适应Loss计算器，自动检测输入中的各种任务并计算对应的损失：
    - 所有的分类loss使用 Poly1Loss
    - CE任务的主预测（必需）
    - 2个额外的CE任务（denoise和prior，可选）
    - 顶点坐标预测MSE损失（可选）
    - 图像重建损失（MSE或其他，可选）
    """
    def __init__(self, args=None):
        super().__init__()
        weight = getattr(args, 'weight', None)
        reduction = getattr(args, 'reduction', 'mean')
        
        # CE任务系数（如果任务存在则使用）
        self.k_main = getattr(args, 'lambda_main', 0.6)
        self.k_denoise = getattr(args, 'lambda_denoise', 0.2)
        self.k_prior = getattr(args, 'lambda_prior', 0.1)
        
        # 顶点坐标系数
        self.k_verteces = getattr(args, 'lambda_verteces', 0.1)
        self.k_mpdiou = getattr(args, 'lambda_mpdiou', 0.5)
        
        # 图像重建系数
        self.k_img_recon_mse = getattr(args, 'lambda_img_recon', 0.5)
        # self.k_img_recon_ssim = getattr(args, 'lambda_img_recon', 0.25)
        
        
        # 损失函数定义
        self.CEloss = Poly1Loss(weight=weight, reduction=reduction)
        self.MSEloss = nn.MSELoss(reduction='mean')
        # 重建图像评估
        # self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=3).cuda()
        return
    
    def forward(
        self,
        inputs: models.fullModel.InferBox,
        targets: datasets.BatchBox,
        *args,
        **kwargs
    ) -> LossBox:
        """
        自动检测输入中存在的任务，计算对应的损失
        
        inputs: InferBox (包含各种可选的预测输出)
        targets: BatchBox (包含各种可选的目标值)
        
        returns: LossBox (包含所有计算的损失分量)
        """
        total_loss = 0.0
        loss_main = None
        loss_denoise = None
        loss_prior = None
        loss_verteces = None
        loss_img_recon_mse = None
        loss_mpdiou = None
        
        
        # ===== 1. 主CE任务（必需） =====
        if inputs.LPs_logits is not None and targets.LPs is not None:
            loss_main = self.CEloss.forward(inputs.LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_main * loss_main
        else:
            # raise ValueError("主预测LPs_logits和目标LPs不能为空")
            pass
        
        # ===== 2. 额外任务1：Denoise CE任务 =====
        if inputs.denoise_LPs_logits is not None and targets.LPs is not None:
            loss_denoise = self.CEloss.forward(inputs.denoise_LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_denoise * loss_denoise
        
        # ===== 3. 额外任务2：Prior CE任务 =====
        if inputs.prior_LPs_logits is not None and targets.LPs is not None:
            loss_prior = self.CEloss.forward(inputs.prior_LPs_logits, targets.LPs)
            total_loss = total_loss + self.k_prior * loss_prior
        
        # ===== 4. 顶点坐标MSE损失 =====
        if inputs.STN_verteces is not None and targets.verteces is not None:
            loss_verteces = self.MSEloss.forward(
                inputs.STN_verteces.flatten(1), 
                targets.verteces
            )
            total_loss = total_loss + self.k_verteces * loss_verteces
            # ===== 4. bbix iou 损失 =====
            if getattr(targets, 'bboxes', None) is not None:
                # inputs.STN_verteces: [B, 4, 2] usually
                pred_verts = inputs.STN_verteces.view(-1, 4, 2)
                # Normalize from [-1, 1] to [0, 1]
                pred_verts_01 = (pred_verts + 1.0) / 2.0
                
                min_xy, _ = pred_verts_01.min(dim=1)
                max_xy, _ = pred_verts_01.max(dim=1)
                pred_bboxes = torch.cat([min_xy, max_xy], dim=1)
                
                tgt_bboxes = targets.bboxes
                
                # MPDIoU calculation
                lt = torch.max(pred_bboxes[:, :2], tgt_bboxes[:, :2])
                rb = torch.min(pred_bboxes[:, 2:], tgt_bboxes[:, 2:])
                wh = (rb - lt).clamp(min=0)
                inter = wh[:, 0] * wh[:, 1]
                area_pred = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
                area_tgt = (tgt_bboxes[:, 2] - tgt_bboxes[:, 0]) * (tgt_bboxes[:, 3] - tgt_bboxes[:, 1])
                union = area_pred + area_tgt - inter
                iou = inter / (union + 1e-6)
                
                d1_2 = (pred_bboxes[:, 0] - tgt_bboxes[:, 0])**2 + (pred_bboxes[:, 1] - tgt_bboxes[:, 1])**2
                d2_2 = (pred_bboxes[:, 2] - tgt_bboxes[:, 2])**2 + (pred_bboxes[:, 3] - tgt_bboxes[:, 3])**2
                
                # w^2 + h^2 = 1^2 + 1^2 = 2 (normalized coords)
                loss_mpdiou = (1 - iou + d1_2 / 2.0 + d2_2 / 2.0).mean()
                total_loss = total_loss + self.k_mpdiou * loss_mpdiou
        
        # ===== 5. 图像重建损失 =====
        if inputs.reconstructed_img is not None and targets.images is not None:
            # 确保维度匹配后再计算损失
            # reconstructed_img 和 images 都应该是 [B, C, H, W] 格式
            loss_img_recon_mse = self.MSEloss.forward(
                inputs.reconstructed_img, 
                targets.hr_LP_img
            )
            # loss_ssim = 1 - self.ssim_module(inputs.reconstructed_img, targets.hr_LP_img)
            total_loss = total_loss + self.k_img_recon_mse * loss_img_recon_mse#+self.k_img_recon_ssim*loss_ssim
        
        return LossBox(
            total_loss=total_loss,
            main_loss=loss_main,
            denoise_loss=loss_denoise,
            prior_loss=loss_prior,
            verteces_loss=loss_verteces,
            img_recon_loss=loss_img_recon_mse,
            bbox_loss=loss_mpdiou,
        )

class RPNetLoss(nn.Module):
    """
    简化版 RPNet 损失函数
    - 仅包含：主分类损失 (CrossEntropy) 和 边界框损失 (GIoU)
    """
    def __init__(self, args=None):
        super().__init__()
        # 获取权重系数，默认为 1.0
        self.k_cls = getattr(args, 'lambda_cls', 1.0)
        self.k_bbox = getattr(args, 'lambda_bbox', 1.0)

        # 分类损失函数
        weight = getattr(args, 'weight', None)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs: models.fullModel.InferBox,
        targets: datasets.BatchBox,) :
        """
        inputs: 需包含预测的 logits (pred_logits) 和 预测的 bboxes (pred_boxes)
        targets: 需包含真实的 labels (gt_labels) 和 真实的 bboxes (gt_boxes)
        注意：bboxes 格式应为 [x1, y1, x2, y2]
        """
        # 1. 主分类损失 (Cross Entropy)
        # 假设 inputs.LPs_logits: [B, N], targets.LPs: [B]
        if inputs.LPs_logits is None or targets.LPs is None:
            loss_cls=0.0
        else:
            loss_cls = self.ce_loss(inputs.LPs_logits, targets.LPs)

        # 2. 边界框损失 (GIoU)
        if inputs.bboxes_hat is None or targets.bboxes is None:
            loss_bbox=0.0
        else:
            pred_bboxes=inputs.bboxes_hat # [B, 4]
            tgt_bboxes = targets.bboxes # [B, 4]
            loss_mse = F.mse_loss(pred_bboxes, tgt_bboxes, reduction='mean')
            loss_giou = GIoU_loss(pred_bboxes, tgt_bboxes, reduction='mean')
            loss_bbox = 2.0 * loss_mse + 1.0 * loss_giou

        # 总损失加权
        total_loss = self.k_cls * loss_cls + self.k_bbox * loss_bbox
        
        return LossBox(total_loss=total_loss, main_loss=loss_cls, bbox_loss=loss_bbox)

'''
infer use eval staistic classes
'''


class LPs_evaluator:
    """
    base class for LPs evaluator
    """

    def __init__(self):
        self.reset()
        return

    def reset(self):
        """reset all statistic states"""
        self.LP_total = 0
        self.LP_correct = 0
        self.LP_total_denoise=0
        self.LP_correct_denoise = 0
        self.LP_total_prior=0
        self.LP_correct_prior = 0
        
        # 字符级别正确率统计
        self.char_total = 0
        self.char_correct = 0
        # self.char_total_denoise = 0
        # self.char_correct_denoise = 0
        # self.char_total_prior = 0
        # self.char_correct_prior = 0
        
        # 去除第一个汉字字符的字符级别正确率统计
        self.char_without_first_total = 0
        self.char_without_first_correct = 0
        # self.char_without_first_total_denoise = 0
        # self.char_without_first_correct_denoise = 0
        # self.char_without_first_total_prior = 0
        # self.char_without_first_correct_prior = 0
        
        self.recon_mse_sum = 0.0
        self.recon_psnr_sum = 0.0
        self.recon_count = 0
        self.vert_mse_sum = 0.0
        self.vert_count = 0
        self.iou_sum = 0.0
        self.iou_count = 0
        self.ap_70 = 0
        self.ap_50=0
        return

    @staticmethod
    def calcu_LP_match(string_logits: torch.Tensor, string_GT: torch.Tensor):
        """
        string_logits: [B,C,N]
        string_GT: [B,N]
        return: matches: [B] True/False
        """
        string_gt = string_GT  # [B, N]
        string_pred = string_logits.argmax(dim=1)  # [B, N]

        # 判断每个样本的预测字符串是否完全等于 GT
        matches = (string_pred == string_gt).all(dim=1)
        return matches  # [B]，每个元素是 True/False
    
    @staticmethod
    def calcu_char_accuracy(string_logits: torch.Tensor, string_GT: torch.Tensor):
        """
        计算字符级别的准确率
        string_logits: [B,C,N]
        string_GT: [B,N]
        return: (total_chars, correct_chars)
        """
        string_gt = string_GT  # [B, N]
        string_pred = string_logits.argmax(dim=1)  # [B, N]
        
        # 计算字符级别的正确匹配
        char_matches = (string_pred == string_gt)  # [B, N]
        total_chars = string_gt.numel()
        correct_chars = char_matches.sum().item()
        
        return total_chars, correct_chars
    
    @staticmethod
    def calcu_char_without_first_accuracy(string_logits: torch.Tensor, string_GT: torch.Tensor):
        """
        计算去除第一个汉字字符后的字符级别准确率
        string_logits: [B,C,N]
        string_GT: [B,N]
        return: (total_chars, correct_chars) excluding first character
        """
        string_gt = string_GT  # [B, N]
        string_pred = string_logits.argmax(dim=1)  # [B, N]
        
        # 去除第一个字符，只计算后续字符
        if string_gt.shape[1] > 1:
            string_gt_without_first = string_gt[:, 1:]  # [B, N-1]
            string_pred_without_first = string_pred[:, 1:]  # [B, N-1]
            
            char_matches = (string_pred_without_first == string_gt_without_first)  # [B, N-1]
            total_chars = string_gt_without_first.numel()
            correct_chars = char_matches.sum().item()
        else:
            total_chars = 0
            correct_chars = 0
        
        return total_chars, correct_chars

    @torch.no_grad()
    def forward_batch(
        self, outputs: models.fullModel.InferBox, targets: datasets.BatchBox
    ):
        """
        outputs: InferBox
        targets: BatchBox
        """
        if outputs.LPs_logits is not None and targets.LPs is not None:
            # 序列级别正确率
            matches = self.calcu_LP_match(outputs.LPs_logits, targets.LPs)
            self.LP_total += matches.shape[0]
            self.LP_correct += matches.sum().item()
            
            # 字符级别正确率
            total_chars, correct_chars = self.calcu_char_accuracy(outputs.LPs_logits, targets.LPs)
            self.char_total += total_chars
            self.char_correct += correct_chars
            
            # 去除第一个字符的字符级别正确率
            total_chars_without_first, correct_chars_without_first = self.calcu_char_without_first_accuracy(outputs.LPs_logits, targets.LPs)
            self.char_without_first_total += total_chars_without_first
            self.char_without_first_correct += correct_chars_without_first
            
        if outputs.denoise_LPs_logits is not None and targets.LPs is not None:
            matches = self.calcu_LP_match(outputs.denoise_LPs_logits, targets.LPs)
            self.LP_total_denoise += matches.shape[0]
            self.LP_correct_denoise += matches.sum().item()
            
            # 字符级别正确率
            total_chars, correct_chars = self.calcu_char_accuracy(outputs.denoise_LPs_logits, targets.LPs)
            self.char_total_denoise += total_chars
            self.char_correct_denoise += correct_chars
            
            # 去除第一个字符的字符级别正确率
            total_chars_without_first, correct_chars_without_first = self.calcu_char_without_first_accuracy(outputs.denoise_LPs_logits, targets.LPs)
            self.char_without_first_total_denoise += total_chars_without_first
            self.char_without_first_correct_denoise += correct_chars_without_first
            
        if outputs.prior_LPs_logits is not None and targets.LPs is not None:
            matches = self.calcu_LP_match(outputs.prior_LPs_logits, targets.LPs)
            self.LP_total_prior += matches.shape[0]
            self.LP_correct_prior += matches.sum().item()
            
            # 字符级别正确率
            total_chars, correct_chars = self.calcu_char_accuracy(outputs.prior_LPs_logits, targets.LPs)
            self.char_total_prior += total_chars
            self.char_correct_prior += correct_chars
            
            # 去除第一个字符的字符级别正确率
            total_chars_without_first, correct_chars_without_first = self.calcu_char_without_first_accuracy(outputs.prior_LPs_logits, targets.LPs)
            self.char_without_first_total_prior += total_chars_without_first
            self.char_without_first_correct_prior += correct_chars_without_first

        # Image Reconstruction Evaluation
        if outputs.reconstructed_img is not None and getattr(targets, 'hr_LP_img', None) is not None:
            recon_img = outputs.reconstructed_img
            hr_img = targets.hr_LP_img
            
            # Calculate MSE per sample
            mse = F.mse_loss(recon_img, hr_img, reduction='none').mean(dim=[1, 2, 3]) # [B]
            self.recon_mse_sum += mse.sum().item()
            
            # Calculate PSNR per sample (assuming [0, 1] range)
            mse_safe = torch.clamp(mse, min=1e-8)
            psnr = 10 * torch.log10(1.0 / mse_safe)
            self.recon_psnr_sum += psnr.sum().item()
            
            self.recon_count += recon_img.size(0)

        # Vertex and BBox Evaluation
        if outputs.STN_verteces is not None and targets.verteces is not None:
            pred_verts = outputs.STN_verteces # [B, 4, 2]
            tgt_verts = targets.verteces # [B, 8]
            B = pred_verts.shape[0]

            # 1. Vertex MSE (on [-1, 1] scale)
            pred_verts_flat = pred_verts.reshape(B, -1)
            tgt_verts_flat = tgt_verts.reshape(B, -1)
            
            mse_verts = F.mse_loss(pred_verts_flat, tgt_verts_flat, reduction='sum')
            self.vert_mse_sum += mse_verts.item()
            self.vert_count += B

            # 2. BBox IoU & AP@0.5
            if targets.bboxes is not None:
                # Convert pred vertices [-1, 1] -> [0, 1]
                pred_verts_01 = (pred_verts.reshape(B, 4, 2) + 1.0) / 2.0
                
                # Get min/max to form bbox [x1, y1, x2, y2]
                min_xy, _ = pred_verts_01.min(dim=1) # [B, 2]
                max_xy, _ = pred_verts_01.max(dim=1) # [B, 2]
                pred_bboxes = torch.cat([min_xy, max_xy], dim=1) # [B, 4]

                # Get target bboxes (targets.bboxes is [B, 4] in [0, 1])
                tgt_bboxes = targets.bboxes
                
                # Calculate IoU
                ious = ops.box_iou(pred_bboxes, tgt_bboxes).diag()
                
                self.iou_sum += ious.sum().item()
                self.iou_count += B
                self.ap_70 += (ious >= 0.7).sum().item()
                self.ap_50 += (ious >= 0.5).sum().item()
        return

    def statistic_Dataset(self,reset: bool = True):
        """return statistic result box"""
        lp_accuracy = self.LP_correct / self.LP_total if self.LP_total > 0 else 0.0
        lp_accuracy_denoise = self.LP_correct_denoise / self.LP_total_denoise if self.LP_total_denoise > 0 else 0.0
        lp_accuracy_prior = self.LP_correct_prior / self.LP_total_prior if self.LP_total_prior > 0 else 0.0
        
        # 字符级别正确率
        char_accuracy = self.char_correct / self.char_total if self.char_total > 0 else 0.0
        
        # 去除第一个字符后的字符级别正确率
        char_without_first_accuracy = self.char_without_first_correct / self.char_without_first_total if self.char_without_first_total > 0 else 0.0
        
        recon_mse = self.recon_mse_sum / self.recon_count if self.recon_count > 0 else None
        recon_psnr = self.recon_psnr_sum / self.recon_count if self.recon_count > 0 else None

        vert_mse = self.vert_mse_sum / self.vert_count if self.vert_count > 0 else None
        mean_iou = self.iou_sum / self.iou_count if self.iou_count > 0 else None
        ap_70 = self.ap_70 / self.iou_count if self.iou_count > 0 else None
        ap_50 = self.ap_50 / self.iou_count if self.iou_count > 0 else None

        if reset:
            self.reset()

        return EvalBox(
            # LP_total=self.LP_total,
            # LP_correct=self.LP_correct,
            LP_acc=lp_accuracy,
            LP_acc_denoise=lp_accuracy_denoise,
            LP_acc_prior=lp_accuracy_prior,
            LP_err=1.0 - lp_accuracy,
            char_acc=char_accuracy,
            char_without_first_acc=char_without_first_accuracy,
            recon_mse=recon_mse,
            recon_psnr=recon_psnr,
            verteces_mse=vert_mse,
            bbox_iou=mean_iou,
            bbox_ap70=ap_70,
            bbox_ap50=ap_50
        )

    pass

class LPs_evaluator_v2:
    """
    Base class for LPs evaluator
    Vertex 和 BBox 的逻辑分离，支持多分支 (Standard, Denoise, Prior) 评测
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistic states using a structured dictionary"""
        self.text_stats = {
            branch: {
                'lp_total': 0, 'lp_correct': 0,
                'char_total': 0, 'char_correct': 0,
                'char_wo_first_total': 0, 'char_wo_first_correct': 0
            }
            for branch in ['standard', 'denoise', 'prior']
        }

        self.recon = {'mse_sum': 0.0, 'psnr_sum': 0.0, 'count': 0}
        
        # 新增 'vert_nme_sum': 0.0
        self.loc = {'vert_mse_sum': 0.0, 'vert_nme_sum': 0.0, 'vert_count': 0, 
                    'iou_sum': 0.0, 'iou_count': 0, 'ap_50': 0, 'ap_70': 0}


    @staticmethod
    def _get_text_matches(string_logits: torch.Tensor, string_gt: torch.Tensor):
        """辅助函数：一次性计算所有的匹配结果，避免重复 argmax"""
        string_pred = string_logits.argmax(dim=1)  # [B, N]
        char_matches = (string_pred == string_gt)  # [B, N]
        lp_matches = char_matches.all(dim=1)       # [B]
        return char_matches, lp_matches

    def _update_text_stats(self, branch_name: str, logits: torch.Tensor, targets: torch.Tensor):
        """统一更新指定分支的文本识别统计数据"""
        char_matches, lp_matches = self._get_text_matches(logits, targets)
        stats = self.text_stats[branch_name]

        # 1. 序列级别 (LP)
        stats['lp_total'] += lp_matches.shape[0]
        stats['lp_correct'] += lp_matches.sum().item()

        # 2. 字符级别 (Char)
        stats['char_total'] += targets.numel()
        stats['char_correct'] += char_matches.sum().item()

        # 3. 去除首字符级别 (Char without first)
        if targets.shape[1] > 1:
            stats['char_wo_first_total'] += targets[:, 1:].numel()
            stats['char_wo_first_correct'] += char_matches[:, 1:].sum().item()

    @torch.no_grad()
    def forward_batch(self, outputs:InferBox, targets:BatchBox):
        """
        处理单个 batch 的输出和目标
        """
        # --- 1. 文本识别评估 (Text Recognition) ---
        if targets.LPs is not None:
            branch_map = {
                'standard': outputs.LPs_logits,
                'denoise': outputs.denoise_LPs_logits,
                'prior': outputs.prior_LPs_logits
            }
            for branch, logits in branch_map.items():
                if logits is not None:
                    self._update_text_stats(branch, logits, targets.LPs)

        # --- 2. 图像重建评估 (Image Reconstruction) ---
        if outputs.reconstructed_img is not None and getattr(targets, 'hr_LP_img', None) is not None:
            mse = F.mse_loss(outputs.reconstructed_img, targets.hr_LP_img, reduction='none').mean(dim=[1, 2, 3])
            
            self.recon['mse_sum'] += mse.sum().item()
            self.recon['psnr_sum'] += (10 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))).sum().item()
            self.recon['count'] += outputs.reconstructed_img.size(0)

        # --- 3. 关键点评估 (Vertex Evaluation) ---
        if outputs.STN_verteces is not None and targets.verteces is not None:
            B = outputs.STN_verteces.shape[0]
            
            # 原有的 MSE
            mse_verts = F.mse_loss(outputs.STN_verteces.reshape(B, -1), targets.verteces.reshape(B, -1), reduction='sum')
            self.loc['vert_mse_sum'] += mse_verts.item()
            self.loc['vert_count'] += B

            # 新增: NME (Normalized Mean Error) 计算
            # 假设 verteces 的逻辑形状为 [B, 4, 2]
            preds_v = outputs.STN_verteces.reshape(B, 4, 2)
            gts_v = targets.verteces.reshape(B, 4, 2)
            
            # 1. 计算每个顶点的 L2 距离 (欧氏距离): [B, 4]
            l2_dist = torch.norm(preds_v - gts_v, dim=-1)
            
            # 2. 计算 GT 多边形的外接矩形对角线作为归一化因子: [B]
            gt_x = gts_v[..., 0]
            gt_y = gts_v[..., 1]
            w = gt_x.max(dim=1)[0] - gt_x.min(dim=1)[0]
            h = gt_y.max(dim=1)[0] - gt_y.min(dim=1)[0]
            diag = torch.sqrt(w**2 + h**2)
            diag = torch.clamp(diag, min=1e-6) # 防止极小目标导致除零异常
            
            # 3. 计算当前 batch 的 NME 并累加
            # 先求4个顶点的平均欧式距离，再除以对角线进行归一化
            nme_batch = (l2_dist.mean(dim=1) / diag).sum().item()
            self.loc['vert_nme_sum'] += nme_batch

        # --- 4. 边界框评估 (BBox Evaluation) ---
        if outputs.bboxes_hat is not None and targets.bboxes is not None:
            ious = ops.box_iou(outputs.bboxes_hat, targets.bboxes).diag()
            
            self.loc['iou_sum'] += ious.sum().item()
            self.loc['iou_count'] += outputs.bboxes_hat.shape[0]
            self.loc['ap_70'] += (ious >= 0.7).sum().item()
            self.loc['ap_50'] += (ious >= 0.5).sum().item()

    def _safe_div(self, num, den):
        """安全除法，防止除以 0"""
        return num / den if den > 0 else 0.0

    def statistic_Dataset(self, reset: bool = True):
        """返回统计结果 Box"""
        
        # 提取 Standard 分支作为主要的文本指标
        std_stats = self.text_stats['standard']
        
        lp_acc = self._safe_div(std_stats['lp_correct'], std_stats['lp_total'])
        lp_acc_denoise = self._safe_div(self.text_stats['denoise']['lp_correct'], self.text_stats['denoise']['lp_total'])
        lp_acc_prior = self._safe_div(self.text_stats['prior']['lp_correct'], self.text_stats['prior']['lp_total'])
        
        char_acc = self._safe_div(std_stats['char_correct'], std_stats['char_total'])
        char_wo_first_acc = self._safe_div(std_stats['char_wo_first_correct'], std_stats['char_wo_first_total'])

        # 提取其他指标
        recon_mse = self._safe_div(self.recon['mse_sum'], self.recon['count']) if self.recon['count'] > 0 else None
        recon_psnr = self._safe_div(self.recon['psnr_sum'], self.recon['count']) if self.recon['count'] > 0 else None
        
        vert_mse = self._safe_div(self.loc['vert_mse_sum'], self.loc['vert_count']) if self.loc['vert_count'] > 0 else None
        vert_nme = self._safe_div(self.loc['vert_nme_sum'], self.loc['vert_count']) if self.loc['vert_count'] > 0 else None
        mean_iou = self._safe_div(self.loc['iou_sum'], self.loc['iou_count']) if self.loc['iou_count'] > 0 else None
        ap_70 = self._safe_div(self.loc['ap_70'], self.loc['iou_count']) if self.loc['iou_count'] > 0 else None
        ap_50 = self._safe_div(self.loc['ap_50'], self.loc['iou_count']) if self.loc['iou_count'] > 0 else None

        if reset:
            self.reset()

        return EvalBox(
            LP_acc=lp_acc,
            LP_acc_denoise=lp_acc_denoise,
            LP_acc_prior=lp_acc_prior,
            LP_err=1.0 - lp_acc,
            char_acc=char_acc,
            char_without_first_acc=char_wo_first_acc,
            recon_mse=recon_mse,
            recon_psnr=recon_psnr,
            verteces_mse=vert_mse,
            verteces_nme=vert_nme,
            bbox_iou=mean_iou,
            bbox_ap70=ap_70,
            bbox_ap50=ap_50
        )



# ---------------------------------------------------------------------------
# exDETR Loss: Hungarian matching + CDN denoise loss
# Matches predictions from ExDETR_Decoder/ExDETR_CDN output to GT.
# ---------------------------------------------------------------------------

class ExDETR_Matcher(nn.Module):
    '''Hungarian/greedy matcher for exDETR grouped-token predictions.
    Cost = k_class * cost_class + k_l1 * cost_vertex_l1 + k_giou * cost_giou
           + k_string * cost_string
    pred_boxes [B,N,8]: 4 vertices (x,y) in [0,1]. GT bbox is xyxy axis-aligned.
    Targets per image: always 1 plate -> greedy argmin (no Hungarian needed).
    '''
    def __init__(
        self,
        k_class: float = 1.0,
        k_l1: float = 5.0,
        k_giou: float = 2.0,
        k_l11: float = 5.0,   # kept for API compatibility, unused
        k_giou1: float = 2.0, # kept for API compatibility, unused
        k_string: float = 10.0,
    ):
        super().__init__()
        self.k_class = k_class
        self.k_l1 = k_l1
        self.k_giou = k_giou
        self.k_string = k_string
        self._ce = nn.CrossEntropyLoss(reduction='none')

    @torch.no_grad()
    def forward(self, outputs: dict, targets: BatchBox) -> torch.Tensor:
        '''Returns indices [B, 1] of the best matching prediction for each image.
        outputs: dict with pred_logits[B,N,C], pred_boxes[B,N,8] (4 vertices [0,1]),
                 pred_string_logits[B,n_char,N,L]
        targets: BatchBox with .bboxes[B,4] xyxy, .plateType[B], .LPs[B,L]
        '''
        B, N, _ = outputs['pred_logits'].shape
        device = outputs['pred_logits'].device

        out_prob = outputs['pred_logits'].softmax(-1)       # [B, N, n_class]
        # pred_boxes: 4 vertices [0,1] -> derive axis-aligned bbox xyxy
        verts = outputs['pred_boxes'].reshape(B, N, 4, 2)   # [B, N, 4, 2]
        pred_bbox_xyxy = torch.cat([verts.min(dim=2).values,
                                    verts.max(dim=2).values], dim=-1)  # [B, N, 4] xyxy

        # plateType: [B] class index; default to 1 (plate present)
        tgt_labels = (
            targets.plateType
            if targets.plateType is not None
            else torch.ones(B, dtype=torch.long, device=device)
        )
        tgt_labels = tgt_labels.to(device)
        tgt_bbox = targets.bboxes.to(device)    # [B, 4] xyxy

        # class cost: advanced indexing, shape [B, N]
        cost_class_b = -out_prob[torch.arange(B, device=device), :, tgt_labels]  # [B, N]

        # vertex L1 cost: batched cdist [B,N,4] vs [B,1,4] -> [B,N,1] -> [B,N]
        cost_bbox_b = torch.cdist(pred_bbox_xyxy, tgt_bbox.unsqueeze(1), p=1).squeeze(-1)  # [B, N]

        # GIoU cost: fold B*N pairs onto diagonal to avoid O((BN)^2) full matrix
        pred_flat = pred_bbox_xyxy.reshape(B * N, 4)
        tgt_flat  = tgt_bbox.unsqueeze(1).expand(B, N, 4).reshape(B * N, 4)
        cost_giou_b = -generalized_box_iou_compilable(pred_flat, tgt_flat).diagonal().reshape(B, N)

        # string cost [B, N, L]
        tgt_lp = targets.LPs
        cost_string_b = torch.zeros(B, N, device=device)
        if tgt_lp is not None:
            tgt_lp = tgt_lp.to(device)
            pred_str = outputs['pred_string_logits']  # [B, n_char, N, L]
            L = pred_str.size(-1)
            tgt_rep = tgt_lp[:, :L].unsqueeze(1).expand(B, N, L)  # [B, N, L]
            pred_str_t = pred_str.permute(0, 2, 3, 1)              # [B, N, L, n_char]
            cost_string_b = self._ce(
                pred_str_t.reshape(B * N * L, -1),
                tgt_rep.reshape(B * N * L),
            ).view(B, N, L).mean(dim=-1)

        C = (
            self.k_class * cost_class_b
            + self.k_l1 * cost_bbox_b
            + self.k_giou * cost_giou_b
            + self.k_string * cost_string_b
        )  # [B, N]
        indices = C.argmin(dim=1, keepdim=True)  # [B, 1]
        return indices


class ExDETR_InferLoss(nn.Module):
    '''Loss for matched exDETR predictions.
    pred_boxes [B,N,8]: 4 vertices (x,y) in [0,1]. bbox derived as bounding rect.
    Returns raw (unweighted) sub-losses; weighting is done in UniversalLoss_exDETR.
    '''
    def __init__(
        self,
        k_class: float = 1.0,   # kept for Matcher API compat, NOT used for loss scaling here
        k_l1: float = 5.0,
        k_giou: float = 2.0,
        k_l11: float = 5.0,     # unused
        k_giou1: float = 2.0,   # unused
        k_string: float = 10.0,
        n_class: int = 2,
        void_class_idx: int = 0,
        void_class_weight: float = 0.1,
    ):
        super().__init__()
        class_weights = torch.ones(n_class)
        class_weights[void_class_idx] = void_class_weight
        self.register_buffer('class_weights', class_weights)
        self.class_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.string_loss = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, targets: BatchBox, indices: torch.Tensor):
        '''indices: [B, 1] matched prediction index per image.
        Returns: (bbox_cls_loss, bbox_l1_loss, bbox_giou_loss, string_loss)  — all unweighted.
        '''
        B = outputs['pred_logits'].size(0)
        device = outputs['pred_logits'].device
        batch_ids = torch.arange(B, device=device)
        idx = indices[:, 0]

        # --- bbox class loss: matched slot -> plateType, rest -> void ---
        tgt_labels = targets.plateType if targets.plateType is not None \
                     else torch.ones(B, dtype=torch.long, device=device)
        tgt_labels = tgt_labels.to(device=device, dtype=torch.long)
        N = outputs['pred_logits'].size(1)
        pred_set_labels = torch.zeros(B, N, dtype=torch.long, device=device)
        pred_set_labels[batch_ids, idx] = tgt_labels
        bbox_cls_loss = F.cross_entropy(outputs['pred_logits'].permute(0, 2, 1), pred_set_labels,
                                        weight=self.class_weights)

        # --- bbox regression loss: matched vertices -> axis-aligned bbox ---
        pred_verts = outputs['pred_boxes'][batch_ids, idx]   # [B, 8]
        pred_verts_2d = pred_verts.reshape(B, 4, 2)          # [B, 4, 2]
        pred_bbox = torch.cat([pred_verts_2d.min(dim=1).values,
                               pred_verts_2d.max(dim=1).values], dim=-1)  # [B, 4] xyxy
        tgt_bbox = targets.bboxes.to(device)                 # [B, 4] xyxy
        bbox_l1_loss  = self.l1_loss(pred_bbox, tgt_bbox)
        bbox_giou_loss = GIoU_loss(pred_bbox, tgt_bbox, reduction='mean')

        # --- string CE loss on matched slot ---
        tgt_lp = targets.LPs
        string_loss = torch.tensor(0.0, device=device)
        if tgt_lp is not None:
            tgt_lp = tgt_lp.to(device)
            pred_str = outputs['pred_string_logits'][batch_ids, :, idx, :]  # [B, n_char, L]
            L = pred_str.size(-1)
            string_loss = self.string_loss(pred_str, tgt_lp[:, :L])

        return bbox_cls_loss, bbox_l1_loss, bbox_giou_loss, string_loss


class ExDETR_DenoiseLoss(nn.Module):
    '''CDN denoise loss for exDETR.
    pos_bbox [B,8]: 4 predicted vertices (x,y) in [0,1]. bbox derived as bounding rect.
    Returns raw (unweighted) sub-losses; weighting is done in UniversalLoss_exDETR.
    '''
    def __init__(
        self,
        k_class: float = 1.0,   # unused — kept for API compat
        k_l1: float = 5.0,
        k_giou: float = 2.0,
        k_l11: float = 5.0,     # unused
        k_giou1: float = 2.0,   # unused
        k_string: float = 10.0,
    ):
        super().__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.string_loss = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, targets: BatchBox):
        '''outputs must contain: pos_class_logit, neg_class_logit, pos_bbox [B,8],
           LP_dn_logits [B,n_char,L].
        Returns: (dn_cls_loss, dn_l1_loss, dn_giou_loss, dn_string_loss) — all unweighted.
        '''
        device = outputs['pos_bbox'].device
        B = outputs['pos_bbox'].size(0)

        # class: pos -> plate present (1), neg -> void (0)
        pos_cls_tgt = torch.ones(B, dtype=torch.long, device=device)
        neg_cls_tgt = torch.zeros(B, dtype=torch.long, device=device)
        dn_cls_loss = (
            self.class_loss(outputs['pos_class_logit'], pos_cls_tgt)
            + self.class_loss(outputs['neg_class_logit'], neg_cls_tgt)
        ) * 0.5

        # bbox regression: derive axis-aligned bbox from predicted vertices
        tgt_bbox = targets.bboxes.to(device)                 # [B, 4] xyxy
        pos_verts = outputs['pos_bbox'].reshape(B, 4, 2)     # [B, 4, 2]
        pos_bbox = torch.cat([pos_verts.min(dim=1).values,
                              pos_verts.max(dim=1).values], dim=-1)  # [B, 4] xyxy
        dn_l1_loss   = self.l1_loss(pos_bbox, tgt_bbox)
        dn_giou_loss = GIoU_loss(pos_bbox, tgt_bbox, reduction='mean')

        # string CE on denoised tokens
        tgt_lp = targets.LPs
        dn_string_loss = torch.tensor(0.0, device=device)
        if tgt_lp is not None:
            tgt_lp = tgt_lp.to(device)
            dn_logits = outputs['LP_dn_logits']  # [B, n_char, L]
            L = dn_logits.size(-1)
            dn_string_loss = self.string_loss(dn_logits, tgt_lp[:, :L])

        return dn_cls_loss, dn_l1_loss, dn_giou_loss, dn_string_loss


class UniversalLoss_exDETR(nn.Module):
    '''
    Full loss for mtLPRtr_CvNxt_FPN_STN_co2D_exDETR_CDN.
    All sub-loss weighting is centralised here; sub-modules return raw scalars.

    LossBox fields used:
      main_loss      — string CE (infer, matched)
      bbox_cls_loss  — LP/void class CE (infer, all slots)
      bbox_reg_loss  — bbox L1 (infer, matched)
      bbox_giou_loss — bbox GIoU (infer, matched)
      denoise_loss   — weighted CDN total (for display)
      verteces_loss  — STN vertex MSE (if present)
      bbox_loss      — STN bbox MPDIoU (if present)
    '''
    def __init__(self, args=None):
        super().__init__()
        k_class  = getattr(args, 'k_class',  1.0)
        k_l1     = getattr(args, 'k_l1',     5.0)
        k_giou   = getattr(args, 'k_giou',   2.0)
        k_l11    = getattr(args, 'k_l11',    5.0)
        k_giou1  = getattr(args, 'k_giou1',  2.0)
        k_string = getattr(args, 'k_string', 10.0)

        # top-level branch weights
        self.k_class  = k_class
        self.k_l1     = k_l1
        self.k_giou   = k_giou
        self.k_string = k_string
        self.k_infer  = getattr(args, 'lambda_infer', 1.0)
        self.k_dn     = getattr(args, 'lambda_dn',    0.5)

        self.matcher    = ExDETR_Matcher(k_class, k_l1, k_giou, k_l11, k_giou1, k_string)
        self.infer_loss = ExDETR_InferLoss(k_class, k_l1, k_giou, k_l11, k_giou1, k_string)
        self.dn_loss    = ExDETR_DenoiseLoss(k_class, k_l1, k_giou, k_l11, k_giou1, k_string)

    def forward(self, inputs: InferBox, targets: BatchBox, *args, **kwargs):
        exdetr = getattr(inputs, 'exdetr_out', None)

        # raw sub-losses (all None until computed)
        bbox_cls_l = bbox_reg_l = bbox_giou_l = string_l = None
        dn_total_l = None

        if exdetr is not None:
            device = exdetr['pred_logits'].device
            total_loss = torch.zeros(1, device=device).squeeze()  # 直接在目标 device 创建，避免 graph break

            # --- infer branch: 4 raw losses ---
            indices = self.matcher(exdetr, targets)
            bbox_cls_l, bbox_reg_l, bbox_giou_l, string_l = \
                self.infer_loss.forward(exdetr, targets, indices)

            infer_total = (
                self.k_class  * bbox_cls_l
                + self.k_l1   * bbox_reg_l
                + self.k_giou * bbox_giou_l
                + self.k_string * string_l
            )
            total_loss = total_loss + self.k_infer * infer_total

            # --- CDN denoise branch (optional) ---
            if 'pos_bbox' in exdetr:
                dn_cls, dn_l1, dn_giou, dn_str = self.dn_loss(exdetr, targets)
                dn_total_l = (
                    self.k_class  * dn_cls
                    + self.k_l1   * dn_l1
                    + self.k_giou * dn_giou
                    + self.k_string * dn_str
                )
                total_loss = total_loss + self.k_dn * dn_total_l

        return LossBox(
            total_loss=total_loss,
            main_loss=string_l,           # string CE — the primary recognition loss
            bbox_cls_loss=bbox_cls_l,     # LP/void classification
            verteces_loss=bbox_reg_l,     # bbox L1
            bbox_loss=bbox_giou_l,   # bbox GIoU
            denoise_loss=dn_total_l,      # CDN branch total (weighted internally above, displayed as-is)
        )
