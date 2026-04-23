from dataclasses import fields, dataclass
from typing import Optional, Dict, Any, Union, List, Tuple
import time
import os
import queue
import threading
from dynaconf import Dynaconf
from torch.utils.tensorboard import SummaryWriter
import torch
import models.Loss
import shutil
import math
import numpy as np
import cv2

@dataclass
class CkptBox:
    """Checkpoint 信息封装类，统一管理 checkpoint 包含的所有状态"""
    # 训练全局步数
    global_step: int
    # 当前训练轮次
    epoch: int
    # 历史最佳准确率
    best_acc: float
    # 模型权重（state_dict）
    model_state_dict: Dict[str, Any]
    # 优化器状态（可选）
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    # 学习率调度器状态（可选）
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    # 混合精度缩放器状态（可选）
    scaler_state_dict: Optional[Dict[str, Any]] = None
    pass

def resolve_attr(root, dotted_name: str, default=None):
    """支持 'A.B.C' 形式的属性解析"""
    obj = root
    for part in dotted_name.split('.'):
        obj = getattr(obj, part,default)
    return obj

class Logger:
    '''
    tensorboard; save/load ckpt; print progress bar
    training process management
    '''
    def __init__(self, args: Dynaconf, global_step: int = 0,best_acc: float = 0.0,backup_cfg: bool= True):
        self.global_step = global_step
        self.best_acc = best_acc
        self.epoch = 0
        self.args = args
        log_dir: str = resolve_attr(self.args.LOGGER, "log_dir", "runs/tmp")
        self.tensorboard = SummaryWriter(log_dir)
        # 后台写入队列：主线程只入队，写盘在独立守护线程完成
        self._tb_queue = queue.SimpleQueue()
        self._tb_thread = threading.Thread(target=self._tb_writer_loop, daemon=True)
        self._tb_thread.start()
        if backup_cfg:
            self.save_cfg(self.args,os.path.join(log_dir, "config.yaml"))
        return

    def _tb_writer_loop(self):
        '''后台线程：从队列取 (tag, value, step) 写入 TensorBoard'''
        while True:
            item = self._tb_queue.get()
            if item is None:   # 毒丸，通知退出
                break
            tag, value, step = item
            self.tensorboard.add_scalar(tag, value, step)

    def _tb_add_scalar(self, tag: str, value, step: int):
        '''主线程调用：非阻塞入队'''
        self._tb_queue.put((tag, value, step))

    def close(self):
        '''训练结束后调用，等待后台线程写完再关闭'''
        self._tb_queue.put(None)
        self._tb_thread.join()
        self.tensorboard.close()
    @staticmethod
    def save_cfg(args:Dynaconf, save_cfg_path: str):
        import yaml
        os.makedirs(os.path.dirname(save_cfg_path), exist_ok=True)
        cfg_dict = args.as_dict()
        with open(save_cfg_path, 'w') as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
            pass
        print(f"配置已备份到: {save_cfg_path}")
        return

    def epoch_init(self, epoch: int, end_step: int):
        self.epoch = epoch
        self.progress_bar = self.ProgressBar(end_step)
        self.tmp_loss_list=[]
        self.train_loss_sum_epoch = torch.tensor(0.0)
        return

    def step_add_1(self, loss: torch.Tensor):
        '''loss: detached loss tensor'''
        self.global_step += 1
        self.progress_bar.now += 1
        self.tmp_loss_list.append(loss.to('cpu', non_blocking=True))  # 异步 D2H，不阻塞 CUDA stream
        return

    def log_step(self,loss_box:models.Loss.LossBox):
        '''log every 100 step, record all loss in tensorboard'''
        self._tb_add_scalar(f'step/main_loss', loss_box.total_loss, self.global_step)
        self.accumulate_tmp_loss()
        # for key, value in loss_box.items():
        #             self.logger.add_scalar(f'step/{key}', value, self.global_step)
        return

    def accumulate_tmp_loss(self):
        # list [], return
        if not self.tmp_loss_list:
            return
        sum_loss = torch.sum(torch.stack(self.tmp_loss_list))  # 保持 CPU tensor，不 .item() 同步
        self.train_loss_sum_epoch += sum_loss
        self.tmp_loss_list.clear()
        return sum_loss

    def log_epoch(self,eval_box:models.Loss.EvalBox,print_cli:bool=True,optimizer:torch.optim.Optimizer=None):
        self.accumulate_tmp_loss()# ensure tmp loss loss void
        self._log_epoch_TB(self.epoch,eval_box,optimizer)
        if print_cli:
            print(
                f"Epoch {self.epoch}/{self.args.num_epochs} | "
                f"Train Loss: {self.train_loss_sum_epoch/self.progress_bar.end:.4f} | "
                f"Eval Err: {eval_box.LP_err:.4f} | "
            )
        return

    def _log_epoch_TB(self, epoch:int, eval_results:models.Loss.EvalBox,optimizer:torch.optim.Optimizer=None):
        self._tb_add_scalar('epoch/Loss', self.train_loss_sum_epoch/self.progress_bar.end, epoch) # last batch loss

        # record eval results all
        eval_fields = fields(eval_results)
        # 遍历所有字段
        for field in eval_fields:
            # 获取字段名和对应的值
            field_name = field.name
            field_value = getattr(eval_results, field_name)

            # 跳过值为None的字段
            if field_value is None:
                continue
            self._tb_add_scalar(
                f"epoch/{field_name}",
                (
                    field_value.item()
                    if isinstance(field_value, torch.Tensor)
                    else field_value
                ),
                epoch,
            )
        if optimizer is not None:
            self._tb_add_scalar('epoch/LR', optimizer.param_groups[0]['lr'], epoch)
        return

    class ProgressBar:
        now:int=0
        end:int=0
        def __init__(self,end:int):
            self.end=end
            return
        def print(self):
            print(f'progress:{self.now}/{self.end}',end='\r')
            return

    def save_ckpt(
        self,
        val_acc: float,
        model: Union[torch.nn.Module, torch._dynamo.eval_frame.OptimizedModule],
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scaler: torch.cuda.amp.GradScaler = None,
        TB_backup:bool=True,
        need_history_ckpt:bool=False
    ):

        state = CkptBox(
            global_step=self.global_step,
            epoch=self.epoch,
            best_acc=self.best_acc,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict() if optimizer else None,
            scheduler_state_dict=scheduler.state_dict() if scheduler else None,
            scaler_state_dict=scaler.state_dict() if scaler else None,
        )
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        # best
        best_file_path = os.path.join(self.args.checkpoint_dir, "best.pth")
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            state.best_acc = self.best_acc
            torch.save(state, best_file_path)
            pass
        # 定义 latest 文件路径并直接保存
        latest_file_path = os.path.join(self.args.checkpoint_dir, "latest.pth")
        torch.save(state, latest_file_path)

        # 如果需要历史备份，则将 latest 文件复制一份
        if need_history_ckpt:
            step_file_path = os.path.join(
                self.args.checkpoint_dir,
                f"{self.args.model.name}_{self.global_step}.pth",
            )
            # 使用 copy2 将刚生成的 latest 复制为带 step 的版本
            shutil.copy2(latest_file_path, step_file_path)

        # backup at tensorboard dir
        if TB_backup:
            shutil.copy2(latest_file_path,self.tensorboard.log_dir)
            shutil.copy2(best_file_path,self.tensorboard.log_dir) if os.path.isfile(best_file_path) else None

        return self.best_acc

    def load_ckpt(self,model,optimizer=None,scheduler=None,scaler=None,ckpt_name:str='latest.pth') -> CkptBox:
        ckpt_dir=getattr(self.args,'checkpoint_dir',None)
        if self.args.resume is False or ckpt_dir is None:
            return None
        try:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            ckpt_box: CkptBox = torch.load(ckpt_path,weights_only=False)
        except FileNotFoundError:
            print(f"未找到 checkpoint 文件: {ckpt_path}，从头开始训练。")
            return None
        # 类型校验
        if not isinstance(ckpt_box, CkptBox):
            raise TypeError(f"Checkpoint 类型错误，期望 CkptBox，实际 {type(ckpt_box)}")

        # 恢复模型、优化器等状态（根据需要）
        model.load_state_dict(ckpt_box.model_state_dict)
        if optimizer and ckpt_box.optimizer_state_dict:
            optimizer.load_state_dict(ckpt_box.optimizer_state_dict)
        if scheduler and ckpt_box.scheduler_state_dict:
            scheduler.load_state_dict(ckpt_box.scheduler_state_dict)
        if scaler and ckpt_box.scaler_state_dict:
            scaler.load_state_dict(ckpt_box.scaler_state_dict)

        # compile model if needed
        if self.args.model.get('complile',True):
            model_cpl=torch.compile(model,mode="reduce-overhead")

        self.global_step = ckpt_box.global_step+1
        self.epoch = ckpt_box.epoch+1
        self.best_acc = ckpt_box.best_acc
        return ckpt_box

    pass

from torch.optim.lr_scheduler import _LRScheduler
import math

class OneCycleRexLR(_LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, 
                 pct_start=0.3, div_factor=25., last_epoch=-1):
        if total_steps is None:
            if epochs is None or steps_per_epoch is None:
                raise ValueError("Error: total_steps or epochs+steps_per_epoch required")
            total_steps = epochs * steps_per_epoch
        self.max_lrs = [max_lr] if isinstance(max_lr, (int, float)) else max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.warmup_steps = int(total_steps * pct_start)
        self.decay_steps = total_steps - self.warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch
        if step_num < self.warmup_steps:
            pct = step_num / self.warmup_steps
            cos_out = 0.5 * (1.0 - math.cos(math.pi * pct))
            return [(base_lr / self.div_factor) + (base_lr - base_lr / self.div_factor) * cos_out for base_lr in self.max_lrs]
        else:
            steps_in_decay = step_num - self.warmup_steps
            x = steps_in_decay / self.decay_steps 
            if x >= 1.0: return [0.0 for _ in self.max_lrs]
            numerator = 1.0 - x
            denominator = 0.5 + 0.5 * (1.0 - x)
            decay_factor = numerator / denominator
            return [base_lr * decay_factor for base_lr in self.max_lrs]

# 纠正不合法的bbox和顶点label
def validate_xyxy_bbox(bbox: Union[List[float], Tuple[float, ...]]) -> Tuple[List[float], bool]:
    """
    检查并纠正 bbox 格式为 [x_min, y_min, x_max, y_max]。
    
    返回:
        (corrected_bbox, is_valid):
        - corrected_bbox: 纠正后的合法 bbox
        - is_valid: 原始 bbox 是否已经合法
    """
    if len(bbox) != 4:
        raise ValueError("Bbox 必须包含 4 个元素 [x1, y1, x2, y2]")
    
    x1, y1, x2, y2 = bbox
    
    # 原始数据是否合法 (左上角必须在右下角的左上方)
    is_valid = (x1 <= x2) and (y1 <= y2)
    
    # 纠正逻辑：提取真正的最小/最大值
    corrected_bbox = [
        min(x1, x2), # x_min (左边缘)
        min(y1, y2), # y_min (上边缘)
        max(x1, x2), # x_max (右边缘)
        max(y1, y2)  # y_max (下边缘)
    ]
    
    return corrected_bbox, is_valid

def validate_xyxy_bbox_tensor(bboxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    检查并纠正 Tensor 格式的 bbox 为 [x_min, y_min, x_max, y_max]。
    
    参数:
        bboxes: 形状为 [B, 4] 的 Tensor，最后一维顺序为 [x1, y1, x2, y2]
        
    返回:
        (corrected_bboxes, is_valid_mask):
        - corrected_bboxes: [B, 4] 纠正后的合法坐标
        - is_valid_mask: [B] 布尔类型 Tensor，表示原始每个 bbox 是否合法
    """
    if bboxes.ndim != 2 or bboxes.shape[-1] != 4:
        raise ValueError(f"输入 Tensor 形状必须为 [B, 4], 当前形状: {bboxes.shape}")

    # 1. 拆分坐标通道
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # 2. 计算原始合法性掩码 (所有条件必须同时满足)
    # 对于车牌识别，通常要求 x1 < x2 且 y1 < y2
    is_valid_mask = (x1 <= x2) & (y1 <= y2)

    # 3. 纠正逻辑：使用 torch.min/max 在 B 维度上批量处理
    # stack 会在 dim=-1 重新组合成 [B, 4]
    corrected_bboxes = torch.stack([
        torch.min(x1, x2), # x_min
        torch.min(y1, y2), # y_min
        torch.max(x1, x2), # x_max
        torch.max(y1, y2)  # y_max
    ], dim=-1)

    return corrected_bboxes, is_valid_mask


def validate_xy_vertex(keypoints: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], bool]:
    """
    检查并纠正 4 个顶点为 [左上, 右上, 右下, 左下] 的顺时针顺序。
    
    返回:
        (corrected_keypoints, is_valid):
        - corrected_keypoints: 纠正后的顶点列表
        - is_valid: 原始顶点顺序是否已经正确
    """
    if len(keypoints) != 4:
        raise ValueError("Keypoints 必须包含 4 个顶点坐标")

    # 1. 计算几何中心
    cx = sum(p[0] for p in keypoints) / 4.0
    cy = sum(p[1] for p in keypoints) / 4.0

    # 2. 定义 atan2 夹角计算
    def get_angle(point: Tuple[float, float]) -> float:
        return math.atan2(point[1] - cy, point[0] - cx)

    # 3. 生成纠正后的标准顺时针坐标
    corrected_keypoints = sorted(keypoints, key=get_angle)

    # 4. 对比原始数据
    is_valid = (keypoints == corrected_keypoints)

    return corrected_keypoints, is_valid

import os
import torch
from torchvision.utils import save_image
from typing import Optional, Union

def save_img_tensor(
    img_tensor: torch.Tensor, 
    filename: str = "output.jpg", 
    dir_name: str = "tmp", 
    max_images: Optional[int] = 8,
    nrow: Optional[int] = None,
    normalize: bool = True
):
    """
    通用的张量保存函数
    :param img_tensor: 输入张量 (C, H, W) 或 (B, C, H, W)
    :param filename: 文件名
    :param dir_name: 保存目录
    :param max_images: 最多保存多少张（防止 Batch 太大撑爆显存或图片过长）
    :param nrow: 每行显示多少张，默认自动计算
    :param normalize: 输入tensor是否归一化到了 [0, 1]
    """
    # 1. 自动创建目录
    os.makedirs(dir_name, exist_ok=True)
    save_path = os.path.join(dir_name, filename)

    # 2. 维度健壮性处理
    # 如果是单张图 (C, H, W)，增加一个 Batch 维度变成 (1, C, H, W)
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    # 3. 截取图片数量
    if max_images is not None:
        img_tensor = img_tensor[:max_images]

    # 4. 自动计算 nrow (默认为图片总数的平方根，看起来更规整)
    if nrow is None:
        nrow = int(len(img_tensor) ** 0.5) if len(img_tensor) > 1 else 1

    # 5. 保存
    save_image(img_tensor.detach().cpu(), save_path, nrow=nrow, normalize=normalize)
    print(f"Saved: {save_path} (Shape: {list(img_tensor.shape)})")

# ---------eval used functions---------

def tensor_to_cv2(tensor_img):
    """将torch tensor [3, H, W] 转换为 cv2 BGR图像"""
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def heatmap_to_jet(heatmap, base_img=None):
    """
    将热力图转换为Jet色彩空间（红色表示热）
    Args:
        heatmap: [H, W] numpy array, 值在0-1之间
        base_img: [H, W, 3] BGR图像（可选），用于overlay
    Returns:
        jet_heatmap: Jet色彩图
        masked_img: overlay在base_img上的结果（如果提供了base_img）
    """
    # 归一化到0-255
    heatmap_norm = np.uint8(255 * np.clip(heatmap, 0, 1))
    jet_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    masked_img = None
    if base_img is not None:
        # 确保base_img是BGR格式且大小匹配
        if jet_heatmap.shape[:2] != base_img.shape[:2]:
            jet_heatmap = cv2.resize(jet_heatmap, (base_img.shape[1], base_img.shape[0]))
        masked_img = cv2.addWeighted(jet_heatmap, 0.5, base_img, 0.5, 0)
    
    return jet_heatmap, masked_img

from datasets.chars import CHARS
def decode_lp(logits, target_length:int =8, char_list:List=CHARS):
    """
    通用解码LP logits到字符串
    
    Args:
        logits: Tensor 或 Numpy 数组，形状应为 (class_num, target_length) 或 (target_length, class_num)
        target_length: 预期的车牌字符长度 (如 8)
        char_list: 字符集列表 (CHARS)
        
    Returns:
        解码后的字符串
    """
    class_num = len(char_list)
    
    # 转换为 Tensor 统一处理维度逻辑
    if not isinstance(logits, torch.Tensor):
        logits = torch.from_numpy(np.array(logits))

    # 获取当前 shape
    shape = logits.shape
    
    # --- 维度检查与对齐 ---
    if logits.dim() == 2:
        if shape[0] == target_length and shape[1] == class_num:
            # 形状为 [L, C]
            idxs = logits.argmax(dim=1)
        elif shape[1] == target_length and shape[0] == class_num:
            # 形状为 [C, L]
            idxs = logits.argmax(dim=0)
        else:
            idxs = logits.argmax(dim=0)
            # print(f"维度不匹配: 期望包含长度 {target_length} 和类别数 {class_num}，"
            #     f"但得到的是 {list(shape)}")
            # raise ValueError(
            #     f"维度不匹配: 期望包含长度 {target_length} 和类别数 {class_num}，"
            #     f"但得到的是 {list(shape)}"
            # )
    elif logits.dim() == 1:
        # 如果已经是索引序列，直接检查长度
        if len(logits) != target_length:
            raise ValueError(f"索引序列长度 {len(logits)} 与期望长度 {target_length} 不符")
        idxs = logits
    else:
        raise ValueError(f"不支持的维度数: {logits.dim()}")

    # --- 转换与解码 ---
    idxs = idxs.cpu().numpy().astype(int)
    
    # 过滤掉超出字符集范围的索引（防御性编程）
    chars = [char_list[i] for i in idxs if 0 <= i < class_num]
    
    return "".join(chars)

def draw_boxes_on_image(img, boxes, box_format='xyxy', norm_type='01', color=(0, 255, 0), thickness=2):
    """
    在图像上绘制多个框
    
    Args:
        img: [H, W, 3] BGR图像
        boxes: [N, 4] 框的数据
        box_format: 'xyxy' (默认) 或 'cxcywh' (中心点和宽高)
        norm_type: 坐标归一化类型。可选 'pixel' (实际像素), '01' ([0,1]区间), '-11' ([-1,1]区间)
        color: RGB/BGR颜色
        thickness: 线条宽度
    """
    img_out = img.copy()
    H, W = img.shape[:2]
    
    for box in boxes:
        # 1. 统一转换为 xyxy 格式的原始数值
        if box_format == 'xyxy':
            x1_raw, y1_raw, x2_raw, y2_raw = box
        elif box_format == 'cxcywh': 
            cx, cy, w, h = box
            x1_raw = cx - w / 2
            y1_raw = cy - h / 2
            x2_raw = cx + w / 2
            y2_raw = cy + h / 2
        else:
            raise ValueError("不支持的 box_format，请使用 'xyxy' 或 'cxcywh'")

        # 2. 根据 norm_type 反归一化到真实像素坐标
        if norm_type == '-11':
            x1 = (x1_raw + 1) / 2.0 * W
            y1 = (y1_raw + 1) / 2.0 * H
            x2 = (x2_raw + 1) / 2.0 * W
            y2 = (y2_raw + 1) / 2.0 * H
        elif norm_type == '01':
            x1 = x1_raw * W
            y1 = y1_raw * H
            x2 = x2_raw * W
            y2 = y2_raw * H
        else: # 'pixel'
            x1, y1, x2, y2 = x1_raw, y1_raw, x2_raw, y2_raw

        # 3. 使用你的合法化函数纠正大小顺序
        corrected_bbox, _ = validate_xyxy_bbox([x1, y1, x2, y2])
        x1_c, y1_c, x2_c, y2_c = corrected_bbox

        # 4. 限制在图像边界内并转换为整型进行绘制
        x1_f = int(max(0, min(W, x1_c)))
        y1_f = int(max(0, min(H, y1_c)))
        x2_f = int(max(0, min(W, x2_c)))
        y2_f = int(max(0, min(H, y2_c)))

        cv2.rectangle(img_out, (x1_f, y1_f), (x2_f, y2_f), color, thickness)
        
    return img_out

class CudaPrefetcher:
    """
    用独立 CUDA stream 将 H→D 传输与 GPU compute 完全重叠。

    原理：
      - 标准 DataLoader 每次 next() 都是：collate → pin_memory → H→D → 返回
      - Prefetcher 在 GPU 执行当前 batch 时，同步在独立 stream 里把下一个 batch
        搬到 GPU，compute 结束后直接用，H→D 延迟被完全隐藏。

    用法：
        prefetcher = tools.CudaPrefetcher(train_loader, device)
        for batch in prefetcher:          # batch 已在 GPU
            optimizer.zero_grad(set_to_none=True)
            ...                           # 不再需要 mv_to_device
    """
    def __init__(self, loader, device: str):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._next = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._it = iter(self.loader)
        self._preload()          # 预取第一个 batch
        return self

    def _preload(self):
        try:
            raw = next(self._it)
        except StopIteration:
            self._next = None
            return
        # 在独立 stream 里异步 H→D，不阻塞默认 stream
        with torch.cuda.stream(self.stream):
            raw.mv_to_device(self.device, non_blocking=True)
        self._next = raw

    def __next__(self):
        if self._next is None:
            raise StopIteration
        # 等独立 stream 的 H→D 完成后才允许默认 stream 消费
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next
        self._preload()          # 立刻异步预取下一个，不等待
        return batch


def draw_vertex4_on_image(img, vertices_list, norm_type='01', color=(0, 255, 0), thickness=2):
    """
    在图像上绘制四边形顶点
    
    Args:
        img: [H, W, 3] BGR图像
        vertices_list: 形状为 [N, 4, 2] 的数组，N个框，每个框4个(x, y)顶点
        norm_type: 'pixel', '01', 或 '-11'
        color: RGB/BGR颜色
        thickness: 线条宽度
    """
    img_out = img.copy()
    H, W = img.shape[:2]
    
    for vertices in vertices_list:
        # 转为 numpy 数组方便批量运算
        pts = np.array(vertices, dtype=np.float32)
        
        # 处理归一化映射
        if norm_type == '-11':
            pts[:, 0] = (pts[:, 0] + 1) / 2.0 * W
            pts[:, 1] = (pts[:, 1] + 1) / 2.0 * H
        elif norm_type == '01':
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            
        # 限制在图像边界内
        pts[:, 0] = np.clip(pts[:, 0], 0, W)
        pts[:, 1] = np.clip(pts[:, 1], 0, H)
        
        # OpenCV 的 polylines 需要 int32 类型的坐标点，且形状为 [1, 4, 2]
        pts = np.int32(pts)
        
        # 绘制闭合多边形
        cv2.polylines(img_out, [pts], isClosed=True, color=color, thickness=thickness)
        
    return img_out

