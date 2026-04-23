import torch


def giou_loss(pred: torch.Tensor, tgt: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Pure-PyTorch GIoU loss，无 boolean-mask scatter，inductor 可完整编译。
    pred / tgt: [..., 4]  xyxy 格式，坐标已归一化到 [0,1]。
    等价替换 torchvision.ops.generalized_box_iou_loss。
    """
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = tgt.unbind(-1)

    # 交集
    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # 各自面积
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + 1e-6

    iou = inter / union

    # 最小外接矩形
    cx1 = torch.minimum(px1, tx1)
    cy1 = torch.minimum(py1, ty1)
    cx2 = torch.maximum(px2, tx2)
    cy2 = torch.maximum(py2, ty2)
    enclosing = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-6

    giou = iou - (enclosing - union) / enclosing
    loss = 1.0 - giou

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss  # 'none'


def generalized_box_iou_compilable(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch pairwise GIoU 矩阵，等价替换 torchvision.ops.generalized_box_iou。
    boxes1: [N, 4] xyxy,  boxes2: [M, 4] xyxy
    返回: [N, M]
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    b1 = boxes1.unsqueeze(1).expand(N, M, 4)  # [N, M, 4]
    b2 = boxes2.unsqueeze(0).expand(N, M, 4)  # [N, M, 4]

    b1x1, b1y1, b1x2, b1y2 = b1.unbind(-1)
    b2x1, b2y1, b2x2, b2y2 = b2.unbind(-1)

    ix1 = torch.maximum(b1x1, b2x1)
    iy1 = torch.maximum(b1y1, b2y1)
    ix2 = torch.minimum(b1x2, b2x2)
    iy2 = torch.minimum(b1y2, b2y2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area1 = (b1x2 - b1x1).clamp(min=0) * (b1y2 - b1y1).clamp(min=0)
    area2 = (b2x2 - b2x1).clamp(min=0) * (b2y2 - b2y1).clamp(min=0)
    union = area1 + area2 - inter + 1e-6

    iou = inter / union

    cx1 = torch.minimum(b1x1, b2x1)
    cy1 = torch.minimum(b1y1, b2y1)
    cx2 = torch.maximum(b1x2, b2x2)
    cy2 = torch.maximum(b1y2, b2y2)
    enclosing = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-6

    return iou - (enclosing - union) / enclosing  # [N, M]


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def validate_xyxy_bbox(bboxes: torch.Tensor):
    """
    检查 xyxy 格式的 bbox 是否合法，如果不合法就进行修正。

    参数:
        bboxes (Tensor): 形状为 [..., 4] 的张量，每个 bbox 是 [x0, y0, x1, y1] 格式。

    返回:
        Tuple[Tensor, Tensor]: 
            - 一个 bool 类型的张量，表示每个 bbox 是否原本合法。
            - 一个修正后的 bbox 张量，仍为 xyxy 格式。
    """
    x0, y0, x1, y1 = bboxes.unbind(-1)

    valid_x = x0 <= x1
    valid_y = y0 <= y1
    is_valid = valid_x & valid_y
    if is_valid.all() :
        return is_valid,bboxes
    # 修正非法的 bbox
    corrected_x0 = torch.min(x0, x1)
    corrected_y0 = torch.min(y0, y1)
    corrected_x1 = torch.max(x0, x1)
    corrected_y1 = torch.max(y0, y1)

    corrected = torch.stack([corrected_x0, corrected_y0, corrected_x1, corrected_y1], dim=-1)

    return is_valid, corrected
