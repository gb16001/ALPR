import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import math
import torch.nn.init as init
##### coord ops.
class CoordPluginWrapper(nn.Module):
    def __init__(self, original_conv1, method='add'):
        super().__init__()
        self.method = method
        out_channels = original_conv1.out_channels

        if self.method == 'add':
            # 保留原生预训练层
            self.conv1 = original_conv1
            # 小插件：通过 1x1 Conv 将 2 dim (XY) 映射到相应的 dim (比如 ResNet 这里的 64)
            self.coord_proj = nn.Conv2d(2, out_channels, kernel_size=1, bias=False)
            # 初始化为0，确保刚加载预训练权重时，“+”上去的 coord 信息为 0，不破坏预训练特征
            nn.init.zeros_(self.coord_proj.weight)
        elif self.method=='add5':
            # 保留原生预训练层
            self.conv1 = original_conv1
            # 小插件：通过 1x1 Conv 将 5 dim (RGB XY) 映射到相应的 dim (比如 ResNet 这里的 64)
            self.coord_proj = nn.Conv2d(
                in_channels=5,
                out_channels=out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None,
            )
            # 初始化为0，确保刚加载预训练权重时，“+”上去的 coord 信息为 0，不破坏预训练特征
            nn.init.zeros_(self.coord_proj.weight)

        elif self.method == 'extend':
            # 直接拓展第一层conv,将 rgb+xy = 5 dim 映射到相应的 dim (64)
            self.conv1 = nn.Conv2d(
                in_channels=5, 
                out_channels=out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None
            )
            # 将预训练的 RGB 权重赋给前3个通道
            self.conv1.weight.data[:, :3, :, :] = original_conv1.weight.data
            # 后2个通道 (XY) 初始化为 0
            self.conv1.weight.data[:, 3:, :, :] = 0
            if original_conv1.bias is not None:
                self.conv1.bias.data = original_conv1.bias.data

    def forward(self, x:torch.Tensor):
        B, C, H, W = x.shape

        if self.method == 'add':
            # 先过原有 conv1 -> 得到特征图 [B, 64, H/2, W/2]
            out = self.conv1(x)
            # 生成与 feature map 分辨率匹配的 coords (ResNet conv1 有下采样)
            _, _, out_H, out_W = out.shape
            xy = get_xy_coords(B, out_H, out_W, x.device)
            # 映射后，使用“+ 的形式”加上去
            return out + self.coord_proj(xy)
        elif self.method == 'add5':
            out = self.conv1(x)
            # 生成与输入图片分辨率匹配的 coords
            xy = get_xy_coords(B, H, W, x.device)
            # 拼接成 5 dim
            x_5dim = torch.cat([x, xy], dim=1)
            return out + self.coord_proj(x_5dim)

        elif self.method == 'extend':
            # 生成与输入图片分辨率匹配的 coords
            xy = get_xy_coords(B, H, W, x.device)
            # 拼接成 5 dim
            x_5dim = torch.cat([x, xy], dim=1)
            # 送入修改过 in_channels=5 的 conv1
            return self.conv1(x_5dim)

def get_xy_coords(batch_size, h, w, device):
    """
    独立计算归一化到 [-1, 1] 的 2D XY 坐标网格。
    该函数不产生梯度，供复用。
    """
    y_loc = torch.linspace(-1, 1, h, device=device)
    x_loc = torch.linspace(-1, 1, w, device=device)
    # indexing='ij' 生成 [H, W] 形状的网格
    y_grid, x_grid = torch.meshgrid(y_loc, x_loc, indexing='ij')
    
    # 堆叠为 [2, H, W] (通道 0 是 x, 通道 1 是 y)
    xy = torch.stack([x_grid, y_grid], dim=0)
    # 扩展为 batch 形式: [B, 2, H, W]
    xy = xy.unsqueeze(0).expand(batch_size, -1, -1, -1)
    
    return xy.detach() # 阻断梯度，不参与网络参数更新

class AddCoords(nn.Module):
    """
    CoordConv 的核心：为输入特征图添加坐标通道
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # 生成 y 轴坐标：从 -1 到 1
        yy_range = torch.linspace(-1, 1, height, device=x.device)
        # 生成 x 轴坐标：从 -1 到 1
        xx_range = torch.linspace(-1, 1, width, device=x.device)

        # 转换为网格矩阵 (height, width)
        yy_grid, xx_grid = torch.meshgrid(yy_range, xx_range, indexing='ij')

        # 扩展维度以匹配输入 (1, 1, height, width)
        yy_grid = yy_grid.expand(batch_size, 1, height, width)
        xx_grid = xx_grid.expand(batch_size, 1, height, width)

        # 在通道维度 (dim=1) 拼接坐标
        x = torch.cat([x, xx_grid, yy_grid], dim=1)
        return x

class STN(nn.Module):
    @staticmethod
    def gen_Theta_0(inSize,p1,p2):
        """
        outSize:(y,x)
        p1,p2:(y,x)rectfangle clip down=object box pos in origin img
        """
        assert len(inSize)==len(p1)==len(p2)
        p1=[2*p1[i]/inSize[i]-1 for i in range(len(p1))]
        p2=[2*p2[i]/inSize[i]-1 for i in range(len(p2))]
        return STN.gen_Theta0_relative(p1, p2) 

    @staticmethod
    def gen_Theta0_relative(p1, p2):
        '''
        p1,p2:(y,x)~(-1,1).rectfangle crop down=object box relative pos in origin img
        '''
        y, x = [(p2[i] + p1[i]) / 2 for i in range(len(p1))]
        b, a = ((p2[i] - p1[i]) / 2 for i in range(len(p1)))
        return [a,0,x,
                0,b,y]#grid point:(x,y)～[-1,1]

    def __init__(
        self,
        L_net: nn.Module,
        L_channels: int,
        outSize,
        theta_0=[1, 0, 0, 0, 1, 0],
        fc_loc_init="zeros",
        tanh_active: bool = False,
        detach_fm2lnet: bool= True
    ):
        """
        STN=localization net+grid affine+sample
        L_net:fm=>(bz,L_channels)=(fc_loc)=>theta:(bz,6)
        grid affine:theta=>pos grid
        theta_0:initial theta
        """
        super(STN, self).__init__()
        self.outSize= outSize 
        # fm=>(bz,L_channels)
        self.localization,self.detach_fm2lnet = L_net,detach_fm2lnet
        # [bz,L_channels]=>[bz,6]
        self.fc_loc = nn.Sequential(
            nn.Linear(L_channels, 3 * 2), 
            nn.Tanh() if tanh_active else nn.Identity(),
        )
        if fc_loc_init=="normal":
            init.normal_(self.fc_loc[0].weight, mean=0.0, std=1e-1)
        elif fc_loc_init=='kaiming':
            init.kaiming_normal_(self.fc_loc[0].weight, nonlinearity='relu')
        elif fc_loc_init=="xavier":
            init.xavier_normal_(self.fc_loc[0].weight, gain=1.0)
        elif fc_loc_init=="zeros":
            self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor(theta_0, dtype=torch.float))
        return

    def forward(
        self,
        x,
        need_affine: bool = False,
        detach_fm2lnet: bool = True,
        detach_fm2sampler: bool = False,
        detach_grid: bool = False,
    ):
        '''if need_affine: return {'fm','grid':[B,H,W,2],'theta':[B,2,3]}
          else return fm'''
        # predict theta
        xs = self.localization(x.detach() if (self.detach_fm2lnet or detach_fm2lnet) else x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)# B,2,3
        # gen affine_grid
        grid = F.affine_grid(theta, [*x.shape[0:2],*self.outSize],align_corners=False)# B,H,W,2
        # sample
        x = F.grid_sample(
            x.detach() if detach_fm2sampler else x,
            grid.detach() if detach_grid else grid,
            align_corners=False,
        )

        if need_affine:
            return {'fm':x,'grid':grid, 'theta':theta}
        else:
            return x

    pass

class STN_ROI(nn.Module):
    def __init__(
        self,
        L_net: nn.Module,
        L_channels: int,
        outSize,
        # theta_0 这里建议给4个值: [tx, ty, sx, sy]
        # 默认 [0, 0, 1, 1] 代表中心位置，覆盖全图 (identity)
        theta_0=[0, 0, 1, 1], 
        detach_fm2lnet: bool = True,
        fc_loc_init="zeros",
        tanh_active: bool = False,
    ):
        """
        STN variant using ROI Pooling.
        特点：
        1. 依然生成 affine grid，用于兼容你的 Corner MSE Loss。
        2. 实际特征采样使用 ROI Pooling。
        3. 强制 theta 为 Axis-aligned（无旋转），只有平移和缩放。
        """
        super(STN_ROI, self).__init__()
        self.outSize = outSize
        self.detach_fm2lnet = detach_fm2lnet
        self.localization = L_net
        
        # 修改：输出 4 个参数 [tx, ty, sx, sy]
        # tx, ty: 中心点偏移 (-1~1)
        # sx, sy: 缩放比例 (0~1通常)
        self.fc_loc = nn.Sequential(
            nn.Linear(L_channels, 4), 
            nn.Tanh() if tanh_active else nn.Identity(),
        )

        # 初始化逻辑
        if fc_loc_init == "normal":
            init.normal_(self.fc_loc[0].weight, mean=0.0, std=1e-1)
        elif fc_loc_init == 'kaiming':
            init.kaiming_normal_(self.fc_loc[0].weight, nonlinearity='relu')
        elif fc_loc_init == "xavier":
            init.xavier_normal_(self.fc_loc[0].weight, gain=1.0)
        elif fc_loc_init == "zeros":
            self.fc_loc[0].weight.data.zero_()
        
        # 初始化 bias
        # 注意：如果传入的是6维的theta_0，这里需要转换，或者手动传入4维的
        if len(theta_0) == 6:
            # 尝试从 identity [1,0,0, 0,1,0] 提取 [0,0,1,1]
            # 对应 [tx, ty, sx, sy] -> [0, 0, 1, 1]
            theta_0_4 = [theta_0[2], theta_0[5], theta_0[0], theta_0[4]]
            self.fc_loc[0].bias.data.copy_(torch.tensor(theta_0_4, dtype=torch.float))
        else:
            self.fc_loc[0].bias.data.copy_(torch.tensor(theta_0, dtype=torch.float))

    def forward(self, x, need_affine: bool = False, detach_fm2sampler: bool = False,**kwargs):
        '''
        Return: 
           fm: ROI Pooled feature map
           grid: (B, H, W, 2) 标准 grid，用于你的 MSE Loss 监督
           theta: (B, 2, 3) 补全后的仿射矩阵
        '''
        # 1. 预测参数 [B, 4] -> (tx, ty, sx, sy)
        xs = self.localization(x.detach() if self.detach_fm2lnet else x)
        params = self.fc_loc(xs)
        
        tx = params[:, 0]
        ty = params[:, 1]
        sx = params[:, 2]
        sy = params[:, 3]

        B = x.shape[0]
        device = x.device

        # 2. 构造标准的 2x3 Affine Matrix (为了生成 Grid 用于监督)
        # 也就是 [[sx, 0, tx], [0, sy, ty]]
        # 这样生成的 Grid 只有缩放和平移，没有旋转
        zeros = torch.zeros_like(tx)
        theta = torch.stack([
            sx, zeros, tx,
            zeros, sy, ty
        ], dim=1).view(-1, 2, 3) # [B, 2, 3]

        # 3. 生成 Grid (你的 Loss 需要这个！)
        # 这个 Grid 的 4 个角就是该 Crop 在原图中的归一化坐标
        grid = F.affine_grid(theta, [*x.shape[0:2], *self.outSize], align_corners=False)

        # 4. 转换坐标给 ROI Pooling
        # 我们需要将归一化坐标 (tx, sx...) 转换为像素坐标 (x1, y1, x2, y2)
        # 这里的 sx, sy 代表 grid 在原图中的 span 比例
        # Box 的边界是: center +/- scale
        
        H_img, W_img = x.shape[2], x.shape[3]
        
        # 计算归一化坐标下的边界 (-1 ~ 1)
        # 注意：affine_grid 的逻辑是 input_coord = matrix @ output_coord
        # 简单来说，如果要截取 sx 大小的区域，边界大约是 tx - sx 到 tx + sx
        # (假设 sx 是半宽/半高)
        
        x1_norm = tx - sx
        x2_norm = tx + sx
        y1_norm = ty - sy
        y2_norm = ty + sy
        
        # 映射到像素坐标 (0 ~ W, 0 ~ H)
        x1 = (x1_norm + 1) * 0.5 * W_img
        x2 = (x2_norm + 1) * 0.5 * W_img
        y1 = (y1_norm + 1) * 0.5 * H_img
        y2 = (y2_norm + 1) * 0.5 * H_img
        
        # 构造 boxes [K, 5] -> [batch_idx, x1, y1, x2, y2]
        batch_idx = torch.arange(B, dtype=x.dtype, device=device).view(-1, 1)
        boxes = torch.cat([
            batch_idx, 
            x1.view(-1, 1), y1.view(-1, 1), 
            x2.view(-1, 1), y2.view(-1, 1)
        ], dim=1)

        # 5. 执行 ROI Pooling (替换 grid_sample)
        # 注意: ROI Pool 可能会因为 float 精度稍微有一点点不对齐，但一般可接受
        feature_input = x.detach() if detach_fm2sampler else x
        
        # 这里的 spatial_scale=1.0 因为我们已经手动把 boxes 乘上了 H, W
        pooled_fm = torchvision.ops.roi_pool(feature_input, boxes, self.outSize, spatial_scale=1.0)

        if need_affine:
            # 返回 grid 用于计算 MSE Loss
            return {'fm': pooled_fm, 'grid': grid, 'theta': theta}
        else:
            return pooled_fm

class STN_grid_sampler(nn.Module):
    '''TODO remove ,这种封装不好用！'''
    @staticmethod
    def gen_Theta_0(inSize,p1,p2):
        """
        outSize:(y,x)
        p1,p2:(y,x)rectfangle clip down=object box pos in origin img
        """
        assert len(inSize)==len(p1)==len(p2)
        p1=[2*p1[i]/inSize[i]-1 for i in range(len(p1))]
        p2=[2*p2[i]/inSize[i]-1 for i in range(len(p2))]
        return STN.gen_Theta0_relative(p1, p2) 

    @staticmethod
    def gen_Theta0_relative(p1, p2):
        '''
        p1,p2:(y,x)~(-1,1).rectfangle crop down=object box relative pos in origin img
        '''
        y, x = [(p2[i] + p1[i]) / 2 for i in range(len(p1))]
        b, a = ((p2[i] - p1[i]) / 2 for i in range(len(p1)))
        return [a,0,x,
                0,b,y]#grid point:(x,y)～[-1,1]

    def __init__(
        self,
        L_channels: int,
        outSize,
        theta_0=[1, 0, 0, 0, 1, 0],
        fc_loc_init="zeros",
        tanh_active: bool = False,
    ):
        """
        STN_grid_sampler=fc_loc+grid affine+sample
        fc_loc:Lnet output=>(fc_loc)=>theta:(bz,6)
        grid affine:theta=>pos grid
        theta_0:initial theta
        """
        super(STN_grid_sampler, self).__init__()
        self.outSize = outSize
        # [bz,L_channels]=>[bz,6]
        self.fc_loc = nn.Sequential(
            nn.Linear(L_channels, 3 * 2), 
            nn.Tanh() if tanh_active else nn.Identity(),
        )
        if fc_loc_init=="normal":
            init.normal_(self.fc_loc[0].weight, mean=0.0, std=1e-1)
        elif fc_loc_init=='kaiming':
            init.kaiming_normal_(self.fc_loc[0].weight, nonlinearity='relu')
        elif fc_loc_init=="xavier":
            init.xavier_normal_(self.fc_loc[0].weight, gain=1.0)
        elif fc_loc_init=="zeros":
            self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor(theta_0, dtype=torch.float))
        return

    def forward(self, fm, theta_latent, need_affine: bool = False):
        '''if need_affine: return {'fm','grid':[B,H,W,2],'theta':[B,2,3]}
          else return fm'''
        # xs = self.localization(x.detach() if self.detach else x)
        # predict theta
        theta = self.fc_loc(theta_latent)
        theta = theta.view(-1, 2, 3)# B,2,3
        # gen affine_grid
        grid = F.affine_grid(theta, [*fm.shape[0:2],*self.outSize],align_corners=False)# B,H,W,2
        # sample
        fm = F.grid_sample(fm, grid, align_corners=False)
        if need_affine:
            return {'fm':fm,'grid':grid, 'theta':theta}
        else:
            return fm
    pass

class STN_projective(nn.Module):
    Theta_identical = [1, 0, 0,
                       0, 1, 0,
                       0, 0]
    @staticmethod
    def gen_Theta0(inSize,p1,p2):
        return STN.gen_Theta_0(inSize,p1,p2)+[0,0]
    @staticmethod
    def gen_Theta0_relative(p1, p2):
        return STN.gen_Theta0_relative(p1, p2)+[0,0]
    @staticmethod
    def gen_grid(H:int, W:int):
        "gen original grid"
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))  # [H, W]
        # x, y = x.flatten(), y.flatten()  # [N]
        ones = torch.ones_like(x)  # [H,W]
        grid = torch.stack([x, y, ones], dim=0)
        return grid# [3,W,H]
    def __init__(self,L_net:nn.Module,L_channels:int,outSize,theta_0=Theta_identical,detach:bool=True):
        '''
        projective transform STN, instead of affin trans
        L_net:fm=>(bz,L_channels)=(fc_loc)=>theta:(bz,8)
        grid gen:theta=>pos grid
        theta_0:initial theta
        detach: detach L_net backprop to fm
        '''
        super().__init__()
        self.outSize,self.detach=outSize,detach
        # fm=>(bz,L_channels)
        self.localization = L_net
        # [bz,L_channels]=>[bz,8]
        self.fc_loc = nn.Sequential(
            nn.Linear(L_channels, 8)
        )
        # init theta=theta_0
        self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor(theta_0, dtype=torch.float))
        self.register_buffer('grid_origin',self.gen_grid(*self.outSize),persistent=False)
        return
    def forward(self, x):
        # predict theta
        xs = self.localization(x.detach() if self.detach else x)
        theta = self.fc_loc(xs)
        theta = torch.concat((theta, torch.ones(theta.size(0), 1,device=theta.device)), dim=-1)
        theta = theta.view(-1, 3, 3)
        # gen affine_grid
        grid = self.projective_tranform(theta,self.grid_origin)
        # sample
        x = F.grid_sample(x, grid)
        return x
    @staticmethod
    def projective_tranform(theta,grid):
        '''
        theta:(B,3,3),grid:(3,W,H)
        '''
        tgtGrid=torch.einsum('bij,jxy->bxyi',theta,grid)
        # tgtGrid=theta@grid#B,3,H,W
        # tgtGrid.permute(0,2,3,1)
        tgtGrid_2d=tgtGrid[...,:2]/tgtGrid[...,2:]#p=(x,y)/z
        return tgtGrid_2d #(B,H,W,2)
    pass 

class STN_tanh(STN):
    # TODO: remove this, too old, out of feature
    def __init__(self, L_net, L_channels, outSize, theta_0=[1, 0, 0, 0, 1, 0], detach = False):
        super().__init__(L_net, L_channels, outSize, theta_0, detach)
        self.fc_loc = nn.Sequential(
            nn.Linear(L_channels, 3 * 2,bias=False),
            nn.Tanh()
        )
        # init theta=theta_0
        self.fc_loc[0].weight.data.zero_()
        self.fc_bias=nn.Parameter(torch.tensor([1,0,0,
         0,1,0],dtype=torch.float))
        return
    def forward(self, x):
        # predict theta
        xs = self.localization(x.detach() if self.detach_fm2lnet else x)
        theta = self.fc_loc(xs)+self.fc_bias
        theta = theta.view(-1, 2, 3)
        # gen affine_grid
        grid = F.affine_grid(theta, [*x.shape[0:2],*self.outSize])
        # sample
        x = F.grid_sample(x, grid)
        return x

class STN_Predictor(nn.Module):
    """阶段1：只负责从 stride 16 特征图中预测 Theta"""
    def __init__(self, L_net: nn.Module, L_channels: int, theta_0=[1, 0, 0, 0, 1, 0], tanh_active: bool = False):
        super().__init__()
        self.localization = L_net
        self.fc_loc = nn.Sequential(
            nn.Linear(L_channels, 6), 
            nn.Tanh() if tanh_active else nn.Identity()
        )
        self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor(theta_0, dtype=torch.float))

    def forward(self, fm_s16):
        # 统一使用 s16 特征预测 theta
        xs = self.localization(fm_s16)
        theta = self.fc_loc(xs)
        return theta.view(-1, 2, 3)  # [B, 2, 3]

class STN_Sampler(nn.Module):
    """阶段2：根据 Theta 在任意尺度的特征图上执行裁剪 (采样)"""
    def __init__(self):
        super().__init__()

    def forward(self, fm, theta, target_size):
        """
        fm: 待裁剪的特征图 (可以是 s4, s8 或 s16)
        theta: [B, 2, 3]
        target_size: (H, W) 裁剪输出尺寸 (例如 s16 对应 8x32，s8 对应 16x64)
        """
        B, C = fm.shape[:2]
        # 根据当前特征图所需的尺寸生成 grid
        grid = F.affine_grid(theta, [B, C, target_size[0], target_size[1]], align_corners=False)
        # 执行采样
        cropped_fm = F.grid_sample(fm, grid, align_corners=False)
        return cropped_fm, grid

class resnetBasicBlock(nn.Module):
    def __init__(self, inChannel:int,outChannel:int,stride:int=2):
        super().__init__()
        self.basicBlock = models.resnet.BasicBlock(
            inChannel,outChannel,stride,
            downsample=nn.Sequential(
                nn.Conv2d(inChannel, outChannel, 3, stride, 1), 
                nn.BatchNorm2d(outChannel)
                ),
        )
        return
    def forward(self,x):
        return self.basicBlock(x)
class PosEncode:

    class LearnPosEncoding_2D(nn.Module):
        def __init__(self, d_model, height, width):
            """
            初始化 Learnable 2D Positional Encoding 模块。
            :param d_model: 每个位置的特征维度。
            :param max_height: 最大高度，用于定义位置编码参数的范围。
            :param max_width: 最大宽度，用于定义位置编码参数的范围。
            """
            super().__init__()
            if d_model % 2 == 0:
                dim_x=dim_y=d_model // 2
            else:
                dim_y=d_model // 2
                dim_x=d_model // 2+1
            
            self.d_model = d_model
            self.max_height = height
            self.max_width = width
            
            # 定义 learnable 的 pos_x 和 pos_y
            self.pos_x = nn.Parameter(torch.randn(width, dim_x))  
            self.pos_y = nn.Parameter(torch.randn(height, dim_y))  

        def forward(self, x):
            """
            生成并返回与输入匹配的 2D 位置编码。
            :param x: 输入张量，形状为 (seq_len, batch_size, d_model)。
            :return: 2D 位置编码，形状为 (seq_len, batch_size, d_model)。
            """
            seq_len = x.size(0)
            # batch_size = x.size(1)

            # 获取位置编码并拼接
            pos_x = self.pos_x.unsqueeze(0).repeat(self.max_height, 1, 1)  # (height, width, d_model//2)
            pos_y = self.pos_y.unsqueeze(1).repeat(1, self.max_width, 1)  # (height, width, d_model//2)
            pos_encoding = torch.cat([pos_x, pos_y], dim=-1)  # (height, width, d_model)
            
            # 展平为 (seq_len, d_model)
            pos_encoding = pos_encoding.view(-1, self.d_model)
            
            return pos_encoding[:seq_len].unsqueeze(1)  # 根据序列长度返回

    class learnPosEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000) -> None:
            super().__init__()
            self.max_len = max_len
            self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))

            return
        def forward(self,x):
            '''循环位置编码'''
            # x = x + self.pe[:x.size(0), :]
            N=x.size(0)
            if N<=self.max_len:
                return self.pe[:N]
            # import warnings
            # warnings.warn(
            #     f"Input len {N} exceeds max_len {self.max_len}, using cyclic pos encoding",
            #     category=UserWarning
            # )
            indices = torch.arange(N, device=x.device)
        
            # 2. 对索引取模，使其落回 [0, max_len - 1] 范围内
            # 例如：0%8=0, 8%8=0, 9%8=1...
            cyclic_indices = indices % self.max_len
            
            # 3. 使用索引从 pe 中取值
            # self.pe[cyclic_indices] 的形状会自动变为 (seq_len, 1, d_model)
            return self.pe[cyclic_indices]
        pass 

    class sinePosEncoding_2D(nn.Module):
        def __init__(self, d_model, height, width):
            super().__init__()
            self.height = height
            self.width = width
            self.d_model = d_model

            # Create a grid of shape (height, width)
            pos_w = torch.arange(0, width).unsqueeze(0).repeat(height, 1)
            pos_h = torch.arange(0, height).unsqueeze(1).repeat(1, width)

            # Calculate position encodings based on sine and cosine functions
            dim_w = d_model // 2
            dim_h = d_model - dim_w
            dim_w_s = dim_w // 2
            dim_w_c = dim_w - dim_w_s
            dim_h_s = dim_h // 2
            dim_h_c = dim_h - dim_h_s
            dims_w = torch.arange(0, dim_w, 1).unsqueeze(0).unsqueeze(0)
            dims_h = torch.arange(0, dim_h, 1).unsqueeze(0).unsqueeze(0)
            
            pos_w = pos_w.unsqueeze(-1).repeat(1, 1, dim_w)
            pos_h = pos_h.unsqueeze(-1).repeat(1, 1, dim_h)
            
            # Apply sin to even indices, cos to odd indices
            pos_encoding_w = torch.zeros(height, width, dim_w)
            pos_encoding_h = torch.zeros(height, width, dim_h)
            
            pos_encoding_w[:, :, 0::2] = torch.sin(pos_w * self.gen_terms(dim_w_s, dims_w))[..., 0::2]
            pos_encoding_w[:, :, 1::2] = torch.cos(pos_w * self.gen_terms(dim_w_c, dims_w))[..., 1::2]
            pos_encoding_h[:, :, 0::2] = torch.sin(pos_h * self.gen_terms(dim_h_s, dims_h))[..., 0::2]
            pos_encoding_h[:, :, 1::2] = torch.cos(pos_h * self.gen_terms(dim_h_c, dims_h))[..., 1::2]

            # Concatenate along the depth (last dimension)
            # pos_encoding shape: [height, width, d_model]
            pos_encoding = torch.cat([pos_encoding_w, pos_encoding_h], dim=-1)
            
            # --- 核心修改区 ---
            # 1. 新版 2D 存储逻辑：调整维度为 [1, d_model, height, width]，对齐 [B, C, y, x]
            pe_2d = pos_encoding.permute(2, 0, 1).unsqueeze(0)
            self.register_buffer('pe_2d', pe_2d)

            # 2. 老版 1D 存储逻辑：保持拉平状态 [height*width, d_model]，增加unsqueeze用于兼容老forward
            pe_1d = pos_encoding.view(height * width, d_model).unsqueeze(1)
            self.register_buffer('pe', pe_1d)

        def gen_terms(self, dim_sine, dims):
            return torch.exp(dims * -(math.log(10000.0) / dim_sine))

        def forward(self, x, batch_first: bool = False):
            '''
            保持旧的接口和行为不变，用于兼容期望 1D Sequence 输入的老代码。
            x: [N, B, C] if not self.batch_first
            x: [B, N, C] if self.batch_first
            '''
            if batch_first:
                _pe = self.pe.transpose(0, 1)  # 转换为 (1, seq_len, d_model)
                return _pe[:, :x.size(1), :]
            else:
                return self.pe[:x.size(0), :, :] 

        def forward_2d(self, x=None, h: int = None, w: int = None):
            '''
            全新的 2D 获取逻辑。统一用于 B, C, y, x 形状。
            可以通过传入 tensor x 自动推断尺寸，也可以直接提供 h 和 w 整数。
            返回: 形状为 [1, d_model, h, w] 的位置编码 Tensor。
            '''
            if x is not None:
                # 假设输入 x 的形状是 [B, C, H, W]
                h, w = x.shape[2], x.shape[3]
            elif h is None or w is None:
                raise ValueError("在 forward_2d 中，必须传入特征图 'x' 或指定高宽 'h', 'w'。")
            
            # 防御性编程：确保请求尺寸不超出初始化时的最大尺寸
            if h > self.height or w > self.width:
                raise ValueError(f"请求的尺寸 ({h}, {w}) 超出了初始化的最大尺寸 ({self.height}, {self.width})。")
            
            # 直接利用切片提取对应的空间范围，形状保持为 [1, C, h, w]
            return self.pe_2d[:, :, :h, :w]
    class sinePosEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000): # , dropout=0
            super().__init__()
            # self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = (
                torch.cos(position * div_term)
                if d_model % 2 == 0
                else torch.cos(position * div_term)[..., :-1]
            )
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            # x = x + self.pe[:x.size(0), :]
            # return self.dropout(x)
            return self.pe[:x.size(0), :]

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn1 = nn.GroupNorm(min(32, out_c), out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.GroupNorm(min(32, out_c), out_c)
        
        if in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.GroupNorm(min(32, out_c), out_c)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

def up_block_pixshuffle_res(in_c: int, out_c: int):
    '''使用PixelShuffle进行上采样，并接ResBlock增加复杂度'''
    return nn.Sequential(
        # 1. 维度扩张 + PixelShuffle (H,W -> 2H,2W)
        nn.Conv2d(in_c, out_c * 4, kernel_size=3, padding=1),
        nn.PixelShuffle(2),
        nn.GroupNorm(min(32, out_c), out_c),
        nn.ReLU(inplace=True),
        # 2. ResBlock 特征精修 (匹配ResNet18复杂度)
        ResBlock(out_c, out_c)
    )

'''deconv上采样基本块'''
def up_block(in_c: int, out_c: int):
    '''使用插值(Interpolation)代替转置卷积，消除棋盘效应'''
    return nn.Sequential(
        # 1. 先进行双线性插值，将尺寸放大2倍
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        # 2. 通过普通卷积学习特征，避免转置卷积的重叠问题
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        # 3. 在重建任务中，尽量使用 GroupNorm 或直接去掉 BN
        # nn.BatchNorm2d(out_c), 
        nn.GroupNorm(min(32, out_c), out_c), # 推荐使用 GN，对生成质量更友好
        nn.ReLU(inplace=True)
    )

def up_block_deconv( in_c:int, out_c:int):
    '''up *2 fm size'''
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )
class ChannelAttention(nn.Module):
    """SE-Block / Channel Attention to re-weight skip connections"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)
    pass
class HighCapacityUpsamplerBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        # 1. 调整 Skip Connection 的特征 (Attention + Compression)
        self.skip_layer = nn.Sequential(
            nn.Conv2d(skip_c, skip_c, 3, 1, 1),
            ChannelAttention(skip_c), # 关键：让网络学会“看”Skip里的哪些信息有用
            nn.BatchNorm2d(skip_c),
            nn.ReLU(True)
        )
        
        # 2. 上采样 (PixelShuffle 没问题，但要配合足够的通道)
        self.up = nn.Sequential(
            nn.Conv2d(in_c, in_c * 4, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_c, in_c, 3, 1, 1), # 平滑一下
            nn.BatchNorm2d(in_c),
            nn.ReLU(True)
        )
        
        # 3. 融合后的强力处理 (这里增加宽度)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            # 再来一个 ResBlock 增加深度
            ResBlock(out_c, out_c) 
        )

    def forward(self, x, skip):
        x = self.up(x)
        # 对齐尺寸处理... (略)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear')
        
        skip = self.skip_layer(skip)
        x = torch.cat([x, skip], dim=1)
        return self.fusion(x)


'''dab detr attention block'''
# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from codes in torch.nn
# ------------------------------------------------------------------------

"""
MultiheadAttention that support query, key, and value to have different dimensions.
Query, key, and value projections are removed.

Mostly copy-paste from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L873
and https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L4837
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import warnings
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn.modules.linear import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F

import warnings
import math

from torch._C import _infer_size, _add_docstr
from torch.nn import _reduction as _Reduction
from torch.nn.modules import utils
from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default
from torch.nn import grad
from torch import _VF
from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple
try:
    from torch.overrides import has_torch_function, handle_torch_function
except:
    from torch._overrides import has_torch_function, handle_torch_function
    pass
Tensor = torch.Tensor

from torch.nn.functional import linear, pad, softmax, dropout

class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        vdim = vdim if vdim is not None else embed_dim
        self.out_proj = Linear(vdim , vdim)

        self.in_proj_bias = None
        self.in_proj_weight = None
        self.bias_k = self.bias_v = None
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.out_proj.bias, 0.)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*\text{num_heads}, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, out_dim=self.vdim)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, out_dim=self.vdim)


def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None,
                                 out_dim: Optional[Tensor] = None
                                 ) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                multi_head_attention_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    v_head_dim = out_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    q = query * scaling
    k = key
    v = value

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == v_head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    # attn_output_weights = softmax(
    #     attn_output_weights, dim=-1)
    attn_output_weights = softmax(
            attn_output_weights - attn_output_weights.max(dim=-1, keepdim=True)[0], dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
