import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

from .baseBlock import *

class Backbone:
    '''backbone define here'''
    @staticmethod
    def resnet18_strid16(d_out:int=75,pre_weights=models.ResNet18_Weights.DEFAULT):
        resnet18 = models.resnet18(weights=pre_weights)
        resnet18 = nn.Sequential(*list(resnet18.children())[:-3])
        conv_k1 = nn.Sequential(
            nn.Conv2d(256, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        return nn.Sequential(resnet18,conv_k1)
    @staticmethod
    def resnet34_strid16(d_out: int = 75, pre_weights=models.ResNet34_Weights.DEFAULT):
        # 加载预训练的 ResNet34
        resnet34 = models.resnet34(weights=pre_weights)
        
        # 同样去掉最后三层 (AvgPool, FC, 以及 Layer4)
        # ResNet34 的 Layer3 输出维度也是 256
        resnet34 = nn.Sequential(*list(resnet34.children())[:-3])
        
        conv_k1 = nn.Sequential(
            nn.Conv2d(256, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        return nn.Sequential(resnet34, conv_k1)
    @staticmethod
    def resnet34_coordAdd_strid16(d_out: int = 75, pre_weights=models.ResNet34_Weights.DEFAULT, coord_method='add'):
        # 加载预训练的 ResNet34
        resnet34 = models.resnet34(weights=pre_weights)
        
        # =============== 插入点 ===============
        # 将原有 conv1 替换为带有 coord 的小插件 
        # coord_method 可选 'add' (+) 或 'concat' (rgb+xy 5 dim) 方便做 Ablation
        resnet34.conv1 = CoordPluginWrapper(resnet34.conv1, method=coord_method)
        # =====================================

        # 同样去掉最后三层 (AvgPool, FC, 以及 Layer4)
        # ResNet34 的 Layer3 输出维度也是 256
        resnet34 = nn.Sequential(*list(resnet34.children())[:-3])
        
        conv_k1 = nn.Sequential(
            nn.Conv2d(256, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        return nn.Sequential(resnet34, conv_k1)
    @staticmethod
    def resnet50_strid16(d_out:int=75,pre_weights=models.ResNet50_Weights.DEFAULT):
        resnet50 = models.resnet50(weights=pre_weights)
        resnet50 = nn.Sequential(*list(resnet50.children())[:-3])
        conv_k1 = nn.Sequential(
                    nn.Conv2d(1024, d_out, 1),
                    nn.BatchNorm2d(d_out),
                    nn.ReLU(True),
        )
        return nn.Sequential(resnet50,conv_k1)
    @staticmethod
    def mobilenetv2_stride16(d_out: int = 75, pre_weights=models.MobileNet_V2_Weights.DEFAULT):
        # 1. 加载预训练的 MobileNetV2
        mobilenet = models.mobilenet_v2(weights=pre_weights)
        
        # 2. 提取特征层
        # features[0:14] 的总 Stride 是 16
        # 此时输出的特征图通道数 (in_channels) 是 96
        backbone = nn.Sequential(*list(mobilenet.features)[:14])
        
        # 3. 这里的输入通道数要改为 96 (MobileNetV2 第13层输出是96)
        conv_k1 = nn.Sequential(
            nn.Conv2d(96, d_out, kernel_size=1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(inplace=True),
        )
        
        return nn.Sequential(backbone, conv_k1)
    pass
    @staticmethod
    def repvgg_stride16(d_out: int = 75):
        # 使用 repvgg_a0，它是最轻量级的版本
        # features 包含 4 个 stage
        model = timm.create_model('repvgg_a0', pretrained=True, features_only=True)
        
        # RepVGG 的 stage 2 输出 stride 8, stage 3 输出 stride 16
        # 我们取到 stage 3 (索引为 3)
        # repvgg_a0 stage 3 的输出通道数通常是 192
        in_channels = 192 
        
        # 封装一下
        class WrappedRepVGG(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                # 返回的是一个 list，我们取第 4 个元素 (index 3)
                return self.m(x)[3]

        backbone = WrappedRepVGG(model)
        
        conv_k1 = nn.Sequential(
            nn.Conv2d(in_channels, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        return nn.Sequential(backbone, conv_k1)
    @staticmethod
    def shufflenetv2_stride16(d_out: int = 75, pre_weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT):
        # 1. 加载预训练的 ShuffleNetV2 (1.0x 版本)
        shufflenet = models.shufflenet_v2_x1_0(weights=pre_weights)
        
        # 2. 提取特征层到 Stage 3
        # 我们需要：conv1 -> maxpool -> stage2 -> stage3
        # 在 torchvision 实现中，这些是独立的属性
        backbone = nn.Sequential(
            shufflenet.conv1,
            shufflenet.maxpool,
            shufflenet.stage2,
            shufflenet.stage3
        )
        
        # 3. 这里的输入通道数是 232 (ShuffleNetV2 x1.0 Stage 3 输出)
        # 如果你用的是 x0.5 版本，这里需要改为 102
        in_channels = 232 
        
        conv_k1 = nn.Sequential(
            nn.Conv2d(in_channels, d_out, kernel_size=1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(inplace=True),
        )
        
        return nn.Sequential(backbone, conv_k1)
    
    @staticmethod
    def convnext_pico_stride16(d_out: int = 75, pretrained: bool = True):
        # 1. 加载 Pico 变体，同样取 index 2 (stride 16)
        backbone = timm.create_model(
            'convnext_pico', 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(2,) 
        )
        
        # 2. 包装器，确保输出 Tensor
        class FeatureWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)[0]

        # 3. ConvNeXt Pico 的 Stage 2 (Stride 16) 通道数是 256
        # (Nano 是 320, Tiny 是 384)
        conv_k1 = nn.Sequential(
            nn.Conv2d(256, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        
        return nn.Sequential(FeatureWrapper(backbone), conv_k1)

    def convnext_nano_stride16(d_out: int = 75, pretrained: bool = True):
        # 1. 使用 out_indices 直接指定输出哪几层
        # Index 0=s4, 1=s8, 2=s16, 3=s32
        # 我们只需要索引 2 (stride 16)
        backbone = timm.create_model(
            'convnext_nano', 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(2,) 
        )
        
        # 2. 定义一个包装类，确保输出的是 Tensor 而不是 timm 默认返回的 list
        class FeatureWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                # features_only=True 返回的是一个 list，取第 0 个元素（即 index 2 的输出）
                return self.model(x)[0]

        # 3. ConvNeXt Nano 在 Stride 16 (Stage 2) 的输出通道数是 320
        conv_k1 = nn.Sequential(
            nn.Conv2d(320, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        
        return nn.Sequential(FeatureWrapper(backbone), conv_k1)
    
    @staticmethod
    def convnext_tiny_strid16(d_out: int = 75, pre_weights=models.ConvNeXt_Tiny_Weights.DEFAULT):
        model = models.convnext_tiny(weights=pre_weights)
        
        # ConvNeXt 的 features 包含 8 个模块：
        # 0: stem(s=4), 1: stage1, 2: downsample(s=8), 3: stage2, 
        # 4: downsample(s=16), 5: stage3, 6: downsample(s=32), 7: stage4
        # 截取前 6 个模块（[:6]），刚好输出 stride 16 的特征图，通道数为 384
        backbone = nn.Sequential(*list(model.features.children())[:6])
        
        conv_k1 = nn.Sequential(
            nn.Conv2d(384, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        return nn.Sequential(backbone, conv_k1)
    @staticmethod
    def convnextv2_tiny_stride16(d_out: int = 75, pretrained: bool = True):
        # 使用 timm 加载模型
        # features_only=True: 允许提取中间层特征
        # out_indices=(2,): 对应 Stage 3 的输出 (Stride 16)
        # ConvNeXt 的索引通常为: 0->s4, 1->s8, 2->s16, 3->s32
        backbone = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in1k', 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(2,)
        )
        
        # 自动获取该层的输出通道数 (对于 Tiny 来说是 384)
        in_channels = backbone.feature_info[2]['num_chs']
        conv_k1 = nn.Sequential(
            nn.Conv2d(in_channels, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        
        # 注意：timm 的 features_only 模型返回的是一个 List (即使只有一个输出)
        # 我们需要包装一下，让其在 forward 时返回 Tensor 供 conv_k1 使用
        class FeatureWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)[0] # 提取 List 中的第一个特征图

        return nn.Sequential(FeatureWrapper(backbone), conv_k1)
    @staticmethod
    def convnext_small_strid16(d_out: int = 75, pre_weights=models.ConvNeXt_Small_Weights.DEFAULT):
        model = models.convnext_small(weights=pre_weights)
        
        # 结构和 Tiny 一致，只是 stage 更深，stride 16 处的通道数依然是 384
        backbone = nn.Sequential(*list(model.features.children())[:6])
        
        conv_k1 = nn.Sequential(
            nn.Conv2d(384, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        return nn.Sequential(backbone, conv_k1)

    @staticmethod
    def efficientnet_v2_s_strid16(d_out: int = 75, pre_weights=models.EfficientNet_V2_S_Weights.DEFAULT):
        model = models.efficientnet_v2_s(weights=pre_weights)
        
        # EfficientNetV2-S 的 features:
        # 0: stem(s=2), 1-3: stage1~3逐渐降采样到 s=8
        # 4: stage4(s=16), 5: stage5(s=16, 通道数160)
        # 取前 6 个模块（[:6]），刚好保持 stride 16
        backbone = nn.Sequential(*list(model.features.children())[:6])
        
        conv_k1 = nn.Sequential(
            nn.Conv2d(160, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        return nn.Sequential(backbone, conv_k1)

    @staticmethod
    def res2net50_strid16(d_out: int = 75):
        
        # res2net50_26w_4s 是 Res2Net 论文中最经典的 50 层变体，极擅长多尺度特征
        model = timm.create_model('res2net50_26w_4s', pretrained=True)
        
        # timm 的 resnet 系列结构与 torchvision 高度一致
        # layer3 的输出对应 stride 16，因为使用了 Bottleneck，此处通道数是 1024
        backbone = nn.Sequential(
            model.conv1, model.bn1, model.act1, model.maxpool,
            model.layer1, model.layer2, model.layer3
        )
        
        conv_k1 = nn.Sequential(
            nn.Conv2d(1024, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(True),
        )
        return nn.Sequential(backbone, conv_k1)
    
    class ConvNeXt_FPN(nn.Module): # TODO del this , use timm fpn instead
        def __init__(self, d_out: int = 64, pretrained: bool = True):
            super().__init__()
            
            # 1. 加载 Backbone，提取 Stride 8, 16, 32 (对应 out_indices 1, 2, 3)
            self.backbone = timm.create_model(
                'convnext_tiny', 
                pretrained=pretrained, 
                features_only=True, 
                out_indices=(1, 2, 3) 
            )
            
            # 获取各层输出通道数 (Tiny大概是: 192, 384, 768)
            feature_info = self.backbone.feature_info
            c2_in = feature_info[0]['num_chs'] # Stride 8
            c3_in = feature_info[1]['num_chs'] # Stride 16
            c4_in = feature_info[2]['num_chs'] # Stride 32
            
            # 2. 侧边连接 (Lateral Connections) - 使用 1x1 Conv 统一通道数
            self.lat_c4 = nn.Conv2d(c4_in, d_out, 1)
            self.lat_c3 = nn.Conv2d(c3_in, d_out, 1)
            self.lat_c2 = nn.Conv2d(c2_in, d_out, 1)
            
            # 3. 平滑卷积 (Smoothing Convs) - 消除上采样带来的混叠效应
            # 为了轻量化，这里使用带有 BN 和 ReLU 的 3x3 卷积
            def smooth_layer():
                return nn.Sequential(
                    nn.Conv2d(d_out, d_out, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(d_out),
                    nn.ReLU(inplace=True)
                )
                
            self.smooth_p4 = smooth_layer()
            self.smooth_p3 = smooth_layer()
            self.smooth_p2 = smooth_layer()

        def forward(self, x):
            # 1. 提取骨干网络特征
            features = self.backbone(x)
            c2, c3, c4 = features[0], features[1], features[2] # s8, s16, s32
            
            # 2. 侧边连接降维
            p4 = self.lat_c4(c4)
            p3 = self.lat_c3(c3)
            p2 = self.lat_c2(c2)
            
            # 3. 自顶向下融合
            # p4 保持不变
            # p3 融合 p4 的上采样
            p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
            # p2 融合 p3 的上采样
            p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
            
            # 4. 平滑处理
            p4 = self.smooth_p4(p4) # Stride 32 
            p3 = self.smooth_p3(p3) # Stride 16 
            p2 = self.smooth_p2(p2) # Stride 8 
            
            # 返回一个字典，方便下游任务按需取用
            return {
                'stn_feat': p2,       # 分辨率较高 (Stride 8)，给 STN 用于寻找顶点/几何对齐
                'trans_feat': p3      # 分辨率适中 (Stride 16)，语义强，给 Transformer 做序列解码
            }
    
    class Timm_FPN(nn.Module):
        def __init__(
            self, 
            backbone_name: str = 'convnext_tiny', # 将主干网络名称作为输入参数
            d_out: int = 64, 
            pretrained: bool = True
        ):
            '''
            convnext timm list:
            convnext_atto
            convnext_femto
            convnext_pico
            convnext_nano
            convnext_tiny
            convnext_small
            convnext_base
            convnext_large
            convnext_xlarge
            '''
            super().__init__()
            
            # 1. 加载 Backbone，提取 Stride 8, 16, 32 (对应 out_indices 1, 2, 3)
            self.backbone = timm.create_model(
                model_name=backbone_name, # 使用传入的参数
                pretrained=pretrained, 
                features_only=True, 
                out_indices=(1, 2, 3) 
            )
            
            # 获取各层输出通道数 
            feature_info = self.backbone.feature_info
            c1_in = feature_info[1]['num_chs'] # Stride 8
            c2_in = feature_info[2]['num_chs'] # Stride 16
            c3_in = feature_info[3]['num_chs'] # Stride 32
            
            # 2. 侧边连接 (Lateral Connections) - 使用 1x1 Conv 统一通道数
            self.lat_c4 = nn.Conv2d(c3_in, d_out, 1)
            self.lat_c3 = nn.Conv2d(c2_in, d_out, 1)
            self.lat_c2 = nn.Conv2d(c1_in, d_out, 1)
            
            # 3. 平滑卷积 (Smoothing Convs) - 消除上采样带来的混叠效应
            def smooth_layer():
                return nn.Sequential(
                    nn.Conv2d(d_out, d_out, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(d_out),
                    nn.ReLU(inplace=True)
                )
                
            self.smooth_p4 = smooth_layer()
            self.smooth_p3 = smooth_layer()
            self.smooth_p2 = smooth_layer()

        def forward(self, x):
            # 1. 提取骨干网络特征
            features = self.backbone(x)
            c2, c3, c4 = features[0], features[1], features[2] # s8, s16, s32
            
            # 2. 侧边连接降维
            p4 = self.lat_c4(c4)
            p3 = self.lat_c3(c3)
            # p2 = self.lat_c2(c2)
            
            # 3. 自顶向下融合
            p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
            # p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
            
            # 4. 平滑处理
            p4 = self.smooth_p4(p4) # Stride 32 
            p3 = self.smooth_p3(p3) # Stride 16 
            # p2 = self.smooth_p2(p2) # Stride 8 
            
            return {
                # 'stn_feat': p2,
                'stride16': p3
            }
        pass

    class Timm_FPN_stride32(nn.Module):
        '''Same as Timm_FPN but returns stride32 feature map.
        For exDETR: backbone output is stride32 fm, smaller spatial size.
        Input img [B,3,H,W] -> output {'stride32': [B, d_out, H/32, W/32]}
        '''
        def __init__(
            self,
            backbone_name: str = 'convnext_nano',
            d_out: int = 256,
            pretrained: bool = True
        ):
            super().__init__()
            self.backbone = timm.create_model(
                model_name=backbone_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(3,)  # only stride 32
            )
            feature_info = self.backbone.feature_info
            c4_in = feature_info[3]['num_chs']  # Stride 32 channels

            self.lat_c4 = nn.Conv2d(c4_in, d_out, 1)
            self.smooth_p4 = nn.Sequential(
                nn.Conv2d(d_out, d_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(d_out),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            features = self.backbone(x)
            c4 = features[0]  # stride 32
            p4 = self.smooth_p4(self.lat_c4(c4))
            return {'stride32': p4}
        pass

    class Splittable_ConvNeXt_FPN(nn.Module):
        def __init__(self, backbone_name='convnext_tiny', d_model=76, pretrained=True):
            super().__init__()
            
            # 1. 临时创建一个 features_only 模型，专门为了动态白嫖各层通道数
            tmp_model = timm.create_model(
                model_name=backbone_name, 
                pretrained=False, 
                features_only=True, 
                out_indices=(0, 1, 2, 3) 
            )
            feature_info = tmp_model.feature_info
            
            # 动态获取各个阶段的输出通道数
            c_s4 = feature_info[0]['num_chs']  # Stage 0 (Stride 4)
            c_s8 = feature_info[1]['num_chs']  # Stage 1 (Stride 8)
            c_s16 = feature_info[2]['num_chs'] # Stage 2 (Stride 16)
            c_s32 = feature_info[3]['num_chs'] # Stage 3 (Stride 32)
            del tmp_model # 用完即弃

            # 2. 挂载完整的 Backbone 模块用于分段执行
            self.full_model = timm.create_model(backbone_name, pretrained=pretrained)
            self.stem = self.full_model.stem
            self.stage0 = self.full_model.stages[0] # 输出 s4
            self.stage1 = self.full_model.stages[1] # 输出 s8
            self.stage2 = self.full_model.stages[2] # 输出 s16
            self.stage3 = self.full_model.stages[3] # 输出 s32

            # 3. 构造 FPN (仅作用于全图特征提取阶段)
            self.lat_c4 = nn.Conv2d(c_s32, d_model, 1)
            self.lat_c3 = nn.Conv2d(c_s16, d_model, 1)
            self.lat_c2 = nn.Conv2d(c_s8, d_model, 1)
            
            def smooth_layer():
                return nn.Sequential(
                    nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(d_model),
                    nn.ReLU(inplace=True)
                )
                
            self.smooth_p4 = smooth_layer()
            self.smooth_p3 = smooth_layer()
            self.smooth_p2 = smooth_layer()

            # 4. 动态通道降维层：用于裁剪后继续下采样的纯净 s16 特征 -> d_model
            self.final_dim_reducer = nn.Conv2d(c_s16, d_model, 1)

        def forward_full_with_fpn(self, x):
            """阶段1：跑通全局 Backbone 并附加 FPN，专为 STN 预测 theta 提供特征"""
            # --- 原生特征 ---
            s4 = self.stage0(self.stem(x))
            s8 = self.stage1(s4)
            s16 = self.stage2(s8)
            s32 = self.stage3(s16)

            # --- FPN 融合 (自顶向下) ---
            p4 = self.lat_c4(s32)
            p3 = self.lat_c3(s16)
            p2 = self.lat_c2(s8)
            
            p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
            p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
            
            p4 = self.smooth_p4(p4)
            p3 = self.smooth_p3(p3) # 给 STN 用的增强 s16 特征
            p2 = self.smooth_p2(p2) 

            return {
                'raw_s4': s4,
                'raw_s8': s8,
                'raw_s16': s16,
                'fpn_s16': p3 
            }

        # --- 阶段2：断点续传 (纯净特征继续下采样) ---
        def forward_crop_s4_to_final(self, cropped_s4):
            s8 = self.stage1(cropped_s4)
            s16 = self.stage2(s8)
            return self.final_dim_reducer(s16)

        def forward_crop_s8_to_final(self, cropped_s8):
            s16 = self.stage2(cropped_s8)
            return self.final_dim_reducer(s16)

        def forward_crop_s16_to_final(self, cropped_s16):
            return self.final_dim_reducer(cropped_s16)
    pass
class Neck:
    '''neck define here'''
    @staticmethod
    def flate():
        return nn.Flatten(-2,-1)
    @staticmethod
    def STN_s16g270_ada(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(# s16
                nn.Conv2d(char_classNum,64,3,2,1), #s32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,32,3,2,1), #s64
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,16,3,2,1), #s128
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((2,9)),
                nn.Flatten(),
            )
        L_channels=36
        outSize=[6,45]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate())
    

    class Lnet_coordconv_Tr(nn.Module):
        '''coord conv+ transformer '''
        def __init__(self, in_channels=75, hidden_dim=128, num_heads=4, num_layers=2):
            super().__init__()
            
            # 1. CoordConv + 第一次降采样 (Stride 2)
            # 输入: [B, 75, 45, 72] -> 输出大致: [B, 64, 23, 36]
            # 为什么要加2? 为了拼接 x, y 坐标图
            self.conv_down1 = nn.Sequential(
                nn.Conv2d(in_channels + 2, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            
            # 2. 第二次降采样 (Stride 2)
            # 输入: [B, 64, 23, 36] -> 输出大致: [B, 128, 12, 18]
            # 此时 seq_len = 12*18 = 216，非常轻量
            self.conv_down2 = nn.Sequential(
                nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True)
            )
            
            # 3. Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim * 4, 
                dropout=0.1, 
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # 4. [CLS] Token
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
            # 5. 回归头
            self.regressor = nn.Sequential(
                nn.LayerNorm(hidden_dim), # 加个 LN 更稳
                nn.Linear(hidden_dim, 64),
                nn.ReLU(True),
                # nn.Linear(64, 6) # 输出 theta
            )
            
            self._init_weights()

        def _init_weights(self):
            # 最后一层初始化为 Identity
            # last_fc = self.regressor[-1]
            # last_fc.weight.data.zero_()
            # last_fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            
            nn.init.normal_(self.cls_token, std=0.02)

        def forward(self, fm):
            # fm: [B, 75, H, W] (e.g., 45, 72)
            B, C, H, W = fm.shape
            device = fm.device
            
            # --- Step 1: CoordConv (生成坐标并拼接) ---
            # 这一步非常重要，必须在降采样之前做，保证原始坐标精度
            y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
            x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
            x = torch.cat([fm, x_coords, y_coords], dim=1) # [B, 77, 45, 72]
            
            # --- Step 2: CNN Downsample ---
            x = self.conv_down1(x) # -> [B, 64, 23, 36]
            x = self.conv_down2(x) # -> [B, 128, 12, 18]
            
            # --- Step 3: Prepare for Transformer ---
            # Flatten spatial dimensions
            x = x.flatten(2).transpose(1, 2) # [B, 128, 216] -> [B, 216, 128]
            
            # Add CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) # [B, 217, 128]
            
            # --- Step 4: Attention & Regression ---
            x = self.transformer(x)
            
            # Take CLS token result
            theta = self.regressor(x[:, 0])
            
            return theta
    
    class Lnet_2Dsine_Tr(nn.Module):
        '''Ablation Version: 移除 CoordConv，改用 2D Sine Positional Encoding'''
        def __init__(self, in_channels=75, hidden_dim=128, num_heads=4, num_layers=2):
            super().__init__()
            
            # 1. 纯 CNN 降采样 (移除 CoordConv 的 +2)
            # 输入: [B, 75, 45, 72] -> 输出大致: [B, 64, 23, 36]
            self.conv_down1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            
            # 2. 第二次降采样
            # 输入: [B, 64, 23, 36] -> 输出大致: [B, 128, 12, 18]
            self.conv_down2 = nn.Sequential(
                nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True)
            )

            # 3. 2D Sine 位置编码
            # 注意：这里的 height 和 width 必须对应 conv_down2 输出的特征图尺寸 (12, 18)
            # (9,15) for CTP
            from .baseBlock import PosEncode
            self.pos_encoding = PosEncode.sinePosEncoding_2D(d_model=hidden_dim, height=9, width=15)
            
            # 4. Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim * 4, 
                dropout=0.1, 
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # 5. [CLS] Token
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
            # 6. 回归头
            self.regressor = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(True),
                # nn.Linear(64, 6) # 输出根据需求定义
            )
            
            self._init_weights()

        def _init_weights(self):
            nn.init.normal_(self.cls_token, std=0.02)

        def forward(self, fm):
            # fm: [B, 75, 45, 72]
            B, C, H, W = fm.shape
            
            # --- Step 1: CNN Downsample (无 CoordConv) ---
            x = self.conv_down1(fm) # [B, 64, 23, 36]
            x = self.conv_down2(x)  # [B, 128, 12, 18]
            
            # --- Step 2: Prepare for Transformer ---
            # Flatten spatial dimensions: [B, 128, 12, 18] -> [B, 216, 128]
            x = x.flatten(2).transpose(1, 2) 
            
            # --- Step 3: Add 2D Sine Position Encoding ---
            # pos_encoding 输出为 [1, seq_len, d_model]
            pe = self.pos_encoding.forward(x, batch_first=True)
            x = x + pe # 直接相加
            
            # --- Step 4: Add CLS token ---
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) # [B, 217, 128]
            
            # --- Step 5: Attention & Regression ---
            x = self.transformer(x)
            
            # 取 CLS token 的特征进行回归
            theta = self.regressor(x[:, 0])
            
            return theta
    
    class Lnet_coord_2Dsine_Tr(nn.Module):
        '''Ablation Version: 移除 CoordConv，改用 2D Sine Positional Encoding'''
        def __init__(self, in_channels=75, hidden_dim=128, num_heads=4, num_layers=2):
            super().__init__()
            
            # 1. 纯 CNN 降采样 (移除 CoordConv 的 +2)
            # 输入: [B, 75+2, 45, 72] -> 输出大致: [B, 64, 23, 36]
            # 为什么要加2? 为了拼接 x, y 坐标图
            self.conv_down1 = nn.Sequential(
                nn.Conv2d(in_channels + 2, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            
            # 2. 第二次降采样
            # 输入: [B, 64, 23, 36] -> 输出大致: [B, 128, 12, 18]
            self.conv_down2 = nn.Sequential(
                nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True)
            )

            # 3. 2D Sine 位置编码
            # 注意：这里的 height 和 width 必须对应 conv_down2 输出的特征图尺寸 (12, 18)
            # (9,15) for CTP
            from .baseBlock import PosEncode
            self.pos_encoding = PosEncode.sinePosEncoding_2D(d_model=hidden_dim, height=18, width=32)
            
            # 4. Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim * 4, 
                dropout=0.1, 
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # 5. [CLS] Token
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            
            # 6. 回归头
            self.regressor = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(True),
                # nn.Linear(64, 6) # 输出根据需求定义
            )
            
            self._init_weights()

        def _init_weights(self):
            nn.init.normal_(self.cls_token, std=0.02)

        def forward(self, fm: torch.Tensor):
            # fm: [B, 75, 45, 72]
            B, C, H, W = fm.shape
            device = fm.device
            y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
            x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
            x = torch.cat([fm, x_coords, y_coords], dim=1) # [B, 77, 45, 72]
            # --- Step 1: CNN Downsample (无 CoordConv) ---
            x = self.conv_down1(x) # [B, 64, 23, 36]
            x = self.conv_down2(x) # [B, 128, 12, 18]
            
            # --- Step 2: Prepare for Transformer ---
            # Flatten spatial dimensions: [B, 128, 12, 18] -> [B, 216, 128]
            pe = self.pos_encoding.forward_2d(x )
            x = x + pe # 直接相加 [B,C,H,W]
            x = x.flatten(2).transpose(1, 2) # B,N,C
            
            # --- Step 4: Add CLS token ---
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) # [B, 217, 128]
            
            # --- Step 5: Attention & Regression ---
            x = self.transformer(x)
            
            # 取 CLS token 的特征进行回归
            theta = self.regressor(x[:, 0])
            
            return theta
            
    class Lnet_biGRU(nn.Module):
        '''for stn at stride 16 fm'''
        def __init__(self,char_classNum:int=75) -> None:
            super().__init__()
            self.L_net_cnn = nn.Sequential(
                nn.Conv2d(char_classNum, 64, 3, 2, 1), 
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, 2, 1), 
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 16, 3, 2, 1), 
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 6, kernel_size=1),
                nn.BatchNorm2d(6),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((2, 9)),
                nn.Flatten(start_dim=2),
            )
            
            # B,N,C input biGRU
            self.biGRU=nn.GRU(input_size=6,hidden_size=16,num_layers=2,batch_first=True,bidirectional=True)
            return
        def forward(self,fm):
            '''fm: B,C,Y,X'''
            x = self.L_net_cnn(fm) #B,C,N
            x=x.permute(0,2,1) # B,N,C
            x, memory = self.biGRU.forward(x) # B,N,2*C ;  2*layer,B,C
            theta_latten=memory.permute(1,0,2).flatten(1) # B,2*layer*C=64
            # theta_latten=x[:,8,:]
            return theta_latten # B,C=64
        pass
    
    class Lnet_coord_biGRU(nn.Module):
        '''结合了 CoordConv 思想的 Lnet_biGRU'''
        def __init__(self, char_classNum: int = 75) -> None:
            super().__init__()
            
            # 1. 添加坐标层
            self.add_coords = AddCoords()
            
            # 2. 修改 CNN 部分
            # 注意：由于添加了 X 和 Y 两个坐标通道，第一个卷积层的输入通道数需要 +2
            self.L_net_cnn = nn.Sequential(
                nn.Conv2d(char_classNum + 2, 64, kernel_size=3, stride=2, padding=1), 
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1), 
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), 
                nn.BatchNorm2d(16),
                nn.ReLU(True),

                nn.Conv2d(16, 6, kernel_size=1),
                nn.BatchNorm2d(6),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((2, 9)),
                nn.Flatten(start_dim=2),
            )
            
            # 3. biGRU 部分保持不变
            # 输入维度是 CNN 输出的通道数 (6)，隐层 16，双向，2层
            self.biGRU = nn.GRU(
                input_size=6, 
                hidden_size=16, 
                num_layers=2, 
                batch_first=True, 
                bidirectional=True
            )

        def forward(self, fm):
            '''fm: (B, C, Y, X)'''
            # 先注入坐标信息
            x = self.add_coords(fm)     # (B, C+2, Y, X)
            
            # 经过 CNN 提取特征
            x = self.L_net_cnn(x)       # (B, 6, N) 其中 N = 2 * 9 = 18
            
            # 准备输入 GRU
            x = x.permute(0, 2, 1)      # (B, N, 6)
            
            # x: (B, N, 32), memory: (4, B, 16) -> (num_layers * num_directions, B, hidden_size)
            x, memory = self.biGRU(x) 
            
            # 提取最后的状态作为空间变换或分类的 latent vector
            # 结果维度: B, (2*2*16) = 64
            theta_latten = memory.permute(1, 0, 2).reshape(x.shape[0], -1) 
            
            return theta_latten # (B, 64)
    
    @staticmethod
    def STN_s16_ada_GRU_g270(char_classNum:int=75,stn_detach:bool=True):
        L_net=Neck.Lnet_biGRU(char_classNum)
        L_channels=64
        outSize=[6,45]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate())
    @staticmethod
    def STN_s16_ada_GRU_2sup_g270(char_classNum:int=75,stn_detach:bool=True):
        L_net=Neck.Lnet_biGRU(char_classNum)
        L_channels=64
        outSize=[6,45]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return stn
    @staticmethod
    def STN_s16_ada_GRU_2sup_g8_32(char_classNum:int=75,stn_detach:bool=True):
        L_net=Neck.Lnet_biGRU(char_classNum)
        L_channels=64
        outSize=[8,32]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(0,0),p2=(19,45))
        stn = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return stn
    @staticmethod
    def STN_s16_ada_coord_GRU_2sup_g8_32(char_classNum:int=75,stn_detach:bool=True):
        L_net=Neck.Lnet_coord_biGRU(char_classNum)
        L_channels=64
        outSize=[8,32]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(0,0),p2=(19,45))
        stn = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return stn
    @staticmethod
    def STN_s16_TR_2sup_g8_32(char_classNum:int=75,stn_detach:bool=True):
        L_net=Neck.Lnet_coordconv_Tr(char_classNum)
        L_channels=64
        outSize=[8,32]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(0,0),p2=(19,45))
        stn = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return stn
    @staticmethod
    def STN_s16_2Dsine_TR_2sup_g8_32(char_classNum:int=75,stn_detach:bool=True):
        L_net=Neck.Lnet_2Dsine_Tr(char_classNum)
        L_channels=64
        outSize=[8,32]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(0,0),p2=(19,45))
        stn = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return stn
    @staticmethod
    def STN_s16_coord_2Dsine_TR_2sup_g8_32(char_classNum:int=75,stn_detach:bool=True):
        L_net=Neck.Lnet_coord_2Dsine_Tr(char_classNum)
        L_channels=64
        outSize=[8,32]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(0,0),p2=(19,45))
        stn = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return stn

    def STNROI_s16_TR_2sup_g8_32(char_classNum:int=75,stn_detach:bool=True):
        L_net=Neck.Lnet_coordconv_Tr(char_classNum)
        L_channels=64
        outSize=[8,32]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(0,0),p2=(19,45))
        stn = STN_ROI(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return stn
    @staticmethod
    def STN_s16g8_32_ada(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(# s16
                nn.Conv2d(char_classNum,64,3,2,1), #s32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,32,3,2,1), #s64
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,16,3,2,1), #s128
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((2,9)),
                nn.Flatten(),
            )
        L_channels=36
        outSize=[8,32]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return stn
    
    @staticmethod
    def STN_s16g8_32_ada_coord(char_classNum: int = 75, stn_detach: bool = True):
        L_net = nn.Sequential(
            # 1. 注入坐标信息
            AddCoords(), 
            
            # 2. 第一层卷积的输入通道数需要 +2 (char_classNum + x_coord + y_coord)
            nn.Conv2d(char_classNum + 2, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # 保持输出通道和尺寸逻辑一致
            nn.Conv2d(16, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((2, 9)),
            nn.Flatten(),
        )
        
        # L_channels 依然是 2 * 2 * 9 = 36，保持不变以确保 theta 映射逻辑一致
        L_channels = 36 
        outSize = [8, 32]
        
        # 保持同样的初始化参数
        theta_0 = STN.gen_Theta_0(inSize=(19, 45), p1=(6, 0), p2=(12, 45))
        
        stn = STN(L_net, L_channels, outSize, detach_fm2lnet=stn_detach, theta_0=theta_0)
        return stn

    @staticmethod
    def STN_s32_coord_2Dsine_TR_2sup_g4_16(char_classNum: int = 256, stn_detach: bool = True):
        '''STN for stride-32 fm. Input fm [B, char_classNum, 4, 16] from 128x512 img.
        Output corrected fm [B, char_classNum, 4, 16].
        Uses Lnet_coord_2Dsine_Tr adapted for small spatial size.'''
        L_net = Neck.Lnet_coord_2Dsine_Tr_s32(char_classNum)
        L_channels = 64
        outSize = [4, 16]
        theta_0 = STN.gen_Theta_0(inSize=(4, 16), p1=(0, 0), p2=(4, 16))
        stn = STN(L_net, L_channels, outSize, detach_fm2lnet=stn_detach, theta_0=theta_0)
        return stn

    class Lnet_coord_2Dsine_Tr_s32(nn.Module):
        '''Localization net for stride-32 fm [B, in_channels, 4, 16].
        Adapted from Lnet_coord_2Dsine_Tr for smaller spatial dims.'''
        def __init__(self, in_channels: int = 256, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
            super().__init__()
            # coord concat + single downsample step (2x) -> [B, hidden_dim, 2, 8]
            self.conv_down1 = nn.Sequential(
                nn.Conv2d(in_channels + 2, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(True)
            )
            # pos encoding for [2, 8]
            from .baseBlock import PosEncode
            self.pos_encoding = PosEncode.sinePosEncoding_2D(d_model=hidden_dim, height=18, width=32)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=0.1,
                activation='gelu', batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.regressor = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(True),
            )
            nn.init.normal_(self.cls_token, std=0.02)

        def forward(self, fm: torch.Tensor):
            B, C, H, W = fm.shape
            device = fm.device
            y_coords = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
            x_coords = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
            x = torch.cat([fm, x_coords, y_coords], dim=1)
            x = self.conv_down1(x)  # [B, hidden_dim, 2, 8]
            pe = self.pos_encoding.forward_2d(x)
            x = x + pe
            x = x.flatten(2).transpose(1, 2)  # [B, 16, hidden_dim]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = self.transformer(x)
            return self.regressor(x[:, 0])

    @staticmethod
    def STN_s16g270_ada(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(# s16
                nn.Conv2d(char_classNum,64,3,2,1), #s32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,32,3,2,1), #s64
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,16,3,2,1), #s128
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((2,9)),
                nn.Flatten(),
            )
        L_channels=36
        outSize=[6,45]
        theta_0=STN.gen_Theta_0(inSize=(19,45),p1=(6,0),p2=(12,45))
        stn = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)
        return nn.Sequential(stn,Neck.flate())
    @staticmethod
    def STN_ada_s8s16g66(char_classNum:int=75,stn_detach:bool=True):
        L_net=nn.Sequential(#s8 fm[128,37,90]
            nn.Conv2d(128,64,3,2,1), #s16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,3,2,1), #s32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,16,3,2,1), #s64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,8,3,2,1), #s128
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 2, kernel_size=1), # channel down=>fm[2,2,9]
            nn.BatchNorm2d(2),
            nn.AdaptiveAvgPool2d((2,9)),
            nn.ReLU(True),
            nn.Flatten(),
        )
        L_channels=36
        outSize=[12,90]
        theta_0=STN.gen_Theta_0(inSize=(37,90),p1=(12,0),p2=(24,90))
        stnS8 = STN(L_net,L_channels,outSize,detach_fm2lnet=stn_detach,theta_0=theta_0)

        L_net=nn.Sequential(#[6,45] =s3=> [2,15]
            nn.Conv2d(char_classNum,32,5,3,2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32,2,1),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Flatten(),
        )
        L_channels=60
        outSize=[3,22]
        stnS16=STN(L_net,L_channels,outSize)
        return nn.Sequential(stnS8,stnS16,Neck.flate())
    class STN_ada_s8_s16_g264(nn.Module):
        '''STN at stride 8 and 16 with GRU latten
        refine stn crop size '''
        def __init__(self,char_classNum:int=75,stn_detach:bool=True) -> None:
            super().__init__()
            # stn at stride 8
            L_net_s8=nn.Sequential(#s8 fm[128,37,90]
            nn.Conv2d(128,64,3,2,1), #s16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,3,2,1), #s32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,16,3,2,1), #s64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,8,3,2,1), #s128
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 2, kernel_size=1), # channel down=>fm[2,2,9]
            nn.BatchNorm2d(2),
            nn.AdaptiveAvgPool2d((8,9)),
            nn.ReLU(True),
            nn.Flatten(),
        )
            L_channels_s8=2*8*9
            outSize_s8=[48,90]
            theta_0_s8=STN.gen_Theta_0(inSize=(37,90),p1=(12,0),p2=(24,90))
            self.stnS8 = STN(L_net_s8,L_channels_s8,outSize_s8,detach_fm2lnet=stn_detach,theta_0=theta_0_s8)
            # stn at stride 16
            L_net_s16=nn.Sequential(#[24,45] =s3=> [8,15]
            nn.Conv2d(char_classNum,32,5,3,2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32,2,1),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Flatten(),
        )
            L_channels_s16=2*8*15
            outSize_s16=[12,22]
            self.stnS16=STN(L_net_s16,L_channels_s16,outSize_s16,detach_fm2lnet=stn_detach)
            return
        def forward(self,fm):
            return '请在外部自行调用两个stn'
    pass


from .detr_TR import Transformer_src_tgt as TR_src_tgt

class Head:
    '''head define here'''
    class TR_ar(nn.Module):
        '''
        autoregressive transformer.
        pos encoder, trasformer strcture defined here.
        '''
        def __init__(self, char_classNum:int=75,fm_len:int=16,LP_len:int=8,nhead:int=2,nEnLayers:int=2,nDelayers:int=2) -> None:
            super().__init__()
            self.d_model,self.FM_len,self.LP_len,self.nhead=char_classNum,fm_len,LP_len,nhead
            # pos encoding
            self.posEncode_en = PosEncode.learnPosEncoding(
                self.d_model, max_len=self.FM_len
            )
            self.posEncode_de = PosEncode.learnPosEncoding(
                self.d_model, max_len=self.LP_len
            )
            # detr at attn transformer
            self.transformer = TR_src_tgt(
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=nEnLayers,
                num_decoder_layers=nDelayers,
                dim_feedforward=self.d_model * 2,
            )
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.LP_len)
            self.register_buffer('tgt_mask', tgt_mask)
            return

        def forward(self,fm,tgt):
            '''
            input fm:B,C,N tgt:B,N,C
            output logits:B,C,N
            '''
            fm=fm.permute(2,0,1)#N,B,C
            tgt=tgt.permute(1,0,2)#N,B,C
            en_pos,de_pos=self.posEncode_en(fm),self.posEncode_de(tgt)
            tgt_mask=self.tgt_mask if self.training else self.tgt_mask[0:tgt.shape[0],0:tgt.shape[0]]
            logits,_=self.transformer.forward(src=fm,tgt=tgt,en_pos_embed=en_pos,de_pos_embed=de_pos,tgt_mask=tgt_mask)
            return logits.permute(1,2,0) #N,B,C=>B,C,N
        pass
    class Ext3taskHead(TR_ar):
        def __init__(self, d_model:int=75,fm_len:int=16,LP_len:int=8,nhead:int=2,nEnLayers:int=2,nDelayers:int=2) -> None:
            super().__init__(d_model,fm_len,LP_len,nhead,nEnLayers,nDelayers)
            # denoise task
            # self.tgt_add_noise()
            
            # prior task
            param_data = torch.empty(self.LP_len, self.d_model)
            nn.init.kaiming_uniform_(param_data) 
            self.prior_Q = nn.Parameter(param_data)

            # attention mask
            self.tgt_mask=self.get_custom_tgt_mask(block_len=self.LP_len)
            # self.register_buffer('tgt_mask', self.get_custom_tgt_mask(block_len=self.LP_len))

            return
        
        @staticmethod
        def get_custom_tgt_mask(block_len=8):
            '''3 task tgt mask 
            block_len: 每个子任务的长度 (Main, Noise, Prior)

            ----------------------
            |Main | Noise | Prior|
            ----------------------
            Main | Causal|  -    |  -
            Noise| Causal| Causal|  -
            Prior| Causal|  -    | Full

            '''
            sz = block_len * 3
            # 初始化为全遮蔽
            mask = torch.full((sz, sz), float('-inf'))
            
            # 定义基础组件
            causal = torch.tril(torch.ones(block_len, block_len), diagonal=0).bool() # 下三角为True即可见
            full_mask = torch.zeros(block_len, block_len).bool() # 全0即不遮蔽
            
            # 1. 填充对角线块 (Main, Noise, Prior 内部)
            # Main: 0:8, 0:8 -> Causal
            mask[0:block_len, 0:block_len].masked_fill_(causal, 0.0)
            
            # Noise: 8:16, 8:16 -> Causal
            mask[block_len:2*block_len, block_len:2*block_len].masked_fill_(causal, 0.0)
            
            # Prior: 16:24, 16:24 -> Full
            mask[2*block_len:3*block_len, 2*block_len:3*block_len] = 0.0
            
            # 2. 填充纵向依赖 (Noise -> Main, Prior -> Main)
            # 根据你的描述，这两块都是 Causal 关系
            # Noise 对 Main 的观察 (行 8:16, 列 0:8)
            mask[block_len:2*block_len, 0:block_len].masked_fill_(causal, 0.0)
            
            # Prior 对 Main 的观察 (行 16:24, 列 0:8)
            mask[2*block_len:3*block_len, 0:block_len].masked_fill_(causal, 0.0)
            
            return mask
        
        @staticmethod
        def tgt_add_noise(tgt: torch.Tensor, noise_ratio: float = 0.9):
            '''
            tgt: B, N, C (one-hot)
            return tgt_noisy: B, N, C (one-hot)
            '''
            
            # 整体克隆，防止修改原数据
            tgt_noisy = tgt.clone()
            B, N, C = tgt_noisy.shape
  

            # --- 批量操作核心逻辑 ---
            
            # 3. 为每个 batch 随机选择一个 [1, N-1] 之间的索引
            # 形状: (B,)
            random_indices = torch.randint(1, N , (B,), device=tgt.device)
            
            # 4. 为每个 batch 随机生成一个新的类别索引
            # 形状: (B,)
            new_classes = torch.randint(0, C, (B,), device=tgt.device)
            
            # 5. 准备 batch 维度的索引
            # 形状: (B,) -> [0, 1, 2, ..., B-1]
            batch_indices = torch.arange(B, device=tgt.device)

            # 6. 批量将选定位置清零
            # 利用高级索引直接访问 B 个特定位置
            tgt_noisy[batch_indices, random_indices, :] = 0
            
            # 7. 批量设置新的 one-hot 值
            tgt_noisy[batch_indices, random_indices, new_classes] = 1

            # p 概率添加噪声
            mask = torch.rand(1, device=tgt.device) > noise_ratio
            return torch.where(mask, tgt, tgt_noisy)
        
        def forward_infer(self,fm,tgt):
            '''
            input fm:B,C,N tgt:B,N,C
            output logits:B,C,N
            '''
            return Head.TR_ar.forward(self,fm,tgt) # B,C,N
            
        def forward_infer_1by1(self,fm):
            return
        def forward_train(self,fm,tgt):
            '''use 3 task tgt gen
            input fm:B,C,N tgt:B,N,C
            output logits:B,C,N
            3 times output: main, denoise, prior
            '''
            B = tgt.shape[0]
            prior_Q_batch = self.prior_Q.unsqueeze(0).expand(B, -1, -1)
            tgt_extended = torch.cat((tgt,self.tgt_add_noise(tgt),prior_Q_batch),dim=1) #B,N,C
            logits = Head.TR_ar.forward(self,fm,tgt_extended) # B,C,N
            # 将 logits 沿着 (N) 均匀拆分为 3 块
            logits_main, logits_denoise, logits_prior = torch.split(logits, self.LP_len, dim=2)# B,C,N
            return logits_main, logits_denoise, logits_prior
        def forward(self,fm,tgt):
            '''
            input fm:B,C,N tgt:B,N,C
            output logits:B,C,N
            '''
            
            if self.training:
                return self.forward_train(fm,tgt)
            else:
                return self.forward_infer(fm,tgt)
            
    pass

class Upsample:
    class UNetDecoder(nn.Module):
        '''TODO fix bug : not sigmoid active final'''
        def __init__(self, in_c, out_c=3):
            super().__init__()
            # Stage 1: up to H/8 (16x64)
            # Skip from Layer2 (128 ch)
            self.up1 = up_block(in_c, 64)
            self.fusion1 = nn.Sequential(
                nn.Conv2d(64 + 128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            # Stage 2: up to H/4 (32x128)
            # Skip from Layer1 (64 ch)
            self.up2 = up_block(64, 32)
            self.fusion2 = nn.Sequential(
                nn.Conv2d(32 + 64, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(True)
            )
            # Stage 3: up to H/2 (64x256)
            # Skip from Conv1 (64 ch)
            self.up3 = up_block(32, 16)
            self.fusion3 = nn.Sequential(
                nn.Conv2d(16 + 64, 16, 3, 1, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            )
            # Stage 4: up to H (128x512)
            self.up4 = up_block(16, out_c)

        def forward(self, x, skips):
            # skips: [f_layer2, f_layer1, f_conv1]
            f_l2, f_l1, f_conv1 = skips
            
            x = self.up1(x)
            if x.shape[-2:] != f_l2.shape[-2:]:
                x = F.interpolate(x, size=f_l2.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, f_l2], dim=1)
            x = self.fusion1(x)
            
            x = self.up2(x)
            if x.shape[-2:] != f_l1.shape[-2:]:
                x = F.interpolate(x, size=f_l1.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, f_l1], dim=1)
            x = self.fusion2(x)
            
            x = self.up3(x)
            if x.shape[-2:] != f_conv1.shape[-2:]:
                x = F.interpolate(x, size=f_conv1.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, f_conv1], dim=1)
            x = self.fusion3(x)
            
            x = self.up4(x)
            return x
    
    class UNetDecoder_res_pixshuff(nn.Module):
        def __init__(self, in_c, out_c=3):
            super().__init__()
            # Stage 1: up to H/8 (16x64)
            # Skip from Layer2 (128 ch)
            self.up1 = up_block_pixshuffle_res(in_c, 64)
            self.fusion1 = nn.Sequential(
                nn.Conv2d(64 + 128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            # Stage 2: up to H/4 (32x128)
            # Skip from Layer1 (64 ch)
            self.up2 = up_block_pixshuffle_res(64, 32)
            self.fusion2 = nn.Sequential(
                nn.Conv2d(32 + 64, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(True)
            )
            # Stage 3: up to H/2 (64x256)
            # Skip from Conv1 (64 ch)
            self.up3 = up_block_pixshuffle_res(32, 16)
            self.fusion3 = nn.Sequential(
                nn.Conv2d(16 + 64, 16, 3, 1, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            )
            # Stage 4: up to H (128x512)
            self.up4 = up_block_pixshuffle_res(16, 8)
            self.conv=nn.Sequential(
                nn.Conv2d(8,out_c,1),
                nn.Sigmoid()
            )
            
            

        def forward(self, x, skips):
            # skips: [f_layer2, f_layer1, f_conv1]
            f_l2, f_l1, f_conv1 = skips
            
            x = self.up1(x)
            if x.shape[-2:] != f_l2.shape[-2:]:
                x = F.interpolate(x, size=f_l2.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, f_l2], dim=1)
            x = self.fusion1(x)
            
            x = self.up2(x)
            if x.shape[-2:] != f_l1.shape[-2:]:
                x = F.interpolate(x, size=f_l1.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, f_l1], dim=1)
            x = self.fusion2(x)
            
            x = self.up3(x)
            if x.shape[-2:] != f_conv1.shape[-2:]:
                x = F.interpolate(x, size=f_conv1.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, f_conv1], dim=1)
            x = self.fusion3(x)
            
            x:torch.Tensor = self.up4(x)
            x=self.conv(x)
            return x
    
    def simple_up16_res_pixshuff( in_c: int, out_c: int = 3):
        '''2,H*16,W*16'''
        
        return nn.Sequential(
                up_block_pixshuffle_res(in_c, 64),
                up_block_pixshuffle_res(64, 32),
                up_block_pixshuffle_res(32, 16),
                up_block_pixshuffle_res(16, 8), # HW*16
                nn.Conv2d(8,out_c,1),
                nn.Sigmoid(),
            )
    def simple_up16( in_c: int, out_c: int = 3):
        '''2,H*16,W*16 '''
        return nn.Sequential(
                up_block(in_c, 64),
                up_block(64, 32),
                up_block(32, 16),
                up_block(16, 8), # HW*16
                nn.Conv2d(8,out_c,1),
                nn.Sigmoid(),
            )
    def simple_up16_decnn( in_c: int, out_c: int = 3):
        '''2,H*16,W*16 '''
        return nn.Sequential(
                up_block_deconv(in_c, 64),
                up_block_deconv(64, 32),
                up_block_deconv(32, 16),
                up_block_deconv(16, 8), # HW*16
                nn.Conv2d(8,out_c,1),
                nn.Sigmoid(),
            )

    
    pass
