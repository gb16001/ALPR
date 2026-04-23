# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transformer_at_attn class repacked by CiT
----------------------------
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers

"""
import copy
import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

def gen_sineembed_for_position(pos_tensor, d_model=256):
    """
    严格遵循 Conditional/DAB-DETR 论文逻辑的 Sine Embedding 生成。
    
    逻辑核心:
    1. 位置编码 (PE) 只应该表示空间中心位置 (cx, cy)。
    2. 宽高 (w, h) 是调制参数 (Modulation)，不应在此处被编码为 Sine 波。
    3. 无论输入是 2D 还是 4D 坐标，输出维度必须严格等于 d_model。
    
    Args:
        pos_tensor: [..., 4] 包含 (cx, cy, w, h) 或 [..., 2] (cx, cy)
        d_model: 输出的 embedding 维度 (例如 256)
        
    Returns:
        pos_embed: [..., d_model] 仅包含 x 和 y 的位置编码
                   前 d_model/2 为 x 的编码，后 d_model/2 为 y 的编码
    """
    # 强制要求 d_model 是偶数，保证 x 和 y 平分
    assert d_model % 2 == 0, "d_model must be divisible by 2"
    
    scale = 2 * math.pi
    
    # 核心修正：dim_t 的长度严格基于 d_model // 2
    # 我们只通过 x 和 y 来填满 d_model，没有 w 和 h 的位置
    half_dim = d_model // 2
    dim_t = torch.arange(half_dim, dtype=torch.float32, device=pos_tensor.device)
    
    # 按照 Attention is All You Need 的标准频率公式
    # 温度 T = 10000
    dim_t = 10000 ** (2 * (dim_t // 2) / half_dim)

    # 提取 x 和 y (无论输入是 2 维还是 4 维，我们只认前两维作为 Position)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale

    # 生成 embedding
    # [N, B, 1] / [dim_t] -> [N, B, half_dim]
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    
    # sin/cos 交叉堆叠
    # 0::2 取偶数位(sin), 1::2 取奇数位(cos)
    # stack 后 dim=3, flatten 后回到 [N, B, half_dim]
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    
    # 拼接：严格的 [PE(x), PE(y)]
    # 结果维度: [N, B, d_model]
    pos = torch.cat((pos_y, pos_x), dim=2)
    
    return pos


def inverse_sigmoid(x, eps=1e-5):
    """sigmoid的反函数"""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class MLP(nn.Module):
    """Simple MLP module"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, padding_mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        padding_mask = padding_mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=padding_mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=padding_mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

class Transformer_src_tgt(Transformer):
    '''forward need src:encoder input; tgt:decoder input'''
    def forward(self, src,tgt, en_pos_embed, de_pos_embed, tgt_mask=None,padding_mask=None):
        '''
        src,tgt: N,B,C
        '''
        # # flatten NxCxHxW to HWxNxC
        # bs, c, h, w = src.shape
        # src = src.flatten(2).permute(2, 0, 1)
        # en_pos_embed = en_pos_embed.flatten(2).permute(2, 0, 1)
        # de_pos_embed = de_pos_embed.unsqueeze(1).repeat(1, bs, 1)
        # padding_mask = padding_mask.flatten(1)

        # tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=padding_mask, pos=en_pos_embed)
        hs = self.decoder.forward(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=padding_mask,
            pos=en_pos_embed,
            query_pos=de_pos_embed,
        )
        return hs, memory
    pass 

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayer_DAB(nn.Module):
    """DAB DETR Decoder Layer
    
    通过concat方式在query和key中添加bbox PE：
    - query: [q_content, query_sine_embed] concat → d_model*2
    - key: [k_content, k_pos] concat → d_model*2
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, rm_self_attn=False):
        super().__init__()
        
        # Self-Attention
        if not rm_self_attn:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
        
        self.rm_self_attn = rm_self_attn
        
        # Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        # Cross-attention输入维度加倍 (d_model*2)
        from .baseBlock import MultiheadAttention
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, 
                                                vdim=d_model)
        
        self.nhead = nhead
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed: Optional[Tensor] = None,
                is_first: bool = False):
        """
        Args:
            tgt: [N, B, d_model] decoder target
            memory: [L, B, d_model] encoder output
            query_pos: [N, B, d_model] query position embedding (from MLP head)
            query_sine_embed: [N, B, d_model] sine position embedding (from bbox)
            pos: [L, B, d_model] encoder position embedding
        """
        
        # ========== Self-Attention =============
        if not self.rm_self_attn:
            q_content = self.sa_qcontent_proj(tgt)
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)
            
            q = q_content + q_pos
            k = k_content + k_pos
            
            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
        
        # ========== Cross-Attention =============
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        k_pos = self.ca_kpos_proj(pos)
        
        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape
        
        # 第一层或keep_query_pos时加入query_pos
        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content
        
        # DAB: concat方式加入sine embedding和pos
        # query: [q, query_sine_embed] concat → d_model*2
        # key: [k, k_pos] concat → d_model*2
        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, 
                                                 n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.repeat_interleave(repeats=bs, dim=1).view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)
        
        tgt2 = self.cross_attn(query=q, key=k, value=v, 
                              attn_mask=memory_mask,
                              key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # ========== FFN =============
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerDecoder_DAB(nn.Module):
    """DAB DETR Decoder - 支持bbox迭代更新和超长序列PE循环复制"""
    
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_len=8, decoder_input_len=8, bbox_embed_diff_each_layer=False):
        """
        Args:
            decoder_layer: TransformerDecoderLayer_DAB instance
            num_layers: 层数
            d_model: 模型维度
            query_len: bbox查询长度 (LP_len=8)
            decoder_input_len: decoder实际输入长度（可以 > query_len）
            bbox_embed_diff_each_layer: 每层是否使用不同的bbox_embed
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.d_model = d_model
        self.query_len = query_len
        self.decoder_input_len = decoder_input_len
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        
        # Bbox回归头：预测相对偏移
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList(
                [MLP(d_model, d_model, 4, 3) for _ in range(num_layers)]
            )
        else:
            self.bbox_embed = MLP(d_model, d_model, 4, 3)
        
        # Query position embedding投影头
        self.ref_point_head = MLP(d_model // 2 * 2, d_model, d_model, 2)  # 输入: d_model (sine embed)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None):
        """
        Args:
            tgt: [decoder_input_len, B, d_model]
            memory: [L, B, d_model] encoder output
            refpoints_unsigmoid: [query_len, B, 4] 初始bbox (unsigmoid)
            pos: [L, B, d_model] encoder position embedding
        
        Returns:
            output: [decoder_input_len, B, d_model]
            references: list of [query_len, B, 4] 每层更新后的bbox
        """
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        
        for layer_id, layer in enumerate(self.layers):
            # 从bbox生成query position embedding
            obj_center = reference_points[..., :4]  # [query_len, B, 4]
            
            # 生成sine embedding
            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model)
            # [query_len, B, d_model]
            
            # 通过MLP投影
            query_pos = self.ref_point_head(query_sine_embed)
            # [query_len, B, d_model]
            
            # 处理超长序列：循环复制query_pos到decoder_input_len
            if self.decoder_input_len > self.query_len:
                # 循环复制
                query_sine_embed_extended = query_sine_embed.repeat(
                    (self.decoder_input_len + self.query_len - 1) // self.query_len, 1, 1
                )[:self.decoder_input_len]
                query_pos_extended = query_pos.repeat(
                    (self.decoder_input_len + self.query_len - 1) // self.query_len, 1, 1
                )[:self.decoder_input_len]
            else:
                query_sine_embed_extended = query_sine_embed
                query_pos_extended = query_pos
            
            # Decoder layer forward
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos_extended,
                query_sine_embed=query_sine_embed_extended,
                is_first=(layer_id == 0)
            )
            
            # Bbox迭代更新 (只在前query_len个token上进行)
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    bbox_delta = self.bbox_embed[layer_id](output[:self.query_len])
                else:
                    bbox_delta = self.bbox_embed(output[:self.query_len])
                
                # 添加到前一个参考点
                bbox_delta[..., :4] += inverse_sigmoid(reference_points)
                new_reference_points = bbox_delta[..., :4].sigmoid()
                
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()
            
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return [
                torch.stack(intermediate).transpose(1, 2),  # [num_layers, B, decoder_input_len, d_model]
                torch.stack(ref_points).transpose(1, 2),   # [num_layers, B, query_len, 4]
            ]
        
        return output.unsqueeze(0)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
