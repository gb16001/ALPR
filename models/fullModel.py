from .subModel import Backbone, Neck
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from dataclasses import dataclass, fields

from .detr_TR import Transformer as detr
from .detr_TR import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, MLP
from .box_ops import box_xyxy_to_cxcywh
from .subModel import Upsample
from .baseBlock import *


@dataclass
class InferBox:
    """
        LPs_logits[B, n_char, L]: LP string logits (best slot for exDETR, decoder output for mtLPRtr)
        plateType_logits[B, n_class]: plate type classification
        bboxes_hat[B, 4]: axis-aligned bbox xyxy in [0,1]
        decoder_verteces[B, 4, 2]: 4 predicted vertices (x,y) in [0,1] from DETR decoder (exDETR only)
        STN_verteces[B, 4, 2]: 4 corner vertices (x,y) from STN grid (STN localization output)
        theta[B, 2, 3]: STN affine matrix
        denoise_LPs_logits, prior_LPs_logits: auxiliary task outputs
        reconstructed_img: MAE reconstruction (mtLPRtr only)
    """

    LPs_logits: Optional[torch.Tensor] = None
    plateType_logits: Optional[torch.Tensor] = None
    bboxes_hat: Optional[torch.Tensor] = None
    decoder_verteces: Optional[torch.Tensor] = None  # exDETR: [B, 4, 2] in [0,1]
    STN_verteces: Optional[torch.Tensor] = None      # STN grid corners: [B, 4, 2] in [-1,1]
    # external_task_denoise_LPs_logits
    denoise_LPs_logits: Optional[torch.Tensor] = None
    # external_task_prior_LPs_logits
    prior_LPs_logits: Optional[torch.Tensor] = None
    theta: Optional[torch.Tensor] = None
    reconstructed_img: Optional[torch.Tensor] = None

    def mv_to_device(self, device):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.to(device))

    def __repr__(self):
        res = [f"{self.__class__.__name__}:"]
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                content = f"Tensor(shape={list(value.shape)})"
            else:
                content = repr(value)
            res.append(f"    {field.name}: {content}")
        return "\n".join(res)


def genBboxes_from_verts(pred_verts: torch.Tensor):
    """Convert 4 corner vertices [B, 4, 2] in [-1,1] to xyxy bbox [B, 4] in [0,1]."""
    verts_01 = (pred_verts + 1.0) / 2.0  # [-1,1] -> [0,1]
    xy_min = verts_01.min(dim=1).values   # [B, 2]
    xy_max = verts_01.max(dim=1).values   # [B, 2]
    return torch.cat([xy_min, xy_max], dim=-1)  # [B, 4] xyxy


class mtLPRtr_CvNxt_FPN_STN_co2D(nn.Module):
    '''
    mtLPRtr: backbone + STN + Transformer-based LPR & MAE
    - Backbone: convnext_tiny + FPN (stride 16)
    - STN: coord+2d sine pos  for transformer
    - LPR: TR decoder
    - MAE: Simple Transformer Encoder + PixelShuffle decoder for reconstruction
    '''
    def __init__(self, mae_mask_ratio=0.15, stn_strong_sup_only=False):
        super().__init__()
        self.mae_mask_ratio = mae_mask_ratio
        self.stn_strong_sup_only = stn_strong_sup_only
        self.d_model = 76  # 75 + 1
        self.LP_len = 8
        self.fm_h, self.fm_w = 8, 32
        self.n_heads = 19  # 76 / 4

        # 1. Backbone
        self.bone = Backbone.Timm_FPN(backbone_name='convnext_tiny', d_out=self.d_model)

        # 2. STN
        self.neck = Neck.STN_s16_coord_2Dsine_TR_2sup_g8_32(char_classNum=76)

        # 3. Position Encoding
        self.enPosEncoder = PosEncode.sinePosEncoding_2D(self.d_model, self.fm_h, self.fm_w)
        self.dePosEncoder = PosEncode.learnPosEncoding(self.d_model, max_len=self.LP_len)

        # 4. LPR Transformer (DETR structure)
        self.LPR_tr = detr(
            d_model=self.d_model,
            nhead=self.n_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=self.d_model * 2
        )

        self.prior_Q = nn.Parameter(torch.empty(self.LP_len, self.d_model))
        nn.init.kaiming_uniform_(self.prior_Q)

        # 5. MAE Head
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 2
        )
        self.mae_encoder = TransformerEncoder(encoder_layer, num_layers=1)
        self.mae_decoder_cnn = Upsample.simple_up16_res_pixshuff(self.d_model)

    def _generate_mae_mask(self, batch_size, seq_len, device):
        """生成布尔型 Mask，True 表示该位置被遮挡"""
        num_mask = int(seq_len * self.mae_mask_ratio)
        noise = torch.rand(batch_size, seq_len, device=device)
        _, mask_indices = torch.topk(noise, k=num_mask, dim=1)
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        mask.scatter_(1, mask_indices, True)
        return mask

    def forward(self, batch_data, need_stn=True, need_mae=True, need_lpr=True, **kwargs):
        imgs = batch_data.images  # [B, 3, H, W]
        B, C_img, H_img, W_img = imgs.shape

        # Stage 1: feature extraction + spatial transform
        fm = self.bone(imgs)
        fm = fm["stride16"]
        stn_results = {}
        if need_stn:
            stn_dict = self.neck.forward(fm, need_affine=True, detach_grid=self.stn_strong_sup_only)
            fm = stn_dict['fm']
            verts = stn_dict['grid'][:, [0, 0, -1, -1], [0, -1, -1, 0], :]
            stn_results['grid'] = verts
            stn_results['theta'] = stn_dict['theta']
            stn_results['bboxes'] = genBboxes_from_verts(verts)

        # Stage 2: prepare sequence
        B, C, H, W = fm.shape
        fm_seq = fm.view(B, C, -1).permute(2, 0, 1)
        pos_enc = self.enPosEncoder.forward_2d(h=H, w=W)
        pos_enc = pos_enc.flatten(2).permute(2, 0, 1)

        src_mask = None
        if need_mae and self.training:
            src_mask = self._generate_mae_mask(B, H * W, fm.device)

        # Stage 3: Transformer encode
        memory = self.LPR_tr.encoder(src=fm_seq, pos=pos_enc, src_key_padding_mask=src_mask)

        # Stage 4: MAE branch
        reconstructed_img = None
        if need_mae:
            mae_feat = self.mae_encoder(src=memory, pos=pos_enc)
            mae_fm = mae_feat.permute(1, 2, 0).view(B, C, H, W)
            reconstructed_img = self.mae_decoder_cnn(mae_fm)

        # Stage 5: LPR branch
        lpr_logits = None
        if need_lpr:
            query_embed = self.prior_Q.unsqueeze(1).expand(-1, B, -1)
            query_pos = self.dePosEncoder(query_embed)
            lpr_out = self.LPR_tr.decoder(
                tgt=query_embed,
                memory=memory,
                pos=pos_enc,
                query_pos=query_pos
            )
            lpr_logits = lpr_out.permute(1, 2, 0)

        return InferBox(
            reconstructed_img=reconstructed_img,
            LPs_logits=lpr_logits,
            STN_verteces=stn_results.get('grid'),
            theta=stn_results.get('theta'),
            bboxes_hat=stn_results.get('bboxes')
        )


class mtLPRtr_CvNxtnano_FPN_STN_co2D(nn.Module):
    '''
    mtLPRtr: backbone + STN + Transformer-based LPR & MAE
    - Backbone: convnext_nano + FPN (stride 16)
    - STN: coord+2d sine pos  for transformer
    - LPR: TR decoder
    - MAE: Simple Transformer Encoder + PixelShuffle decoder for reconstruction
    '''
    def __init__(self, mae_mask_ratio=0.15, stn_strong_sup_only=False):
        super().__init__()
        self.mae_mask_ratio = mae_mask_ratio
        self.stn_strong_sup_only = stn_strong_sup_only
        self.d_model = 76  # 75 + 1
        self.LP_len = 8
        self.fm_h, self.fm_w = 8, 32
        self.n_heads = 19  # 76 / 4

        # 1. Backbone
        self.bone = Backbone.Timm_FPN(backbone_name='convnext_nano', d_out=self.d_model)

        # 2. STN
        self.neck = Neck.STN_s16_coord_2Dsine_TR_2sup_g8_32(char_classNum=76)

        # 3. Position Encoding
        self.enPosEncoder = PosEncode.sinePosEncoding_2D(self.d_model, self.fm_h, self.fm_w)
        self.dePosEncoder = PosEncode.learnPosEncoding(self.d_model, max_len=self.LP_len)

        # 4. LPR Transformer (DETR structure)
        self.LPR_tr = detr(
            d_model=self.d_model,
            nhead=self.n_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=self.d_model * 2
        )

        self.prior_Q = nn.Parameter(torch.empty(self.LP_len, self.d_model))
        nn.init.kaiming_uniform_(self.prior_Q)

        # 5. MAE Head
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 2
        )
        self.mae_encoder = TransformerEncoder(encoder_layer, num_layers=1)
        self.mae_decoder_cnn = Upsample.simple_up16_res_pixshuff(self.d_model)

    def _generate_mae_mask(self, batch_size, seq_len, device):
        """生成布尔型 Mask，True 表示该位置被遮挡"""
        num_mask = int(seq_len * self.mae_mask_ratio)
        noise = torch.rand(batch_size, seq_len, device=device)
        _, mask_indices = torch.topk(noise, k=num_mask, dim=1)
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        mask.scatter_(1, mask_indices, True)
        return mask

    def forward(self, batch_data, need_stn=True, need_mae=True, need_lpr=True, **kwargs):
        imgs = batch_data.images  # [B, 3, H, W]
        B, C_img, H_img, W_img = imgs.shape

        # Stage 1: feature extraction + spatial transform
        fm = self.bone(imgs)
        fm = fm["stride16"]
        stn_results = {}
        if need_stn:
            stn_dict = self.neck.forward(fm, need_affine=True, detach_grid=self.stn_strong_sup_only)
            fm = stn_dict['fm']
            verts = stn_dict['grid'][:, [0, 0, -1, -1], [0, -1, -1, 0], :]
            stn_results['grid'] = verts
            stn_results['theta'] = stn_dict['theta']
            stn_results['bboxes'] = genBboxes_from_verts(verts)

        # Stage 2: prepare sequence
        B, C, H, W = fm.shape
        fm_seq = fm.view(B, C, -1).permute(2, 0, 1)
        pos_enc = self.enPosEncoder.forward_2d(h=H, w=W)
        pos_enc = pos_enc.flatten(2).permute(2, 0, 1)

        src_mask = None
        if need_mae and self.training:
            src_mask = self._generate_mae_mask(B, H * W, fm.device)

        # Stage 3: Transformer encode
        memory = self.LPR_tr.encoder(src=fm_seq, pos=pos_enc, src_key_padding_mask=src_mask)

        # Stage 4: MAE branch
        reconstructed_img = None
        if need_mae:
            mae_feat = self.mae_encoder(src=memory, pos=pos_enc)
            mae_fm = mae_feat.permute(1, 2, 0).view(B, C, H, W)
            reconstructed_img = self.mae_decoder_cnn(mae_fm)

        # Stage 5: LPR branch
        lpr_logits = None
        if need_lpr:
            query_embed = self.prior_Q.unsqueeze(1).expand(-1, B, -1)
            query_pos = self.dePosEncoder(query_embed)
            lpr_out = self.LPR_tr.decoder(
                tgt=query_embed,
                memory=memory,
                pos=pos_enc,
                query_pos=query_pos
            )
            lpr_logits = lpr_out.permute(1, 2, 0)

        return InferBox(
            reconstructed_img=reconstructed_img,
            LPs_logits=lpr_logits,
            STN_verteces=stn_results.get('grid'),
            theta=stn_results.get('theta'),
            bboxes_hat=stn_results.get('bboxes')
        )


# ConvNeXt Pico backbone ablation (inherits mtLPRtr_CvNxt_FPN_STN_co2D, swaps backbone only)
class ab_backbone_convnext_pico(mtLPRtr_CvNxt_FPN_STN_co2D):
    def __init__(self, mae_mask_ratio=0.15, stn_strong_sup_only=False):
        super().__init__(mae_mask_ratio, stn_strong_sup_only)
        self.bone = Backbone.Timm_FPN(backbone_name='convnext_pico', d_out=self.d_model)


# ---------------------------------------------------------------------------
# exDETR: Extended DETR for joint detection + recognition
# Key ideas:
#   1. Grouped tokens: Lgroup tokens per plate prediction.
#      Token[0] -> bbox (8 coords, two 4-pt boxes) + class
#      Token[1:Lgroup] -> character logits (Lgroup-1 chars)
#   2. Hungarian matching uses all three costs (class, bbox, string).
#   3. CDN: one positive + one negative denoise group appended to decoder input.
#      Positive: GT bbox+chars. Negative: noisy bbox, GT chars.
# ---------------------------------------------------------------------------

class ExDETR_Decoder(nn.Module):
    '''
    exDETR grouped-token decoder.
    num_predict: number of plate predictions (detection slots).
    Lgroup: tokens per prediction (1 bbox token + Lgroup-1 char tokens).
    Architecture:
      - tgt_infer: [num_predict*Lgroup, d_model] learnable content queries
      - de_pos_infer: [Lgroup, d_model] learnable positional queries (tiled)
      - TransformerDecoder with 2 layers
    Output dict:
      pred_logits: [B, num_predict, n_class]
      pred_boxes:  [B, num_predict, 8]   (4 vertices × (x,y), sigmoid [0,1]; bbox = bounding rect)
      pred_string_logits: [B, n_character, num_predict, Lgroup-1]
    '''
    def __init__(
        self,
        num_predict: int = 10,
        Lgroup: int = 9,
        nlayers: int = 2,
        nhead: int = 8,
        d_ffn: int = 512,
        d_model: int = 256,
        n_class: int = 2,
        n_bbox_vertex: int = 4,
        n_character: int = 75,
    ):
        super().__init__()
        self.num_predict = num_predict
        self.Lgroup = Lgroup
        self.n_character = n_character
        self.n_class = n_class

        self.tgt_infer = nn.Embedding(num_predict * Lgroup, d_model)
        self.de_pos_infer = nn.Embedding(Lgroup, d_model)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_ffn)
        self.TRdecoder = TransformerDecoder(decoder_layer, nlayers)

        self.class_proj = nn.Linear(d_model, n_class)
        self.bbox_proj = MLP(d_model, d_model, n_bbox_vertex * 2, 3)
        self.character_proj = nn.Linear(d_model, n_character)

    def prepare_query(self, B: int):
        tgt = self.tgt_infer.weight.unsqueeze(1).repeat(1, B, 1)
        pos = self.de_pos_infer.weight.unsqueeze(1).repeat(self.num_predict, B, 1)
        return tgt, pos

    def read_memory(self, en_outputs: dict):
        return en_outputs['memory'], en_outputs['mask'], en_outputs['en_pos']

    def seg_prediction(self, B: int, output: torch.Tensor):
        '''Split grouped tokens into bbox tokens and char tokens.'''
        seq_len = output.size(0)
        is_char = torch.ones(seq_len, dtype=torch.bool, device=output.device)
        is_char[::self.Lgroup] = False

        tokens_bbox = output[~is_char]
        tokens_char = output[is_char]

        outputs_class = self.class_proj(tokens_bbox).permute(1, 0, 2)
        outputs_coord = self.bbox_proj(tokens_bbox).sigmoid().permute(1, 0, 2)

        outputs_string = self.character_proj(tokens_char)
        outputs_string = outputs_string.view(
            self.num_predict, self.Lgroup - 1, B, self.n_character
        ).permute(2, 3, 0, 1)

        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord,
            'pred_string_logits': outputs_string,
        }

    def forward(self, en_outputs: dict):
        memory, mask, en_pos = self.read_memory(en_outputs)
        _, B, _ = memory.shape
        tgt, pos = self.prepare_query(B)
        output = self.TRdecoder(
            tgt, memory,
            memory_key_padding_mask=mask,
            pos=en_pos,
            query_pos=pos,
        )
        return self.seg_prediction(B, output)


class ExDETR_CDN(nn.Module):
    '''
    CDN wrapper for ExDETR_Decoder.
    Appends two denoise groups (pos+neg) to the decoder input with causal mask.
    content query = [class_embed, lps_embed]
    pos query     = [de_pos_infer[0] + bbox_embed, de_pos_infer[1:]]

    forward inputs dict:
        en_outputs: dict{memory, mask, en_pos}
        pos_bbox:   [B, 1, 8]   GT vertices (4 × (x,y) in [0,1], from STN grid corners)
        neg_bbox:   [B, 1, 8]   noisy GT vertices
        LPs_delay:  [B, Lgroup-1]  GT char ids (shifted by 1 for teacher forcing)
    forward returns same keys as ExDETR_Decoder.seg_prediction plus CDN outputs.
      pos_bbox [B,8] / neg_bbox [B,8]: predicted vertices (x,y) [0,1]; bbox = bounding rect.
    '''
    def __init__(
        self,
        decoder: ExDETR_Decoder,
        d_model: int = 256,
        n_bbox_vertex: int = 4,
        n_character: int = 75,
    ):
        super().__init__()
        self.decoder = decoder
        self.d_model = d_model
        self.classEmbed = nn.Embedding(decoder.n_class, d_model)
        self.lps_embed = nn.Embedding(n_character, d_model)
        self.bbox_embed = MLP(n_bbox_vertex * 2, d_model // 2, d_model, 3)
        self.register_buffer('tgt_mask', self._build_qk_mask())

    def _build_qk_mask(self) -> torch.Tensor:
        '''Build self-attn mask: infer tokens can see everything; denoise groups
        are causal and cannot see each other across pos/neg.'''
        L_infer = self.decoder.num_predict * self.decoder.Lgroup
        L_dn = 2 * self.decoder.Lgroup

        a = torch.zeros(L_infer, L_infer)
        b = torch.full((L_infer, L_dn), float('-inf'))
        c = torch.zeros(L_dn, L_infer)

        def causal(sz):
            return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

        d = torch.full((L_dn, L_dn), float('-inf'))
        d[:self.decoder.Lgroup, :self.decoder.Lgroup] = causal(self.decoder.Lgroup)
        d[self.decoder.Lgroup:, self.decoder.Lgroup:] = causal(self.decoder.Lgroup)

        return torch.cat([
            torch.cat([a, b], dim=1),
            torch.cat([c, d], dim=1),
        ], dim=0)

    def _prepare_dn_query(self, B: int, inputs: dict):
        '''Build denoise tgt and pos queries.'''
        tgt_infer, pos_infer = self.decoder.prepare_query(B)

        pos_bbox = inputs['pos_bbox'].squeeze(1)
        neg_bbox = inputs['neg_bbox'].squeeze(1)
        pos_bbox_emb = self.bbox_embed(pos_bbox)
        neg_bbox_emb = self.bbox_embed(neg_bbox)

        noise_pos = self.decoder.de_pos_infer.weight.unsqueeze(1).repeat(2, B, 1)
        noise_pos[0] = noise_pos[0] + pos_bbox_emb
        noise_pos[self.decoder.Lgroup] = noise_pos[self.decoder.Lgroup] + neg_bbox_emb

        lps_ids = inputs['LPs_delay'].permute(1, 0)
        lps_emb = self.lps_embed(lps_ids)
        class_ids = torch.tensor([1, 0], device=tgt_infer.device).unsqueeze(1)
        class_emb = self.classEmbed(class_ids).repeat(1, B, 1)

        tgt_dn = torch.cat([
            class_emb[[0]], lps_emb,   # positive group
            class_emb[[1]], lps_emb,   # negative group
        ], dim=0)

        tgt_full = torch.cat([tgt_infer, tgt_dn], dim=0)
        pos_full = torch.cat([pos_infer, noise_pos], dim=0)
        return tgt_full, pos_full

    def forward(self, inputs: dict, denoise: bool = True):
        en_outputs = inputs['en_outputs']
        if not denoise:
            return self.decoder.forward(en_outputs)

        memory, mask, en_pos = self.decoder.read_memory(en_outputs)
        _, B, _ = memory.shape

        tgt_full, pos_full = self._prepare_dn_query(B, inputs)
        output = self.decoder.TRdecoder(
            tgt_full, memory,
            tgt_mask=self.tgt_mask,
            memory_key_padding_mask=mask,
            pos=en_pos,
            query_pos=pos_full,
        )

        L_infer = self.decoder.num_predict * self.decoder.Lgroup
        out_infer = self.decoder.seg_prediction(B, output[:L_infer])

        tokens_dn = output[L_infer:]
        bbox_tokens = tokens_dn[[0, self.decoder.Lgroup]]
        char_tokens_pos = tokens_dn[1:self.decoder.Lgroup]

        pos_neg_class = self.decoder.class_proj(bbox_tokens)
        pos_neg_bbox = self.decoder.bbox_proj(bbox_tokens).sigmoid()
        lp_dn_logits = self.decoder.character_proj(char_tokens_pos).permute(1, 2, 0)

        pos_class_logit, neg_class_logit = pos_neg_class.unbind(0)
        pos_bbox_pred, neg_bbox_pred = pos_neg_bbox.unbind(0)

        out_dn = {
            'pos_class_logit': pos_class_logit,
            'neg_class_logit': neg_class_logit,
            'pos_bbox': pos_bbox_pred,
            'neg_bbox': neg_bbox_pred,
            'LP_dn_logits': lp_dn_logits,
        }
        return {**out_infer, **out_dn}


class exDETR_CDN_CvNxt_FPN_STN_co2D(nn.Module):
    '''
    exDETR method with ConvNeXt backbone (Method 2 in the paper).

    Architecture:
      Image [B,3,H,W]
        -> ConvNeXt-nano + FPN (stride32)  -> fm [B, d_model, H/32, W/32]
        -> STN neck (coord+2D-sine TR localization)  -> corrected fm + vertices/theta
        -> Flatten + 2D sine positional encoding  -> sequence [H/32*W/32, B, d_model]
        -> 2-layer DETR encoder  -> memory
        -> ExDETR decoder: Lgroup grouped tokens per prediction
             token[0] -> bbox (8 coords) + plate class
             token[1:Lgroup] -> char logits (Lgroup-1 characters)
        -> CDN denoise branch during training

    Output: InferBox with:
        LPs_logits: [B, n_char, Lgroup-1]  best-confidence slot string logits (for evaluator)
        bboxes_hat: [B, 4] xyxy bbox from STN
        verteces:   [B, 4, 2] STN corner points
        exdetr_out: dict with pred_logits, pred_boxes, pred_string_logits, + CDN keys (for loss)
    '''
    def __init__(
        self,
        num_predict: int = 10,
        Lgroup: int = 9,
        n_class: int = 2,
        n_character: int = 75,
        d_model: int = 256,
        stn_detach: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = 8

        # 1. Backbone: ConvNeXt-nano + FPN, stride32 output
        self.bone = Backbone.Timm_FPN_stride32(
            backbone_name='convnext_nano',
            d_out=d_model,
            pretrained=True,
        )

        # 2. STN neck for stride-32 fm
        self.neck = Neck.STN_s32_coord_2Dsine_TR_2sup_g4_16(
            char_classNum=d_model,
            stn_detach=stn_detach,
        )

        # 3. DETR encoder (2 layers)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=self.n_heads, dim_feedforward=d_model * 4
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2)

        # 4. exDETR decoder + CDN
        _decoder = ExDETR_Decoder(
            num_predict=num_predict,
            Lgroup=Lgroup,
            nlayers=2,
            nhead=self.n_heads,
            d_ffn=d_model * 4,
            d_model=d_model,
            n_class=n_class,
            n_bbox_vertex=4,
            n_character=n_character,
        )
        self.cdn_decoder = ExDETR_CDN(
            decoder=_decoder,
            d_model=d_model,
            n_bbox_vertex=4,
            n_character=n_character,
        )

        # 5. Positional encoding for stride-32 fm [4, 16] from 128x512 input
        self.fm_h, self.fm_w = 4, 16
        self.enPosEncoder = PosEncode.sinePosEncoding_2D(d_model, self.fm_h, self.fm_w)

    def forward(self, batch_data, need_stn: bool = True, denoise: bool = True, **kwargs):
        '''
        batch_data: BatchBox with .images [B,3,H,W]
        During training pass denoise=True and batch_data must have:
            .bboxes    [B,4]  (xyxy GT bbox)
            .LPs_delay [B,8]  (shifted char GT ids for CDN)
        Returns InferBox; exDETR outputs stored in exdetr_out attribute.
        '''
        imgs = batch_data.images
        B = imgs.size(0)

        # 1. Backbone
        fm_dict = self.bone(imgs)
        fm = fm_dict['stride32']

        # 2. STN
        stn_results = {}
        if need_stn:
            stn_dict = self.neck.forward(fm, need_affine=True)
            fm = stn_dict['fm']
            verts = stn_dict['grid'][:, [0, 0, -1, -1], [0, -1, -1, 0], :]
            stn_results['grid'] = verts
            stn_results['theta'] = stn_dict['theta']
            stn_results['bboxes'] = genBboxes_from_verts(verts)

        # 3. Prepare encoder input
        B2, C2, H2, W2 = fm.shape
        pos_enc = self.enPosEncoder.forward_2d(h=H2, w=W2)
        pos_enc = pos_enc.flatten(2).permute(2, 0, 1)
        fm_seq = fm.view(B2, C2, -1).permute(2, 0, 1)

        # 4. Encode
        memory = self.encoder(fm_seq, pos=pos_enc)
        en_outputs = {'memory': memory, 'mask': None, 'en_pos': pos_enc}

        # 5. Decode with optional CDN
        if denoise and self.training and getattr(batch_data, 'bboxes', None) is not None:
            if stn_results.get('grid') is not None:
                verts_01 = (stn_results['grid'].reshape(B, 4, 2) + 1.0) / 2.0
                pos_verts = verts_01.reshape(B, 8).unsqueeze(1)
            else:
                tgt = box_xyxy_to_cxcywh(batch_data.bboxes)
                cx, cy, w, h = tgt.unbind(-1)
                x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
                pos_verts = torch.stack([x1,y1, x2,y1, x2,y2, x1,y2], dim=-1).unsqueeze(1)
            neg_verts = (pos_verts + 0.05 * torch.randn_like(pos_verts)).clamp(0, 1)
            lps_delay = batch_data.LPs_delay[:, :self.cdn_decoder.decoder.Lgroup - 1]
            cdn_inputs = {
                'en_outputs': en_outputs,
                'pos_bbox': pos_verts,
                'neg_bbox': neg_verts,
                'LPs_delay': lps_delay,
            }
            exdetr_out = self.cdn_decoder.forward(cdn_inputs, denoise=True)
        else:
            exdetr_out = self.cdn_decoder.forward({'en_outputs': en_outputs}, denoise=False)

        # Pick best-confidence slot
        pred_logits = exdetr_out['pred_logits']
        best_idx = pred_logits[:, :, 1].argmax(dim=1)
        pred_str = exdetr_out['pred_string_logits']
        lps_logits = pred_str[torch.arange(B, device=pred_str.device), :, best_idx, :]

        best_verts = exdetr_out['pred_boxes'][torch.arange(B, device=pred_str.device), best_idx]
        best_verts_2d = best_verts.reshape(B, 4, 2)
        bboxes_from_decoder = torch.cat([best_verts_2d.min(dim=1).values,
                                         best_verts_2d.max(dim=1).values], dim=-1)

        box = InferBox(
            LPs_logits=lps_logits,
            bboxes_hat=bboxes_from_decoder,
            decoder_verteces=best_verts_2d,
            STN_verteces=stn_results.get('grid'),
            theta=stn_results.get('theta'),
        )
        box.exdetr_out = exdetr_out
        return box
