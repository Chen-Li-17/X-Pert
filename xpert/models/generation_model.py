import os
import math
from typing import Mapping, Optional, Tuple, Any, Union

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from torch.utils.data import dataset
from tqdm import trange

import numpy as np

from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

from .model import (
    ExprDecoder,
    ZILNDecoder,
    BCE_MSE_Decoder,
    DoseFiLM,
    MVCDecoder,
    ContinuousValueEncoder,
    FastTransformerEncoderWrapper,
    FlashTransformerEncoderLayer,
    MLPDecoder,
    FiLM,
)
from ..utils import map_raw_id_to_vocab_id
from .. import logger

from .flamingo_func import GatedCrossAttentionBlock, PerceiverResampler, MLP, MLP_gene

class TransformerGenerator(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int,
        n_cls: int,
        vocab: Any,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        pert_pad_id: int = 2,
        do_mvc: bool = False,
        domain_spec_batchnorm: Union[bool, str] = False,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        pert_embed_dict = None,
        gpt_emb_dim = 1000,
        model_mode = None,
        attn_gate_mode = None,
        pert_mode = 'gene',
        drug_embed_mode = 'gpt',
        pert_flag_mode = True,
        use_scgpt_layer = True,
        use_scgpt_input = True,
        mask_mode = True,
        add_token = False,
        init_mode = False,

        dosage_mode_type = 0, # 0: direct multiply; 1: learnable vector

        decoder_type = 'mse',

        cross_mode = True,

        output_mode = 'scgpt', # scgpt; mlp
        feature_dim = None,

        cross_norm = False,

        film_mode = False,

    ):
        super().__init__()
        # print('this is the new version 4')
        self.model_type = "Transformer"
        self.d_model = d_model
        self.pad_token_id = vocab[pad_token]
        self.pad_value = pad_value
        self.pert_pad_id = pert_pad_id
        self.ecs_threshold = ecs_threshold
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        if use_fast_transformer:
            try:
                from flash_attn.flash_attention import FlashMHA
            except ImportError:
                import warnings

                warnings.warn(
                    "flash-attn is not installed, using pytorch transformer instead. "
                    "Set use_fast_transformer=False to avoid this warning. "
                    "Installing flash-attn is highly recommended."
                )
                use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer

        self.add_token = add_token
        if self.add_token:
            self.encoder_plus = GeneEncoder(len(vocab), d_model, padding_idx=vocab[pad_token])
        else:
            self.encoder_plus = None

        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)

        self.drug_pert_encoder = ContinuousValueEncoder(d_model, dropout)

        print("Using simple batchnorm instead of domain specific batchnorm")
        self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        if use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # self.decoder = nn.Linear(d_model, 1)
        self.decoder = ExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
        )
        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
            )

        self.sim = Similarity(temp=0.5)
        self.creterion_cce = nn.CrossEntropyLoss()

        # - construct the pert_embed_dict
        self.pert_embed_dict = pert_embed_dict
        # - get the model_mode
        self.model_mode = model_mode
        # - get the linear layer for gpt embed
        self.linear_1 = nn.Linear(gpt_emb_dim, d_hid, bias = True)

        self.pert_mode = pert_mode
        self.drug_embed_mode = drug_embed_mode
        self.pert_flag_mode = pert_flag_mode
        self.use_scgpt_layer = use_scgpt_layer
        self.use_scgpt_input = use_scgpt_input
        self.init_mode = init_mode
        self.dosage_mode_type = dosage_mode_type
        self.decoder_type = decoder_type
        self.cross_mode = cross_mode
        self.output_mode = output_mode
        self.cross_norm = cross_norm
        self.film_mode = film_mode

    

        if self.decoder_type == 'ziln':
            self.decoder_ziln = ZILNDecoder(
                d_model,
                shared_hidden=True,
            )
        if self.decoder_type == 'bce_mse':
            self.decoder_bce_mse = BCE_MSE_Decoder(
                d_model,
            )

        if not cross_mode:
            self.gene_mlp = MLP_gene([gpt_emb_dim, d_hid*2, d_hid], batch_norm=False, last_layer_act="linear")

        if self.output_mode == 'mlp':
            self.pert_decoder = MLPDecoder(d_hid, d_hid, d_hid, feature_dim, p_dropout=0.1)

        # if drug_embed_mode == 'gpt':
        #     self.drug_mlp = MLP([gpt_emb_dim, d_hid*2, d_hid], batch_norm=False, last_layer_act="linear")
        # elif drug_embed_mode == 'grover':
        #     self.drug_mlp = MLP([gpt_emb_dim, d_hid*2, d_hid], batch_norm=False, last_layer_act="linear")
        # elif drug_embed_mode == 'rdkit':
        #     self.drug_mlp = MLP([gpt_emb_dim, d_hid*2, d_hid], batch_norm=False, last_layer_act="linear")
        # elif drug_embed_mode == 'morgan':
        #     self.drug_mlp = MLP([gpt_emb_dim, d_hid*2, d_hid], batch_norm=False, last_layer_act="linear")
        # else:
        #     raise ValueError()
        
        self.drug_mlp = MLP([gpt_emb_dim, d_hid*2, d_hid], batch_norm=False, last_layer_act="linear")
        
        self.dose_vector = nn.Parameter(torch.randn(gpt_emb_dim, ))

        self.DoseFiLM = DoseFiLM(d_model = gpt_emb_dim,
                                 hidden_dim = d_hid)

        # - get the PerceiverResampler
        self.perceive = PerceiverResampler(
            dim = d_hid,
            depth = 2,
            dim_head = 64,
            heads = 8,
            num_latents = 12,    # the number of latents to shrink your media sequence to, perceiver style
            num_media_embeds = 4,  # say you have 4 images maximum in your dialogue
            mask_mode = mask_mode,
        )
        # - get the cross_attn
        self.attn_gate_mode = attn_gate_mode
        if self.cross_mode:
            self.cross_attn_layers = nn.ModuleList(
                [GatedCrossAttentionBlock(
                dim = d_hid,
                dim_head = 64,
                heads = 8,
                attn_gate_mode = self.attn_gate_mode,
                use_scgpt_layer = self.use_scgpt_layer,
                use_scgpt_input = self.use_scgpt_input,
                cross_norm = cross_norm,
            ) for _ in range(nlayers)
            ]
            )
        else:
            self.cross_attn_layers = nn.ModuleList(
                [GatedCrossAttentionBlock(
                dim = d_hid,
                dim_head = 64,
                heads = 8,
                attn_gate_mode = self.attn_gate_mode,
                use_scgpt_layer = self.use_scgpt_layer,
                use_scgpt_input = self.use_scgpt_input,
            ) for _ in range(nlayers)
            ]
            )

        if self.film_mode:
            self.film_layers = nn.ModuleList(
                [
                    FiLM(d_hid, d_hid, groups=4, mlp_ratio=1)
                    for _ in range(nlayers)
                ]
            )


        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
        if self.add_token:
            self.encoder_plus.embedding.weight.data.uniform_(-initrange, initrange)
        import torch.nn.init as init
        if self.init_mode:
            # 遍历模型的所有层进行初始化
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    # 对于线性层，使用Xavier初始化
                    init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        init.zeros_(module.bias)  # 偏置项初始化为零
                elif isinstance(module, nn.Conv2d):
                    # 对于卷积层，使用He初始化
                    init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # 适应ReLU激活函数
                    if module.bias is not None:
                        init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm1d):
                    # 对于BatchNorm，使用常见的初始化方法
                    init.ones_(module.weight)  # 权重初始化为1
                    init.zeros_(module.bias)   # 偏置项初始化为0
                elif isinstance(module, nn.Embedding):
                    # 对于嵌入层，使用均匀分布初始化
                    init.uniform_(module.weight, -initrange, initrange)
                elif isinstance(module, nn.LayerNorm):
                    # 对于LayerNorm，通常初始化为1
                    init.ones_(module.weight)
                    init.zeros_(module.bias)
                elif isinstance(module, nn.TransformerEncoderLayer):
                    # 对于Transformer的每一层，初始化其各个参数
                    init.xavier_uniform_(module.self_attn.in_proj_weight)  # 线性变换的权重
                    init.xavier_uniform_(module.self_attn.out_proj.weight)  # self attention的输出权重
                    init.xavier_uniform_(module.linear1.weight)  # 第一个前馈层
                    init.xavier_uniform_(module.linear2.weight)  # 第二个前馈层
                    if module.linear1.bias is not None:
                        init.zeros_(module.linear1.bias)
                    if module.linear2.bias is not None:
                        init.zeros_(module.linear2.bias)
                    if module.self_attn.in_proj_bias is not None:
                        init.zeros_(module.self_attn.in_proj_bias)
                    if module.self_attn.out_proj.bias is not None:
                        init.zeros_(module.self_attn.out_proj.bias)
                # 可以继续添加其他类型的层的初始化

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags,
        src_key_padding_mask: Tensor,
        batch_pert_embed: Tensor,
        pert_mask: Tensor,
        batch_dosages_pad = None,
    ) -> Tensor:
        if not self.add_token:
            src = self.encoder(src)  # (batch, seq_len, embsize)
        else:
            src = self.encoder_plus(src)  # (batch, seq_len, embsize)
            
        self.cur_gene_token_embs = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        
        if self.pert_mode == 'gene':
            perts = self.pert_encoder(input_pert_flags)  # (batch, seq_len, embsize)
        elif self.pert_mode == 'drug' and self.pert_flag_mode:
            perts = self.drug_pert_encoder(input_pert_flags)  # (batch, seq_len, embsize)
        elif self.pert_mode == 'drug' and not self.pert_flag_mode:
            pass
        else:
            raise ValueError()

        if self.pert_flag_mode:
            total_embs = src + values + perts
        else:
            total_embs = src + values

        total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        if self.cross_mode:

            # - add flamingo cross attention
            if self.pert_mode == 'gene':
                batch_pert_embed = self.linear_1(batch_pert_embed)
            elif self.pert_mode == 'drug':
                if batch_dosages_pad != None: # whether use the dosage info
                    if self.dosage_mode_type == 0:
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        batch_pert_embed = self.drug_mlp(batch_pert_embed * batch_dosages_pad) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 1:
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec = repeat(self.dose_vector, 'd -> b p d', b=batch_dosages_pad.shape[0], p=batch_dosages_pad.shape[1])
                        batch_pert_embed = self.drug_mlp(batch_pert_embed * (dose_vec*batch_dosages_pad)) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 2:
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec = repeat(self.dose_vector, 'd -> b p d', b=batch_dosages_pad.shape[0], p=batch_dosages_pad.shape[1])
                        batch_pert_embed = self.drug_mlp(batch_pert_embed + (dose_vec*batch_dosages_pad)) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 3:
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec = repeat(self.dose_vector, 'd -> b p d', b=batch_dosages_pad.shape[0], p=batch_dosages_pad.shape[1])
                        batch_pert_embed = self.drug_mlp(batch_pert_embed) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 4: # 两层MLP
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec_gamma, dose_vec_beta = self.DoseFiLM(batch_dosages_pad)
                        batch_pert_embed = self.drug_mlp(batch_pert_embed * dose_vec_gamma + dose_vec_beta) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 5: # 两层MLP
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec_gamma, dose_vec_beta = self.DoseFiLM(batch_dosages_pad)
                        batch_pert_embed = self.drug_mlp(batch_pert_embed + dose_vec_beta) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 6: # 两层MLP
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec_gamma, dose_vec_beta = self.DoseFiLM(batch_dosages_pad)
                        batch_pert_embed = self.drug_mlp(batch_pert_embed * dose_vec_gamma) # batch_pert_embed: batch * pert num * embed dim
                    else:
                        raise ValueError()
                else:
                    batch_pert_embed = self.drug_mlp(batch_pert_embed)
            else:
                raise ValueError()
            
            # batch_pert_embed = rearrange(batch_pert_embed, 'b d -> b 1 d')
            perceived = self.perceive(batch_pert_embed, pert_mask)
        
        if not self.cross_mode and self.pert_mode == 'gene':
            batch_pert_embed = self.gene_mlp(batch_pert_embed)
            # batch_pert_embed = batch_pert_embed.unsqueeze(1)  # batch_pert_embed is already batch * pert num * h dim, no need to unsqueeze

            total_embs += batch_pert_embed
            


        output = total_embs
        if self.film_mode:
            for cross_attn, self_attn, film_layer in zip(self.cross_attn_layers, self.transformer_encoder.layers, self.film_layers):
                if self.use_scgpt_layer:
                    output = self_attn(output,
                                    src_key_padding_mask=src_key_padding_mask)
                    
                if self.cross_mode:
                    output = cross_attn(output,
                                        perceived,
                                        media_locations = None,
                                        )
                    
                output = film_layer(
                    output,
                    batch_pert_embed.squeeze(1),
                )
        else:
            for cross_attn, self_attn in zip(self.cross_attn_layers, self.transformer_encoder.layers):
                if self.cross_mode:
                    output = cross_attn(output,
                                        perceived,
                                        media_locations = None,
                                        )
                if self.use_scgpt_layer:
                    output = self_attn(output,
                                    src_key_padding_mask=src_key_padding_mask)
        
        # output = self.transformer_encoder(
        #     total_embs, src_key_padding_mask=src_key_padding_mask
        # )

        return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags: Tensor,
        src_key_padding_mask: Tensor,
        batch_pert_embed,
        pert_mask,
        batch_dosages_pad,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """

        if self.output_mode == 'scgpt':
            if self.explicit_zero_prob and not do_sample and not self.training:
                do_sample = True
                logger.warning("Auto set do_sample to True when model is in eval mode.")


            transformer_output = self._encode(
                src, values, input_pert_flags, src_key_padding_mask, batch_pert_embed, pert_mask, batch_dosages_pad
            )
            output = {}
            if self.decoder_type == 'mse' or self.decoder_type == 'balance_mse':
                mlm_output = self.decoder(transformer_output)
            elif self.decoder_type == 'ziln':
                mlm_output = self.decoder_ziln(transformer_output)
                return mlm_output
            elif self.decoder_type == 'bce_mse':
                mlm_output = self.decoder_bce_mse(transformer_output)
                return mlm_output
            else:
                raise ValueError()

            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
                output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
            else:
                output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mlm_zero_probs"] = mlm_output["zero_probs"]

            cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
            if CLS:
                output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
            if MVC:
                mvc_output = self.mvc_decoder(
                    cell_emb,
                    self.cur_gene_token_embs,
                )  # (batch, seq_len)
                if self.explicit_zero_prob and do_sample:
                    bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                    output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
                else:
                    output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
                if self.explicit_zero_prob:
                    output["mvc_zero_probs"] = mvc_output["zero_probs"]
            if ECS:
                # Here using customized cosine similarity instead of F.cosine_similarity
                # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
                # normalize the embedding
                cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
                cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

                # mask out diagnal elements
                mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
                cos_sim = cos_sim.masked_fill(mask, 0.0)
                # only optimize positive similarities
                cos_sim = F.relu(cos_sim)

                output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        elif self.output_mode == 'mlp':
            # - add flamingo cross attention
            if self.pert_mode == 'gene':
                batch_pert_embed = self.linear_1(batch_pert_embed)
            elif self.pert_mode == 'drug':
                if batch_dosages_pad != None: # whether use the dosage info
                    if self.dosage_mode_type == 0:
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        batch_pert_embed = self.drug_mlp(batch_pert_embed * batch_dosages_pad) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 1:
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec = repeat(self.dose_vector, 'd -> b p d', b=batch_dosages_pad.shape[0], p=batch_dosages_pad.shape[1])
                        batch_pert_embed = self.drug_mlp(batch_pert_embed * (dose_vec*batch_dosages_pad)) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 2:
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec = repeat(self.dose_vector, 'd -> b p d', b=batch_dosages_pad.shape[0], p=batch_dosages_pad.shape[1])
                        batch_pert_embed = self.drug_mlp(batch_pert_embed + (dose_vec*batch_dosages_pad)) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 3:
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec = repeat(self.dose_vector, 'd -> b p d', b=batch_dosages_pad.shape[0], p=batch_dosages_pad.shape[1])
                        batch_pert_embed = self.drug_mlp(batch_pert_embed) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 4: # 两层MLP
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec_gamma, dose_vec_beta = self.DoseFiLM(batch_dosages_pad)
                        batch_pert_embed = self.drug_mlp(batch_pert_embed * dose_vec_gamma + dose_vec_beta) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 5: # 两层MLP
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec_gamma, dose_vec_beta = self.DoseFiLM(batch_dosages_pad)
                        batch_pert_embed = self.drug_mlp(batch_pert_embed + dose_vec_beta) # batch_pert_embed: batch * pert num * embed dim
                    elif self.dosage_mode_type == 6: # 两层MLP
                        batch_dosages_pad = batch_dosages_pad.unsqueeze(2) # (batch * pert num)->(batch * pert num * 1)
                        dose_vec_gamma, dose_vec_beta = self.DoseFiLM(batch_dosages_pad)
                        batch_pert_embed = self.drug_mlp(batch_pert_embed * dose_vec_gamma) # batch_pert_embed: batch * pert num * embed dim
                    else:
                        raise ValueError()
                else:
                    batch_pert_embed = self.drug_mlp(batch_pert_embed)
            else:
                raise ValueError()
            
            batch_pert_embed = batch_pert_embed.squeeze(1) 
            batch_pert_output = self.pert_decoder(batch_pert_embed)
            output = {}
            output["mlm_output"] = batch_pert_output

        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        output_to_cpu: bool = True,
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [N, seq_len]
            values: Tensor, shape [N, seq_len]
            src_key_padding_mask: Tensor, shape [N, seq_len]

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        outputs = []
        N = src.size(0)
        device = next(self.parameters()).device
        for i in trange(0, N, batch_size):
            output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
            )
            if output_to_cpu:
                output = output.cpu()
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def pred_perturb(
        self,
        batch_data,
        include_zero_gene="batch-wise",
        gene_ids=None,
        amp=True,
    ) -> Tensor:
        """
        Args:
            batch_data: a dictionary of input data with keys.

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, -1)  # (batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, -1)

        if include_zero_gene in ["all", "batch-wise"]:
            assert gene_ids is not None
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
            else:  # batch-wise
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = self(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=True,
                )
            output_values = output_dict["mlm_output"].float()
            pred_gene_values = torch.zeros_like(ori_gene_values)
            pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values
    
    def pred_perturb_new(
        self,
        batch_data,
        include_zero_gene="batch-wise",
        gene_ids=None,
        amp=True,
    ) -> Tensor:
        """
        Args:
            batch_data: a dictionary of input data with keys.

        Returns:
            output Tensor of shape [N, seq_len]
        """
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)
        batch_size = len(batch_data.pert)
        x: torch.Tensor = batch_data.x
        # ori_gene_values = x[:, 0].view(batch_size, n_genes)
        ori_gene_values = x
        # pert_flags = x[:, 1].long().view(batch_size, n_genes)
        pert_flags = batch_data.pert_flags.long()

        if include_zero_gene in ["all", "batch-wise"]:
            assert gene_ids is not None
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
            else:  # batch-wise
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = self(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=True,
                )
            output_values = output_dict["mlm_output"].float()
            pred_gene_values = torch.zeros_like(ori_gene_values)
            pred_gene_values[:, input_gene_ids] = output_values
        return pred_gene_values


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
