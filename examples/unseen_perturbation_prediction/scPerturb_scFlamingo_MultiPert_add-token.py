######################################import
import scanpy as sc
import pandas as pd
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch_geometric.data import Data

import pickle
import sys
import requests

from types import MethodType

import json
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
machine_mode = 'xglab' # 'volcano': bytedance server; 'xglab': lab server

if machine_mode == 'volcano':
    # sys.path.insert(0, "/hpc-cache-pfs/home/lichen/code/scFlamingo/other_code/scGPT_flamingo/scGPT")
    sys.path.insert(0, "/hpc-cache-pfs/home/lichen/code/scFlamingo/other_code/scGPT_flamingo_MultiPert/scGPT")
elif machine_mode == 'xglab':
    # sys.path.insert(0, "/nfs/public/lichen/code/single_cell_perturbation/volc_scFlamingo/xpert_gene_perturbation/scGPT_flamingo_MultiPert/scGPT/")
    sys.path.insert(0, "/nfs/public/lichen/code/single_cell_perturbation/volc_scFlamingo/drug_perturbation/scGPT_flamingo_MultiPert_drug/scGPT/")
else:
    raise ValueError()

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

# matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

set_seed(42)

from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

from types import MethodType
import importlib
from scperturb import *
import anndata as ad

if machine_mode == 'volcano':
    sys.path.append("/hpc-cache-pfs/home/lichen/code/A200_code/single_cell_perturbation/scPerturb/Byte_Pert_Data/")
elif machine_mode == 'xglab':
    sys.path.append("/nfs/public/lichen/code/single_cell_perturbation/volc_scFlamingo/xpert_gene_perturbation/Byte_Pert_Data/")
else:
    raise ValueError()
    
import v1
from v1.utils import *
from v1.dataloader import *

from config import prefix_list, gene_pert_dict

import argparse

# importlib.reload(v1)
# importlib.reload(v1.utils)  
# importlib.reload(v1.dataloader)


###################################### functions
import torch
import torch.nn.functional as F

def masked_smooth_l1_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Compute the masked Huber / Smooth-L1 loss between `input` and `target`.

    Parameters
    ----------
    input  : (B, G) predicted expression values
    target : (B, G) ground-truth expression values
    mask   : (B, G) 0/1 (or bool) tensor indicating which positions contribute
    beta   : float, the transition point from L2 to L1 (δ, default 1.0)

    Returns
    -------
    torch.Tensor : scalar loss = sum(mask * SmoothL1) / mask.sum()
    """
    mask = mask.float()
    # PyTorch ≥1.9 支持 beta（δ）参数；较旧版本可用默认 beta=1
    loss = F.smooth_l1_loss(
        input * mask,
        target * mask,
        beta=beta,
        reduction="sum",
    )
    return loss / mask.sum()

def masked_focal_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the masked Focal-MSE loss between `input` and `target`.

    Parameters
    ----------
    input  : (B, G) predicted values
    target : (B, G) ground-truth values
    mask   : (B, G) 0/1 (or bool) tensor indicating valid positions
    gamma  : focusing parameter; γ>0 增强对大误差的关注 (γ≈1 常用)
    eps    : small constant to avoid zero gradients when |e|=0

    Returns
    -------
    torch.Tensor : scalar loss = sum(mask * (|e|+eps)^γ * e^2) / mask.sum()
    """
    mask = mask.float()
    err  = (input - target) * mask            # shape [B, G]
    abs_err = err.abs().clamp(min=eps)        # 避免零梯度
    weight = abs_err.pow(gamma)               # (|e| + eps)^γ

    loss = (weight * err.pow(2)).sum()        # Focal-MSE with mask
    return loss / mask.sum()

def masked_huber_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    delta: float = 0.5
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss_fn = nn.HuberLoss(delta=delta)
    loss = loss_fn(input * mask, target * mask)
    return loss / mask.sum()

from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))  # linear warm-up
        progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))  # cosine decay

    return LambdaLR(optimizer, lr_lambda)

def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(tqdm(train_loader)):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        # ori_gene_values = x[:, 0].view(batch_size, n_genes)
        ori_gene_values = x
        # pert_flags = x[:, 1].long().view(batch_size, n_genes)
        pert_flags = batch_data.pert_flags.long()
        target_gene_values = batch_data.y  # (batch_size, n_genes)


        batch_perts = batch_data.pert
        batch_perts = [[i.split('+')[0]] if 'ctrl' in i else i.split('+') for i in batch_perts] # we first focus on the single perts
        max_pert_len = max([len(perts) for perts in batch_perts])

        batch_pert_embed = torch.zeros(batch_size, max_pert_len, gpt_emb_dim).float()
        pert_mask = torch.ones(batch_size, max_pert_len)
        # batch_pert_embed = torch.tensor(np.array([(pert_embed_dict[pert[0]]) for pert in batch_perts])).float()
        for i, perts in enumerate(batch_perts):
            for j, pert in enumerate(perts):
                batch_pert_embed[i, j, :] = torch.tensor(np.array(pert_embed_dict[pert]))
                pert_mask[i, j] = 0
        batch_pert_embed = batch_pert_embed.to(device)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            if delta_mode:
                target_values = target_values - input_values

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                batch_pert_embed = batch_pert_embed,
                pert_mask = pert_mask,
                batch_dosages_pad = None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        if scheduler_type in ['cosine', 'cosine_warm']:
            scheduler.step()

        # torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:07.6f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.5f} | mse {cur_mse:5.5f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            # ori_gene_values = x[:, 0].view(batch_size, n_genes)
            ori_gene_values = x
            # pert_flags = x[:, 1].long().view(batch_size, n_genes)
            pert_flags = batch_data.pert_flags.long()
            target_gene_values = batch_data.y  # (batch_size, n_genes)
            
            
            batch_perts = batch_data.pert
            batch_perts = [[i.split('+')[0]] if 'ctrl' in i else i.split('+') for i in batch_perts] # we first focus on the single perts
            max_pert_len = max([len(perts) for perts in batch_perts])

            batch_pert_embed = torch.zeros(batch_size, max_pert_len, gpt_emb_dim).float()
            pert_mask = torch.ones(batch_size, max_pert_len)
            # batch_pert_embed = torch.tensor(np.array([(pert_embed_dict[pert[0]]) for pert in batch_perts])).float()
            for i, perts in enumerate(batch_perts):
                for j, pert in enumerate(perts):
                    batch_pert_embed[i, j, :] = torch.tensor(np.array(pert_embed_dict[pert]))
                    pert_mask[i, j] = 0
            batch_pert_embed = batch_pert_embed.to(device)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device)
                else:  # when batch-wise
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )

                # sample input_gene_id
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                if delta_mode:
                    target_values = target_values - input_values

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_pert_embed = batch_pert_embed,
                    pert_mask = pert_mask,
                    batch_dosages_pad = None,

                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=True,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss = criterion(output_values, target_values, masked_positions)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()
    return total_loss / len(val_loader), total_error / len(val_loader)


def pred_perturb_new(
    model,
    batch_data,
    include_zero_gene="batch-wise",
    gene_ids=None,
    amp=True,
):
    """
    Args:
        batch_data: a dictionary of input data with keys.

    Returns:
        output Tensor of shape [N, seq_len]
    """
    model.eval()
    device = next(model.parameters()).device
    batch_data.to(device)
    batch_size = len(batch_data.pert)
    x: torch.Tensor = batch_data.x
    # ori_gene_values = x[:, 0].view(batch_size, n_genes)
    ori_gene_values = x
    # pert_flags = x[:, 1].long().view(batch_size, n_genes)
    pert_flags = batch_data.pert_flags.long()


    batch_perts = batch_data.pert
    batch_perts = [[i.split('+')[0]] if 'ctrl' in i else i.split('+') for i in batch_perts] # we first focus on the single perts
    max_pert_len = max([len(perts) for perts in batch_perts])

    batch_pert_embed = torch.zeros(batch_size, max_pert_len, gpt_emb_dim).float()
    pert_mask = torch.ones(batch_size, max_pert_len)
    # batch_pert_embed = torch.tensor(np.array([(pert_embed_dict[pert[0]]) for pert in batch_perts])).float()
    for i, perts in enumerate(batch_perts):
        for j, pert in enumerate(perts):
            batch_pert_embed[i, j, :] = torch.tensor(np.array(pert_embed_dict[pert]))
            pert_mask[i, j] = 0
    batch_pert_embed = batch_pert_embed.to(device)

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
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                batch_pert_embed = batch_pert_embed,
                pert_mask = pert_mask,
                batch_dosages_pad = None,

                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=True,
            )
        output_values = output_dict["mlm_output"].float()
        pred_gene_values = torch.zeros_like(ori_gene_values)
        pred_gene_values[:, input_gene_ids] = output_values

        if delta_mode:
            pred_gene_values = input_values + pred_gene_values
    return pred_gene_values

def eval_perturb_new(
    loader: DataLoader, model: TransformerGenerator, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = pred_perturb_new(model, batch, include_zero_gene, gene_ids=gene_ids)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float64)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float64)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float64)

    return results


###################################### init paras
# init para

# - init dataloader para
data_dir = '/nfs/public/lichen/data/single_cell/perturb_data/scPerturb/raw/scPerturb_rna/statistic_20240520'
pert_cell_filter = 100 # this is used to filter perts, cell number less than this will be filtered
seed = 2024 # this is the random seed
split_type = 1 # 1 for unseen perts; 0 for unseen celltypes
split_ratio = [0.7, 0.2, 0.1] # train:test:val; val is used to choose data, test is for final validation
var_num = 5000 # selecting hvg number
num_de_genes = 20 # number of de genes
bs_train = 32 # batch size of trainloader
bs_test = bs_train * 2 # batch size of testloader
lr = 1e-4
add_control = False


# - init scGPT para
# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 2

n_hvg = 0  # number of highly variable genes
include_zero_gene = "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
# max_seq_len = 1536
max_seq_len = 6000


# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
cell_emb_style = "cls"
mvc_decoder_style = "inner product, detach"
amp = True

if machine_mode == 'volcano':
    load_model = "/hpc-cache-pfs/home/lichen/code/scFlamingo/other_code/scGPT_flamingo/scGPT/save/scGPT_human"
elif machine_mode == 'xglab':
    load_model = "/nfs/public/lichen/code/single_cell_perturbation/scGPT_flamingo/scGPT/save/scGPT_human"

load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
batch_size = 64
eval_batch_size = 64

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0.2  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

# logging
log_interval = 100

# dataset and evaluation choices
data_name = "adamson"
split = "simulation"
if data_name == "norman":
    perts_to_plot = ["SAMD1+ZBTB1"]
elif data_name == "adamson":
    perts_to_plot = ["KCTD16+ctrl"]

# - 恢复默认参数
plt.rcdefaults()


# # - multi gpu para
# n_gpu = 2
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # device_ids = list(range(n_gpu))
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device_ids = [2, 3]

# - training epoch
epochs = 40
max_seq_len = 6000
early_stop = epochs
save_flag = True
schedule_interval = 5
lr = 1e-4  # or 1e-4
gamma = 0.98

gpt_emb_dim = 1536

# delta_mode = False
# attn_gate_mode = True
# load_cxg_weight = False

model_dir = Path(load_model)
model_config_file = model_dir / "args.json"
with open(model_config_file, "r") as f:
    model_configs = json.load(f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="scF apply L1000")
    parser.add_argument('--model_mode', type=str, default=None)
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument('--n_gpus', nargs='+', type=int, help='List of integers')

    args = parser.parse_args()
    # args.model_mode = 'scFlamingo_v2'
    model_mode = args.model_mode

    # - multi gpu para
    n_gpu = len(args.n_gpus)
    device = torch.device(f"cuda:{args.n_gpus[0]}" if torch.cuda.is_available() else "cpu")
    device_ids = args.n_gpus

    # 'scFlamingo_v1': use cross attention, no pert add to the tokens
    # 'scFlamingo_v2': use cross attention, add pert embed to the tokens
    # 'scFlamingo_v3': use cross attention, no pert add to the tokens, set the gate to fixed 1
    # 'scFlamingo_v4': use cross attention, add pert embed to the tokens, no using the pretrain weight
    # 'scFlamingo_v5': use cross attention, add pert embed to the tokens, using the pretrain weight, predict delta

    # 'scFlamingo_v6': use cross attention, add pert embed to the tokens, add tokens


    delta_mode = False
    attn_gate_mode = True
    load_cxg_weight = True
    mask_mode = True

    pert_mode = 'gene' # drug; gene
    drug_embed_mode = 'grover' # gpt; grover; rdkit
    pert_flag_mode = True
    # delta_mode = False
    # attn_gate_mode = True
    # load_cxg_weight = True
    use_scgpt_layer = True
    use_scgpt_input = True
    use_scgpt_trans_weight = True
    save_every_epoch = False
    loss_type = 'mse'
    pert_data_version = 'pert_data_v1.pkl'
    split_col = 'data_split_1' # for sciplex_3: data_split_0: random split perts; data_split_1: split drugs; data_split_2: split cell types, A549 as the val set
    debug_mode_1 = False # make the whole code run fast in only one epoch
    # mask_mode = True
    add_token = True # whether add non-overlap genes as the new tokens
    dosage_mode = True # whether use dosages of the drug
    init_mode = False # change the init mode of scGPT model, ensure every layer be initialized
    dosage_mode_type = 0 # 0: direct multiply; 1: learnable vector

    load_encoder_plus = False # when add tokens, load weight to encoder_plus

    max_norm = 1.0
    loss_type = 'mse'
    scheduler_type = 'steplr' # 'steplr'; 'cosine'

    cross_mode = True # True: use cross attn; False: directly add pert to scgpt each token

    if model_mode == 'scFlamingo_v1':
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True
    elif model_mode == 'scFlamingo_v2':
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True
    elif model_mode == 'scFlamingo_v3':
        delta_mode = False
        attn_gate_mode = False
        load_cxg_weight = True
    elif model_mode == 'scFlamingo_v4':
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = False
    elif model_mode == 'scFlamingo_v5':
        delta_mode = True
        attn_gate_mode = True
        load_cxg_weight = True

    elif model_mode == 'scFlamingo_v6':
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
    elif model_mode == 'scFlamingo_v7':
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = False

        schedule_interval = 1
        lr = 1e-4  # or 1e-4
        gamma = 0.90
        epochs = 20

    elif model_mode == 'scFlamingo_v8': # add token but not load cxg weight
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = False

        pert_mode = 'gene'
        add_token = True

        schedule_interval = 1
        lr = 1e-4  # or 1e-4
        gamma = 0.98
        epochs = 20

    elif model_mode == 'scFlamingo_v9': # add token; load encoder_plus
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        schedule_interval = 1
        lr = 1e-4  # or 1e-4
        gamma = 0.98
        epochs = 20

    elif model_mode == 'scFlamingo_v10': # add token; load encoder_plus
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        schedule_interval = 1
        lr = 1e-4  # or 1e-4
        gamma = 0.90
        epochs = 20

    elif model_mode == 'scFlamingo_v11': # add token; load encoder_plus; huber loss
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        schedule_interval = 1
        lr = 1e-4  # or 1e-4
        gamma = 0.90
        epochs = 20

        loss_type = 'huber' # gamma = 0.5

    elif model_mode == 'scFlamingo_v12': # add token; load encoder_plus; cosine_warm scheduler
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        # schedule_interval = 1
        lr = 1e-4  # or 1e-4
        # gamma = 0.90
        epochs = 20

        scheduler_type = 'cosine_warm'

    elif model_mode == 'scFlamingo_v13': # add token; load encoder_plus; cosine_warm scheduler
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        # schedule_interval = 1
        lr = 5e-5  # or 1e-4
        # gamma = 0.90
        epochs = 20

        scheduler_type = 'cosine_warm'

    elif model_mode == 'scFlamingo_v13_test': # add token; load encoder_plus; cosine_warm scheduler
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        # schedule_interval = 1
        lr = 5e-5  # or 1e-4
        # gamma = 0.90
        epochs = 20

        scheduler_type = 'cosine_warm'

    elif model_mode == 'scFlamingo_v14': # add token; load encoder_plus; cosine_warm scheduler; use delta mode
        delta_mode = True
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        # schedule_interval = 1
        lr = 5e-5  # or 1e-4
        # gamma = 0.90
        epochs = 20

        scheduler_type = 'cosine_warm'

    elif model_mode == 'scFlamingo_v15': # add token; load encoder_plus; cosine_warm scheduler; use less layers
        delta_mode = False
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        # schedule_interval = 1
        lr = 5e-5  # or 1e-4
        # gamma = 0.90
        epochs = 20

        scheduler_type = 'cosine_warm'

        model_configs["nlayers"] = 2
        model_configs["nheads"] = 8

    elif model_mode == 'scFlamingo_v17': # add token; load encoder_plus; cosine_warm scheduler; use less layers; predict delta
        delta_mode = True
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        # schedule_interval = 1
        lr = 5e-5  # or 1e-4
        # gamma = 0.90
        epochs = 20

        scheduler_type = 'cosine_warm'

        model_configs["nlayers"] = 2
        model_configs["nheads"] = 8

    elif model_mode == 'scFlamingo_v18': # add token; load encoder_plus; cosine_warm scheduler; use less layers; predict delta; no pert_flag
        delta_mode = True
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        # schedule_interval = 1
        lr = 5e-5  # or 1e-4
        # gamma = 0.90
        epochs = 20

        scheduler_type = 'cosine_warm'

        model_configs["nlayers"] = 2
        model_configs["nheads"] = 8

        bs_train = 80
        pert_flag_mode = False

    elif model_mode == 'scFlamingo_v19': # add token; load encoder_plus; cosine_warm scheduler; use less layers; predict delta; no pert_flag; cross mode False
        delta_mode = True
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        # schedule_interval = 1
        lr = 5e-5  # or 1e-4
        # gamma = 0.90
        epochs = 20

        scheduler_type = 'cosine_warm'

        model_configs["nlayers"] = 2
        model_configs["nheads"] = 8

        bs_train = 80
        pert_flag_mode = False

        cross_mode = False
    elif model_mode == 'scFlamingo_v20': # add token; load encoder_plus; cosine_warm scheduler; use less layers; predict delta; add pert_flag; cross mode False
        delta_mode = True
        attn_gate_mode = True
        load_cxg_weight = True

        pert_mode = 'gene'
        add_token = True
        load_encoder_plus = True

        # schedule_interval = 1
        lr = 5e-5  # or 1e-4
        # gamma = 0.90
        epochs = 20

        scheduler_type = 'cosine_warm'

        model_configs["nlayers"] = 2
        model_configs["nheads"] = 8

        bs_train = 80
        pert_flag_mode = True

        cross_mode = False




    else:
        raise ValueError()


    for prefix in prefix_list:
        # prefix = 'ReplogleWeissman2022_K562_essential'
        # prefix = 'AdamsonWeissman2016_GSM2406675_10X001'
        # prefix = 'NormanWeissman2019_filtered'
        # prefix = 'SunshineHein2023'
        prefix = args.prefix
        print('='*20, prefix)
        fix_seed(2024)

        if machine_mode == 'volcano':
            tmp_dir = '/hpc-cache-pfs/home/lichen/data/scPerturb_data/single_dataset/process_pert_data_v2'
        elif machine_mode == 'xglab':
            tmp_dir = '/nfs/public/lichen/results/single_cell_perturbation/perturbation_benchmark/scPerturb'
        
        # - this is for load dataloader
        # save_prefix = f'GEARS'
        save_prefix = f'GEARS_v2-prefix_{prefix}-pert_cell_filter_{pert_cell_filter}-\
seed_{seed}-split_type_{split_type}-var_num_{var_num}-num_de_genes_{num_de_genes}-bs_train_32-\
bs_test_32'
        # save_prefix = f'GEARS_v2-multipert-select'
        # save_prefix = f'GEARS_v2-multipert-select_v2'
        # save_prefix = f'GEARS_v2-multipert-select_v4'
        # GEARS_v2-multipert: all single train, and all multi test
        # GEARS_v2-multipert-select: select 50% single train, and all multi test
        # GEARS_v2-multipert-select_v2: select 75% single train, and all multi test
        # GEARS_v2-multipert-select_v3: select 50% single train, unseen 0 also as trained; usneen_1 and unseen_2 as test
        # GEARS_v2-multipert-select_v4: select 70% single train, unseen 0 also as trained; usneen_1 and unseen_2 as test

        save_dir = os.path.join(tmp_dir, prefix, save_prefix)
        # - load pert_data
        pert_data = pickle.load(open(os.path.join(save_dir,'pert_data.pkl'), 'rb'))

        # - save in scGPT folder
        if machine_mode == 'volcano':
            tmp_dir = '/hpc-cache-pfs/home/lichen/result/scFlamingo/scPerturb_single_dataset'
        elif machine_mode == 'xglab':
            tmp_dir = '/nfs/public/lichen/results/single_cell_perturbation/perturbation_benchmark/volc_result/scFlamingo/scPerturb_single_dataset'
        save_prefix = f'model_mode_{model_mode}'
        # save_prefix = f'xpert-prefix_{prefix}-multipert-epoch_{str(epochs)}-gpu_{str(n_gpu)}-bs_{str(bs_train)}-select_v2'
        # save_prefix = f'model_mode_{model_mode}-multipert-epoch_{str(epochs)}-gpu_{str(n_gpu)}-bs_{str(bs_train)}-select_v4'
        # xpert-prefix_{prefix}-multipert; xpert-prefix_{prefix}-multipert-epoch_50

    #         save_prefix = f'scGPT-prefix_{prefix}-pert_cell_filter_{pert_cell_filter}-\
    # seed_{seed}-split_type_{split_type}-var_num_{var_num}-num_de_genes_{num_de_genes}-bs_train_{bs_train}-\
    # bs_test_{bs_test}-add_control_{add_control}' # correct the input and output
        save_dir = os.path.join(tmp_dir, prefix, save_prefix)
        os.makedirs(save_dir, exist_ok=True)
        print(f"saving to {save_dir}")

        # - add logger
        logger = scg.logger
        scg.utils.add_file_handler(logger, Path(save_dir) / "run.log")
        # log running date and current git commit
        logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # - get go genes; special set for GEARS
        pert_data.get_gene2go()
        pert_data.set_pert_genes()

        # - get dataset for scGPT
        pert_data.get_Data_scgpt(num_de_genes = pert_data.num_de_genes,
                                dataset_name = ['train', 'test', 'val'],
                                add_control = add_control)

        # - add necessary elements for gears
        pert_data.modify_gears()

        # - get dataloader
        trainloader, testloader, valloader = pert_data.get_dataloader(mode='all',
                                                                    bs_train = int(bs_train) * n_gpu,
                                                                    bs_test = int(bs_test) * n_gpu)

        ##############################################
        # - get the pert_embed_dict
        # -- read gpt embed
        if machine_mode == 'volcano':
            save_dir_2 = '/hpc-cache-pfs/home/lichen/data/utils_data/perturbation_embed/scPerturb/v1'
        elif machine_mode == 'xglab':
            save_dir_2 = '/nfs/public/lichen/results/single_cell_perturbation/perturbation_benchmark/volc_result/gpt_embed/scPerturb/v1'
        gene_embed = pd.read_csv(os.path.join(save_dir_2, prefix, 'pert_embed.csv'), sep = ",", index_col=0)
        # -- get all the pert_names
        total_perts = []
        for pert_list in [pert_data.train_perts, pert_data.test_perts, pert_data.val_perts]:
            for pert in pert_list:
                if ';' in pert:
                    total_perts.extend(pert.split('; '))
                else:
                    total_perts.append(pert)
        total_perts = np.unique(total_perts)       
        print(len(total_perts))

        # -- create pert_embed_dict
        pert_embed_dict = {}
        np.random.seed(2024)
        for pert in total_perts:
            if pert in gene_embed.columns:
                pert_embed_dict[pert] = gene_embed.loc[:, pert].values
            else:
                print(f'{pert} not in gene_embed')
                pert_embed_dict[pert] = gene_embed.loc[:, np.random.choice(gene_embed.columns, 1)[0]].values

        ##############################################
        # - get the model 
        model_dir = Path(load_model)


        # - get the vocab
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)

        # token2idx = json.load(vocab_file)
        with vocab_file.open("r") as f:
            token2idx = json.load(f)
        print('length of the token is :',len(token2idx))


        # - add token to the vocab
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        vocab_ori = copy.deepcopy(vocab)

        if add_token:
            # - add non-overlap genes to the vocab
            add_genes = np.setdiff1d(pert_data.adata.var_names, list(vocab.get_stoi().keys()))
            for gene in add_genes:
                if gene not in vocab:
                    vocab.append_token(gene)

        # pert_data.adata.var["id_in_vocab"] = [
        #     1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
        # ]
        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var_names
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        genes = pert_data.adata.var["gene_name"].tolist()


        # - load the model
        # with open(model_config_file, "r") as f:
        #     model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]

        # - get token of input genes
        vocab.set_default_index(vocab["<pad>"])
        gene_ids = np.array(
            [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
        )
        n_genes = len(genes)


        # - intial the model

        # ntokens = len(vocab)  # size of vocabulary
        ntokens = len(vocab_ori)  # size of vocabulary
        model = TransformerGenerator(
            ntokens,
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=n_layers_cls,
            n_cls=1,
            vocab=vocab,
            dropout=dropout,
            pad_token=pad_token,
            pad_value=pad_value,
            pert_pad_id=pert_pad_id,
            do_mvc=MVC,
            cell_emb_style=cell_emb_style,
            mvc_decoder_style=mvc_decoder_style,
            use_fast_transformer=use_fast_transformer,

            pert_embed_dict = pert_embed_dict,
            gpt_emb_dim = gpt_emb_dim,
            model_mode = model_mode,
            attn_gate_mode = attn_gate_mode,
            pert_mode = pert_mode,
            drug_embed_mode = drug_embed_mode,
            pert_flag_mode = pert_flag_mode,
            use_scgpt_layer = use_scgpt_layer,
            use_scgpt_input = use_scgpt_input,

            mask_mode = mask_mode,
            add_token = add_token,

            init_mode = init_mode,

            dosage_mode_type = dosage_mode_type,

            cross_mode = cross_mode,
        )
        if load_cxg_weight:
            # - load the model
            if load_param_prefixs is not None and load_model is not None:
                # only load params that start with the prefix
                model_dict = model.state_dict()
                pretrained_dict = torch.load(model_file)
                pretrained_dict = {
                    k: v
                    for k, v in pretrained_dict.items()
                    if any([k.startswith(prefix) for prefix in load_param_prefixs])
                }
                for k, v in pretrained_dict.items():
                    logger.info(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)

                # model.load_state_dict(model_dict)

                # 加载修改后的权重
                missing_keys_info = model.load_state_dict(model_dict, strict=False)
                print("Missing keys:", missing_keys_info.missing_keys)
                print("Unexpected keys:", missing_keys_info.unexpected_keys)

            if add_token and load_encoder_plus:
                with torch.no_grad():
                    # Step 1: 复制前 n 个 embedding 和 norm
                    n = model.encoder.embedding.num_embeddings
                    model.encoder_plus.embedding.weight[:n] = model.encoder.embedding.weight
                    model.encoder_plus.enc_norm.weight[:] = model.encoder.enc_norm.weight
                    model.encoder_plus.enc_norm.bias[:] = model.encoder.enc_norm.bias

                    # Step 2: 获取前 n 个 embedding 的均值和标准差
                    pretrained_embed = model.encoder.embedding.weight  # shape: (n, d)
                    mean = pretrained_embed.mean(dim=0)          # shape: (d,)
                    std = pretrained_embed.std(dim=0)            # shape: (d,)

                    # Step 3: 初始化新 token（从 n 到 end）为相同均值和方差
                    m = model.encoder_plus.embedding.num_embeddings - n
                    model.encoder_plus.embedding.weight[n:] = torch.normal(
                        mean=mean.expand(m, -1),
                        std=std.expand(m, -1)
                    )


        # - add the parallel
        model = torch.nn.DataParallel(model, device_ids=device_ids)

        # - put model on device
        model.to(device)


        # - intial the loss
        # criterion = masked_mse_loss
        if loss_type == 'mse':
            criterion = masked_mse_loss
        elif loss_type == 'huber':
            criterion = masked_huber_loss
        else:
            raise ValueError()
        criterion_cls = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if scheduler_type == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=gamma)
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader)*epochs)
        elif scheduler_type == 'cosine_warm':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(len(trainloader)*epochs*0.05),
                num_training_steps=len(trainloader)*epochs
            )
        else:
            raise ValueError()
        
        scaler = torch.cuda.amp.GradScaler(enabled=amp)


        best_val_loss = float("inf")
        best_model = None
        patience = 0

        train_metrics_list, train_metrics_pert_list, val_metrics_list, val_metrics_pert_list = [], [], [], []

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train_loader = pert_data.dataloader["train_loader"]
            valid_loader = pert_data.dataloader["val_loader"]

            train(
                model,
                train_loader,
            )
            val_loss, val_mre = evaluate(
                model,
                valid_loader,
            )
            elapsed = time.time() - epoch_start_time
            logger.info("-" * 89)
            logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} |"
            )
            logger.info("-" * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                logger.info(f"Best model with score {best_val_loss:5.4f}")
                patience = 0

                torch.save(
                    best_model.state_dict(),
                    Path(save_dir) / f"model_best.pt",
                )
            else:
                patience += 1
                if patience >= early_stop:
                    logger.info(f"Early stop at epoch {epoch}")
                    break
            
            # - save the results
            # test_loader = pert_data.dataloader["test_loader"]
            train_res = eval_perturb_new(train_loader, model, device)
            val_res = eval_perturb_new(valid_loader, model, device)
            train_metrics, train_metrics_pert = compute_metrics(train_res)
            val_metrics, val_metrics_pert = compute_metrics(val_res)
            
            train_metrics_list.append(train_metrics)
            train_metrics_pert_list.append(train_metrics_pert)
            val_metrics_list.append(val_metrics)
            val_metrics_pert_list.append(val_metrics_pert)


            # torch.save(
            #     model.state_dict(),
            #     Path(save_dir) / f"model_{epoch}.pt",
            # )
            if scheduler_type == 'steplr':
                scheduler.step()

        torch.save(
            best_model.state_dict(),
            Path(save_dir) / f"model_best.pt",
        )
        test_loader = pert_data.dataloader["test_loader"]
        test_res = eval_perturb_new(test_loader, best_model, device)
        # test_metrics, test_pert_res = compute_metrics(test_res)

        # out = get_metric(pert_data.adata, test_res)
        # out_non_dropout = non_dropout_analysis(pert_data.adata, test_res)

        # - save results
        model_prefix = f'lr_{lr}' 
        os.makedirs(os.path.join(save_dir,f'result_{model_prefix}'),exist_ok=True)

        if save_flag:
            # -- save pkl
            pickle.dump(test_res, open(os.path.join(save_dir,f'result_{model_prefix}/test_res.pkl'), 'wb'))
            pickle.dump(train_metrics_list, open(os.path.join(save_dir,f'result_{model_prefix}/train_metrics_list.pkl'), 'wb'))
            pickle.dump(train_metrics_pert_list, open(os.path.join(save_dir,f'result_{model_prefix}/train_metrics_pert_list.pkl'), 'wb'))
            pickle.dump(val_metrics_list, open(os.path.join(save_dir,f'result_{model_prefix}/val_metrics_list.pkl'), 'wb'))
            pickle.dump(val_metrics_pert_list, open(os.path.join(save_dir,f'result_{model_prefix}/val_metrics_pert_list.pkl'), 'wb'))
            # pickle.dump(out, open(os.path.join(save_dir,f'result_{model_prefix}/out.pkl'), 'wb'))
            # pickle.dump(out_non_dropout, open(os.path.join(save_dir,f'result_{model_prefix}/out_non_dropout.pkl'), 'wb'))

            # -- save plot
            merge_plot(train_metrics_list, 'train',os.path.join(save_dir,f'result_{model_prefix}/train.png'))
            merge_plot(val_metrics_list, 'test',os.path.join(save_dir,f'result_{model_prefix}/test.png'))

        else:
            # -- save plot
            merge_plot(train_metrics_list, 'train',None)
            merge_plot(val_metrics_list, 'test',None)

        torch.cuda.empty_cache()

        break