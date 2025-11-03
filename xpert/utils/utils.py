import scanpy as sc
import pandas as pd
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys
import requests
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import random
import torch

def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True 

def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush = True, file = sys.stderr)

def dataverse_download(url, save_path):
    """
    Dataverse download helper with progress bar

    Args:
        url (str): the url of the dataset
        path (str): the path to save the dataset
    """
    
    if os.path.exists(save_path):
        print_sys('Found local copy...')
    else:
        print_sys("Downloading...")
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
def get_genes_from_perts(perts):
    """
    Returns list of genes involved in a given perturbation list
    """

    if type(perts) is str:
        perts = [perts]
    gene_list = [p.split('+') for p in np.unique(perts)]
    gene_list = [item for sublist in gene_list for item in sublist]
    gene_list = [g for g in gene_list if g != 'ctrl']
    return list(np.unique(gene_list))

def get_dropout_non_zero_genes(adata):
    
    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]
    
    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    pert_full_id2pert = dict(adata.obs[['condition_name', 'condition']].values)

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    for pert in adata.uns['rank_genes_groups'].keys():
        p = pert_full_id2pert[pert]
        # X = np.mean(adata[adata.obs.condition == p].X, axis = 0)
        X = np.mean(adata[adata.obs.condition_name == pert].X, axis = 0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top = adata.uns['rank_genes_groups'][pert]
        gene_idx_top = [gene_id2idx[i] for i in top]

        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
        non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]

        non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
        non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)
        
    non_zero = np.where(np.array(X)[0] != 0)[0]
    zero = np.where(np.array(X)[0] == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))
    
    adata.uns['top_non_dropout_de_20'] = top_non_dropout_de_20
    adata.uns['non_dropout_gene_idx'] = non_dropout_gene_idx
    adata.uns['non_zeros_gene_idx'] = non_zeros_gene_idx
    adata.uns['top_non_zero_de_20'] = top_non_zero_de_20
    
    return adata

def get_pert_celltype(pert_group_list):
    perts = np.unique([i.split(' | ')[0] for i in pert_group_list])
    celltypes = np.unique([i.split(' | ')[1] for i in pert_group_list])
    return perts, celltypes


def plot_loss(tmp_list,
              # test_list,
              dataset,
              name):
    
    import matplotlib.pyplot as plt
    # 假设损失函数值存储在 loss_values 中
    # plt.figure(figsize=(8,6))

    # 创建 x 轴的值（可以是迭代次数、时间步长等）
    iterations = range(1, len(tmp_list) + 1)

    # 绘制损失函数曲线
    plt.plot(iterations, tmp_list, 
            #  marker='o', 
             linestyle='-', 
             label='train')
    # plt.plot(iterations, test_list, marker='o', linestyle='-', label='test')
    

    # 添加标题和标签
    plt.title(f'{name} Curve - {dataset}')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    # plt.legend()

    # 显示网格
    plt.grid(True)

    # # 显示图形
    # plt.show()
    
def merge_plot(metrics_list, dataset, save_dir=None):
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    tmp_list = [i['mse'] for i in metrics_list]
    plot_loss(tmp_list, dataset, 'MSE')

    plt.subplot(2, 2, 2)
    tmp_list = [i['pearson'] for i in metrics_list]
    plot_loss(tmp_list, dataset, 'PCC')

    plt.subplot(2, 2, 3)
    tmp_list = [i['mse_de'] for i in metrics_list]
    plot_loss(tmp_list, dataset, 'MSE_de')

    plt.subplot(2, 2, 4)
    tmp_list = [i['pearson_de'] for i in metrics_list]
    plot_loss(tmp_list, dataset, 'PCC_de')

    # 调整子图之间的间距
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir)
        
def get_info_txt(pert_data = None, 
                 save_dir = None):

    # - get the para setting
    basic_para = f'\
    pert_cell_filter: {pert_data.pert_cell_filter} # this is a test \n\
    seed:             {pert_data.seed} # this is the random seed\n\
    split_type:       {pert_data.split_type} # 1 for unseen perts; 0 for unseen celltypes \n\
    split_ratio:      {pert_data.split_ratio} # train:test:val; val is used to choose data, test is for final validation \n\
    var_num:          {pert_data.var_num} # selecting hvg number \n\
    num_de_genes:     {pert_data.num_de_genes} # number of de genes \n\
    bs_train:         {pert_data.bs_train} # batch size of trainloader \n\
    bs_test:          {pert_data.bs_test} # batch size of testloader \n\
    '

    # - get the pert info
    ori_pert_num = pert_data.value_counts.shape[0]
    filtered_filter_num = pert_data.value_counts[pert_data.value_counts['Counts']<pert_data.pert_cell_filter].shape[0]
    final_pert_num = ori_pert_num - filtered_filter_num
    pert_num = len(pert_data.filter_perturbation_list)
    pert_info = f'\
    Number of original perts is: {ori_pert_num}\n\
    Number of filtered perts is: {filtered_filter_num}\n\
    After filter, number of non-ctrl perts: {pert_num}\n\
    After filter, number of pert genes: {len(pert_data.pert_gene_list)}\n\
    '
    # print(pert_info)


    # - get the gene_info
    exclude_var_num = len(pert_data.exclude_var_list)
    exclude_pert_num = 0
    for pert in pert_data.filter_perturbation_list:
        for gene in pert.split(' | ')[0].split('; '):
            if gene not in pert_data.adata_split.var_names:
                exclude_pert_num += 1
                break

    exclude_gene_num_gears = 0
    for gene in pert_data.pert_gene_list:
        if gene not in pert_data.pert_names:
            exclude_gene_num_gears += 1
    
    exclude_pert_num_gears = 0
    for pert in pert_data.filter_perturbation_list:
        for gene in pert.split(' | ')[0].split('; '):
            if gene not in pert_data.pert_names:
                exclude_pert_num_gears += 1
                break
            
    gene_info = f'\
    {exclude_var_num}/{len(pert_data.pert_gene_list)} genes are not in var names\n\
    {exclude_pert_num}/{len(pert_data.filter_perturbation_list)} perts are not in var names\n\
    \n\
    {exclude_gene_num_gears}/{len(pert_data.pert_gene_list)} genes are not in GEARS.pert_names\n\
    {exclude_pert_num_gears}/{len(pert_data.filter_perturbation_list)} perts are not in GEARS.pert_names\n\
    '
    # print(gene_info)

    # - get the adata_info
    pert_count_dict = {}
    for pert in pert_data.filter_perturbation_list:
        pert_count_dict[pert] = len(pert_data.adata_split[pert_data.adata_split.obs['perturbation_group']==pert])
    ctrl_count = len(pert_data.adata_split[pert_data.adata_split.obs['perturbation_new']=='control'])
    adata_info = f'\
    adata shape: {(pert_data.adata_split.shape)}\n\
    pert average cell num: {int(np.mean(list(pert_count_dict.values())))}\n\
    ctrl cell num: {int(ctrl_count)}\n\
    '
    # print(adata_info)

    from tabulate import tabulate

    # - get the sgRNA info
    sgRNA_info = f'\
    {len(pert_data.multi_sgRNA_pert_list)}/{len(pert_data.filter_perturbation_list)} perts have more than 1 sgRNA\n\
    {len(pert_data.df_sgRNA_edis_dict)}/{len(pert_data.multi_sgRNA_pert_list)} perts are in the var_names\n\
    {len(pert_data.filter_pert_sgRNA)} pert_sgRNA are filtered\n\
    '
    # print(sgRNA_info)

    # - get the split info
    split_info = f'\
    perts num of train: val: test = {len(pert_data.train_perts)}: {len(pert_data.val_perts)}: {len(pert_data.test_perts)} \n\
    '
    # print(split_info)

    # 指定文件名
    filename = os.path.join(save_dir, f"INFO_{pert_data.prefix}.txt")

    # 将信息写入文件
    with open(filename, "w") as file:
        file.write('*'*20+'Parameter Setting'+'*'*20+'\n')
        file.write(basic_para)
        file.write('\n')
        
        file.write('*'*20+'Pert Info'+'*'*20+'\n')
        file.write(pert_info)
        file.write('\n')
        
        file.write('*'*20+'Gene Info'+'*'*20+'\n')
        file.write(gene_info)
        file.write('\n')
        
        file.write('*'*20+'adata Info'+'*'*20+'\n')
        file.write(adata_info)
        file.write('\n')
        
        file.write('*'*20+'Data Split Info'+'*'*20+'\n')
        file.write(split_info)
        file.write('\n')
        
        file.write('*'*20+'Top 10 pert of edis'+'*'*20+'\n')
        file.write(tabulate(pert_data.df_pert_edis[0:10].round(2), headers='keys', tablefmt='mixed_grid'))
        file.write('\n')
        
        file.write('*'*20+'sgRNA Info'+'*'*20+'\n')
        file.write(sgRNA_info)
        file.write('\n')
        
        
def deeper_analysis_new(adata, test_res, de_column_prefix = 'rank_genes_groups_cov', most_variable_genes = None):
    
    metric2fct = {
           'pearson': pearsonr,
           'mse': mse
    }

    pert_metric = {}

    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]] # 这是adata中所有ctrl的mean
    
    if most_variable_genes is None:
        most_variable_genes = np.argsort(np.std(mean_expression, axis = 0))[-200:]
        
    gene_list = adata.var['gene_name'].values

    # 可能这里的ctrl需要修改
    for pert in np.unique(test_res['pert_cat']):
        metric2fct = {
        'pearson': pearsonr,
        'mse': mse
        }
        pert_metric[pert] = {}
        de_idx = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:20]]
        de_idx_200 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:200]]
        de_idx_100 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:100]]
        de_idx_50 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:50]]
        de_idx_all = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:]]

        pert_idx = np.where(test_res['pert_cat'] == pert)[0]    
        pred_mean = np.mean(test_res['pred_de'][pert_idx], axis = 0).reshape(-1,)
        true_mean = np.mean(test_res['truth_de'][pert_idx], axis = 0).reshape(-1,)
        
        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0) - ctrl[0]) - np.sign(test_res['truth'][pert_idx].mean(0) - ctrl[0]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(geneid2name)
        pert_metric[pert]['frac_correct_direction_all'] = frac_correct_direction

        de_idx_map = {'top20_de': de_idx,
                      'top50_de': de_idx_50,
                      'top100_de': de_idx_100,
                      'top200_de': de_idx_200,
                      'all_de':de_idx_all,
                     }
        
        for val in list(de_idx_map.keys()):
            
            direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[de_idx_map[val]] - ctrl[0][de_idx_map[val]]) - np.sign(test_res['truth'][pert_idx].mean(0)[de_idx_map[val]] - ctrl[0][de_idx_map[val]]))            
            frac_correct_direction = len(np.where(direc_change == 0)[0])/len(de_idx_map[val])
            pert_metric[pert]['frac_correct_direction_' + str(val)] = frac_correct_direction

        mean = np.mean(test_res['truth_de'][pert_idx], axis = 0)
        std = np.std(test_res['truth_de'][pert_idx], axis = 0)
        min_ = np.min(test_res['truth_de'][pert_idx], axis = 0)
        max_ = np.max(test_res['truth_de'][pert_idx], axis = 0)
        q25 = np.quantile(test_res['truth_de'][pert_idx], 0.25, axis = 0)
        q75 = np.quantile(test_res['truth_de'][pert_idx], 0.75, axis = 0)
        q55 = np.quantile(test_res['truth_de'][pert_idx], 0.55, axis = 0)
        q45 = np.quantile(test_res['truth_de'][pert_idx], 0.45, axis = 0)
        q40 = np.quantile(test_res['truth_de'][pert_idx], 0.4, axis = 0)
        q60 = np.quantile(test_res['truth_de'][pert_idx], 0.6, axis = 0)

        zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
        nonzero_des = np.setdiff1d(list(range(20)), zero_des)
        if len(nonzero_des) == 0:
            pass
            # pert that all de genes are 0...
        else:            
            # 原来这个nonzero是不变化的gene；以及这里都是对top 20 de gene的分析
            direc_change = np.abs(np.sign(pred_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]) - np.sign(true_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]))            
            frac_correct_direction = len(np.where(direc_change == 0)[0])/len(nonzero_des)
            pert_metric[pert]['frac_correct_direction_20_nonzero'] = frac_correct_direction
            
            in_range = (pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des])
            frac_in_range = sum(in_range)/len(nonzero_des)
            pert_metric[pert]['frac_in_range'] = frac_in_range

            in_range_5 = (pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des])
            frac_in_range_45_55 = sum(in_range_5)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_45_55'] = frac_in_range_45_55

            in_range_10 = (pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des])
            frac_in_range_40_60 = sum(in_range_10)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_40_60'] = frac_in_range_40_60

            in_range_25 = (pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des])
            frac_in_range_25_75 = sum(in_range_25)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_25_75'] = frac_in_range_25_75

            zero_idx = np.where(std > 0)[0]
            sigma = (np.abs(pred_mean[zero_idx] - mean[zero_idx]))/(std[zero_idx])
            pert_metric[pert]['mean_sigma'] = np.mean(sigma)
            pert_metric[pert]['std_sigma'] = np.std(sigma)
            pert_metric[pert]['frac_sigma_below_1'] = 1 - len(np.where(sigma > 1)[0])/len(zero_idx)
            pert_metric[pert]['frac_sigma_below_2'] = 1 - len(np.where(sigma > 2)[0])/len(zero_idx)

        ## correlation on delta
        p_idx = np.where(test_res['pert_cat'] == pert)[0]

        # 感觉这一部分和之前的有重复，怪怪的
        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(test_res['pred'][p_idx].mean(0)- ctrl[0], test_res['truth'][p_idx].mean(0)-ctrl[0])[0]
                if np.isnan(val):
                    val = 0

                pert_metric[pert][m + '_delta'] = val
                
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0

                pert_metric[pert][m + '_delta_de'] = val

        # 感觉这里提出fold change是很奇怪的指标，暂时没想明白含义
        
        ## up fold changes > 10?
        pert_mean = np.mean(test_res['truth'][p_idx], axis = 0).reshape(-1,)

        fold_change = pert_mean/ctrl
        fold_change[np.isnan(fold_change)] = 0
        fold_change[np.isinf(fold_change)] = 0
        ## this is to remove the ones that are super low and the fold change becomes unmeaningful
        fold_change[0][np.where(pert_mean < 0.5)[0]] = 0

        o =  np.where(fold_change[0] > 0)[0]

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_all'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))


        o = np.intersect1d(np.where(fold_change[0] <0.333)[0], np.where(fold_change[0] > 0)[0])

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_downreg_0.33'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))


        o = np.intersect1d(np.where(fold_change[0] <0.1)[0], np.where(fold_change[0] > 0)[0])

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_downreg_0.1'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        o = np.where(fold_change[0] > 3)[0]

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_upreg_3'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        o = np.where(fold_change[0] > 10)[0]

        pred_fc = test_res['pred'][p_idx].mean(0)[o]
        true_fc = test_res['truth'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_upreg_10'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        # 这种most variable gene需要关注吗？不是很明白；计算的是所有pert的most variable，直接根据std来计算
        
        ## most variable genes
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[most_variable_genes] - ctrl[0][most_variable_genes], test_res['truth'][p_idx].mean(0)[most_variable_genes]-ctrl[0][most_variable_genes])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top200_hvg'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[most_variable_genes], test_res['truth'][p_idx].mean(0)[most_variable_genes])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top200_hvg'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[most_variable_genes], test_res['truth'][p_idx].mean(0)[most_variable_genes])
                pert_metric[pert][m + '_top200_hvg'] = val


        ## top 20/50/100/200 DEs 这里是非常关键的部分，之后也可以从这里看
        for prefix in list(de_idx_map.keys()):
            de_idx = de_idx_map[prefix]
            for m, fct in metric2fct.items():
                if m != 'mse':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][m + f'_delta_{prefix}'] = val


                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][m + f'_{prefix}'] = val
                else:
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])
                    pert_metric[pert][m + f'_{prefix}'] = val
        
        
        # - 添加我们的分组
        p_thre_1 = 0.001
        p_thre_2 = 0.1
        adata_pert = adata[adata.obs['perturbation_group']==pert2pert_full_id[pert]]
        adata_ctrl = adata[list(adata_pert.obs['control_barcode'])]
        names = adata.uns['rank_genes_groups'][pert2pert_full_id[pert]]
        pvals_adj = adata.uns['pvals_adj'][pert2pert_full_id[pert]]
        
        expr_df = pd.DataFrame({'gene':adata_ctrl.var_names,
                                'expr':np.array(np.mean(adata_ctrl.X, axis=0)).ravel()})
        pval_df = pd.DataFrame({'gene':names,
                                'pvals_adj':np.array(pvals_adj)})
        expr_df.index = expr_df['gene']
        pval_df.index = pval_df['gene']
        
        # finally we choose de genes with rules in this df
        gene_df = pd.merge(pval_df,expr_df,how='left',left_index=True,right_index=True)
        # gene_df = gene_df.sort_values(by='pvals_adj', ascending=True)

        gene_df_sig = gene_df[gene_df['pvals_adj']<p_thre_1]
        gene_df_nonsig = gene_df[gene_df['pvals_adj']>=p_thre_2]

        gene_df_nonsig = gene_df_nonsig.sort_values(by='expr', ascending=False)

        # 1. de part
        if len(gene_df_sig) < 20:
            gene_list = list(gene_df.index)[0:20]
        else:
            gene_list = list(gene_df_sig.index)
        idx_list = [geneid2idx[i] for i in gene_list]
        idx_top20 = [geneid2idx[i] for i in list(gene_df.index)[0:20]]
        idx_top50 = [geneid2idx[i] for i in list(gene_df.index)[0:50]]
        idx_top100 = [geneid2idx[i] for i in list(gene_df.index)[0:100]]
        idx_top200 = [geneid2idx[i] for i in list(gene_df.index)[0:200]]
        de_idx_map = {'top20': idx_top20,
                      'top50': idx_top50,
                      'top100': idx_top100,
                      'top200': idx_top200,
                      'sig_DE': idx_list,
                     }
        metric2fct = {
            'pearson': pearsonr,
            'mse': mse,
            'mae': mae,
            'change_ratio': get_change_ratio,
            'spearman': spearmanr,  
        }
        name = 'DE_'
        for prefix in list(de_idx_map.keys()):
            de_idx = de_idx_map[prefix]
            for m, fct in metric2fct.items():
                if m == 'pearson' or m == 'spearman':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_delta_{prefix}'] = val


                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                elif m == 'change_ratio':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                else:
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val
        
        # 2. non-de & high express
        if len(gene_df_nonsig) < 200: # if most genes are de genes
            gene_df_exp = gene_df_sig.sort_values(by='expr', ascending=False)
            idx_top20 = [geneid2idx[i] for i in list(gene_df_exp.index)[0:20]]
            idx_top50 = [geneid2idx[i] for i in list(gene_df_exp.index)[0:50]]
            idx_top100 = [geneid2idx[i] for i in list(gene_df_exp.index)[0:100]]
            idx_top200 = [geneid2idx[i] for i in list(gene_df_exp.index)[0:200]]
        else:
            idx_top20 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[0:20]]
            idx_top50 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[0:50]]
            idx_top100 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[0:100]]
            idx_top200 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[0:200]]
            
        de_idx_map = {'top20': idx_top20,
                      'top50': idx_top50,
                      'top100': idx_top100,
                      'top200': idx_top200,
                     }
        metric2fct = {
            'pearson': pearsonr,
            'mse': mse,
            'mae': mae,
            'change_ratio': get_change_ratio,
            'spearman': spearmanr,  
        }
        name = 'NonDE-high_'
        for prefix in list(de_idx_map.keys()):
            de_idx = de_idx_map[prefix]
            for m, fct in metric2fct.items():
                if m == 'pearson' or m == 'spearman':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_delta_{prefix}'] = val


                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                elif m == 'change_ratio':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                else:
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val
        
        # 3. non-de & low express
        if len(gene_df_nonsig) < 200: # if most genes are de genes
            gene_df_exp = gene_df_sig.sort_values(by='expr', ascending=False)
            idx_top20 = [geneid2idx[i] for i in list(gene_df_exp.index)[::-1][0:20]]
            idx_top50 = [geneid2idx[i] for i in list(gene_df_exp.index)[::-1][0:50]]
            idx_top100 = [geneid2idx[i] for i in list(gene_df_exp.index)[::-1][0:100]]
            idx_top200 = [geneid2idx[i] for i in list(gene_df_exp.index)[::-1][0:200]]
        else:
            idx_top20 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[::-1][0:20]]
            idx_top50 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[::-1][0:50]]
            idx_top100 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[::-1][0:100]]
            idx_top200 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[::-1][0:200]]
        de_idx_map = {'top20': idx_top20,
                      'top50': idx_top50,
                      'top100': idx_top100,
                      'top200': idx_top200,
                     }
        metric2fct = {
            'pearson': pearsonr,
            'mse': mse,
            'mae': mae,
            'change_ratio': get_change_ratio,
            'spearman': spearmanr,  
        }
        name = 'NonDE-low_'
        for prefix in list(de_idx_map.keys()):
            de_idx = de_idx_map[prefix]
            for m, fct in metric2fct.items():
                if m == 'pearson' or m == 'spearman':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_delta_{prefix}'] = val


                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                elif m == 'change_ratio':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                else:
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val

    return pert_metric


def get_change_ratio(pred, truth):
    truth, pred = np.array(truth), np.array(pred)
    # change_ratio = np.array([0 if np.isinf(i) else i for i in (np.abs(pred-truth)/truth)])
    change_ratio = np.abs(pred-truth)/truth
    change_ratio[np.isnan(change_ratio)] = 0
    change_ratio[change_ratio>5] = 5
    return np.mean(change_ratio)

def import_TF_data(TF_info_matrix=None, TF_info_matrix_path=None, TFdict=None):
    """
    Load data about potential-regulatory TFs.
    You can import either TF_info_matrix or TFdict.
    For more information on how to make these files, please see the motif analysis module within the celloracle tutorial.

    Args:
        TF_info_matrix (pandas.DataFrame): TF_info_matrix.

        TF_info_matrix_path (str): File path for TF_info_matrix (pandas.DataFrame).

        TFdict (dictionary): Python dictionary of TF info.
    """

#     if self.adata is None:
#         raise ValueError("Please import scRNA-seq data first.")

#     if len(self.TFdict) != 0:
#         print("TF dict already exists. The old TF dict data was deleted. \n")

    if not TF_info_matrix is None:
        tmp = TF_info_matrix.copy()
        tmp = tmp.drop(["peak_id"], axis=1)
        tmp = tmp.groupby(by="gene_short_name").sum()
        TFdict = dict(tmp.apply(lambda x: x[x>0].index.values, axis=1))

    if not TF_info_matrix_path is None:
        tmp = pd.read_parquet(TF_info_matrix_path)
        tmp = tmp.drop(["peak_id"], axis=1)
        tmp = tmp.groupby(by="gene_short_name").sum()
        TFdict = dict(tmp.apply(lambda x: x[x>0].index.values, axis=1))

#     if not TFdict is None:
#         self.TFdict=TFdict.copy()

#     # Update summary of TFdata
#     self._process_TFdict_metadata()
    return TFdict

def transform_name(my_str):
    if '+' in my_str:
        tmp_list = [i for i in my_str.split('+') if i!='ctrl']
        return '; '.join(tmp_list) + ' | ' + 'lymphoblasts'
    if ' | ' in my_str:
        tmp_list = [i for i in my_str.split(' | ')[0].split('; ')]
        if len(tmp_list)==1:
            return '+'.join([tmp_list[0], 'ctrl'])
        else:
            return '+'.join(tmp_list)
        
def non_dropout_analysis(adata, test_res):
    metric2fct = {
           'pearson': pearsonr,
           'mse': mse
    }

    pert_metric = {}
    
    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]
    
    gene_list = adata.var['gene_name'].values

    for pert in np.unique(test_res['pert_cat']):
        pert_metric[pert] = {}
        
        pert_idx = np.where(test_res['pert_cat'] == pert)[0]    
        de_idx = [geneid2idx[i] for i in adata.uns['top_non_dropout_de_20'][pert2pert_full_id[pert]]]
        non_zero_idx = adata.uns['non_zeros_gene_idx'][pert2pert_full_id[pert]]
        non_dropout_gene_idx = adata.uns['non_dropout_gene_idx'][pert2pert_full_id[pert]]
             
        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]) - np.sign(test_res['truth'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(de_idx)
        pert_metric[pert]['frac_correct_direction_top20_non_dropout'] = frac_correct_direction
        
        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(de_idx)
        pert_metric[pert]['frac_opposite_direction_top20_non_dropout'] = frac_direction_opposite
        
        frac_direction_opposite = len(np.where(direc_change == 1)[0])/len(de_idx)
        pert_metric[pert]['frac_0/1_direction_top20_non_dropout'] = frac_direction_opposite
        
        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[non_zero_idx] - ctrl[0][non_zero_idx]) - np.sign(test_res['truth'][pert_idx].mean(0)[non_zero_idx] - ctrl[0][non_zero_idx]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(non_zero_idx)
        pert_metric[pert]['frac_correct_direction_non_zero'] = frac_correct_direction

        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(non_zero_idx)
        pert_metric[pert]['frac_opposite_direction_non_zero'] = frac_direction_opposite
        
        frac_direction_opposite = len(np.where(direc_change == 1)[0])/len(non_zero_idx)
        pert_metric[pert]['frac_0/1_direction_non_zero'] = frac_direction_opposite
        
        direc_change = np.abs(np.sign(test_res['pred'][pert_idx].mean(0)[non_dropout_gene_idx] - ctrl[0][non_dropout_gene_idx]) - np.sign(test_res['truth'][pert_idx].mean(0)[non_dropout_gene_idx] - ctrl[0][non_dropout_gene_idx]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(non_dropout_gene_idx)
        pert_metric[pert]['frac_correct_direction_non_dropout'] = frac_correct_direction
        
        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(non_dropout_gene_idx)
        pert_metric[pert]['frac_opposite_direction_non_dropout'] = frac_direction_opposite
        
        frac_direction_opposite = len(np.where(direc_change == 1)[0])/len(non_dropout_gene_idx)
        pert_metric[pert]['frac_0/1_direction_non_dropout'] = frac_direction_opposite
        
        mean = np.mean(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        std = np.std(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        min_ = np.min(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        max_ = np.max(test_res['truth'][pert_idx][:, de_idx], axis = 0)
        q25 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.25, axis = 0)
        q75 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.75, axis = 0)
        q55 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.55, axis = 0)
        q45 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.45, axis = 0)
        q40 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.4, axis = 0)
        q60 = np.quantile(test_res['truth'][pert_idx][:, de_idx], 0.6, axis = 0)
        
        zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
        nonzero_des = np.setdiff1d(list(range(20)), zero_des)
        
        if len(nonzero_des) == 0:
            pass
            # pert that all de genes are 0...
        else:            
            pred_mean = np.mean(test_res['pred'][pert_idx][:, de_idx], axis = 0).reshape(-1,)
            true_mean = np.mean(test_res['truth'][pert_idx][:, de_idx], axis = 0).reshape(-1,)
           
            in_range = (pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des])
            frac_in_range = sum(in_range)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_non_dropout'] = frac_in_range

            in_range_5 = (pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des])
            frac_in_range_45_55 = sum(in_range_5)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_45_55_non_dropout'] = frac_in_range_45_55

            in_range_10 = (pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des])
            frac_in_range_40_60 = sum(in_range_10)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_40_60_non_dropout'] = frac_in_range_40_60

            in_range_25 = (pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des])
            frac_in_range_25_75 = sum(in_range_25)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_25_75_non_dropout'] = frac_in_range_25_75

            zero_idx = np.where(std > 0)[0]
            sigma = (np.abs(pred_mean[zero_idx] - mean[zero_idx]))/(std[zero_idx])
            pert_metric[pert]['mean_sigma_non_dropout'] = np.mean(sigma)
            pert_metric[pert]['std_sigma_non_dropout'] = np.std(sigma)
            pert_metric[pert]['frac_sigma_below_1_non_dropout'] = 1 - len(np.where(sigma > 1)[0])/len(zero_idx)
            pert_metric[pert]['frac_sigma_below_2_non_dropout'] = 1 - len(np.where(sigma > 2)[0])/len(zero_idx)
        
        p_idx = np.where(test_res['pert_cat'] == pert)[0]
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top20_de_non_dropout'] = val


                val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top20_de_non_dropout'] = val
            else:
                val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])
                pert_metric[pert][m + '_top20_de_non_dropout'] = val
                
    return pert_metric



def get_metric(adata, 
               test_res, 
               most_variable_genes = None,
               p_thre_1 = 0.01,
               p_thre_2 = 0.1):

    pert_metric = {}

    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))

    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.shape[1])
    
    if most_variable_genes is None:
        most_variable_genes = np.argsort(np.std(mean_expression, axis = 0))[-200:]
        
    gene_list = adata.var['gene_name'].values

    # - get the result
    print('get metrics... ...')
    for pert in tqdm(np.unique(test_res['pert_cat'])):
        p_idx = np.where(test_res['pert_cat'] == pert)[0]   
        pert_metric[pert] = {}
        
        # - set basic info
        adata_pert = adata[adata.obs['perturbation_group']==pert2pert_full_id[pert]]
        adata_ctrl = adata[list(adata_pert.obs['control_barcode'])]
        
        # -- get ctrl expr
        ctrl_X = adata_ctrl.X
        if not isinstance(ctrl_X, np.ndarray):
            ctrl_X = ctrl_X.toarray()
        ctrl = np.mean(ctrl_X, axis = 0).reshape(-1,)
        
        names = adata.uns['rank_genes_groups'][pert2pert_full_id[pert]]
        pvals_adj = adata.uns['pvals_adj'][pert2pert_full_id[pert]]
        scores = adata.uns['scores'][pert2pert_full_id[pert]]
        
        expr_df = pd.DataFrame({'gene':adata_ctrl.var_names,
                                'expr':np.array(np.mean(adata_ctrl.X, axis=0)).ravel()})
        pval_df = pd.DataFrame({'gene':names,
                                'pvals_adj':np.array(pvals_adj),
                                'scores':np.array(scores)})
        expr_df.index = expr_df['gene']
        pval_df.index = pval_df['gene']
        
        # finally we choose de genes with rules in this df
        gene_df = pd.merge(pval_df,expr_df,how='left',left_index=True,right_index=True)
        gene_df = gene_df.sort_values(by='pvals_adj', ascending=True)

        gene_df_sig = gene_df[gene_df['pvals_adj']<p_thre_1]
        gene_df_nonsig = gene_df[gene_df['pvals_adj']>=p_thre_2]

        gene_df_nonsig = gene_df_nonsig.sort_values(by='expr', ascending=False)

        # - 1. de part
        if len(gene_df_sig) < 20:
            gene_list = list(gene_df.index)[0:20]
        else:
            gene_list = list(gene_df_sig.index)
        idx_list = [geneid2idx[i] for i in gene_list]
        idx_top20 = [geneid2idx[i] for i in list(gene_df.index)[0:20]]
        idx_top50 = [geneid2idx[i] for i in list(gene_df.index)[0:50]]
        idx_top100 = [geneid2idx[i] for i in list(gene_df.index)[0:100]]
        idx_top200 = [geneid2idx[i] for i in list(gene_df.index)[0:200]]
        de_idx_map = {'top20': idx_top20,
                      'top50': idx_top50,
                      'top100': idx_top100,
                      'top200': idx_top200,
                      'DE_sig': idx_list,
                     }
        metric2fct = {
            'pearson': pearsonr,
            'mse': mse,
            'mae': mae,
            'change_ratio': get_change_ratio,
            'spearman': spearmanr,  
        }
        name = 'DE_'
        for prefix in list(de_idx_map.keys()):
            de_idx = de_idx_map[prefix]
            for m, fct in metric2fct.items():
                if m == 'pearson' or m == 'spearman':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_delta_{prefix}'] = val


                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                elif m == 'change_ratio':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                else:
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val
         
        # - 2. de & equal direction for acc
        if len(gene_df_sig) < 20:
            gene_list = list(gene_df.index)[0:20]
        else:
            gene_list = list(gene_df_sig.index)

        gene_df_pos = gene_df[gene_df['scores'] >= 0]
        gene_df_neg = gene_df[gene_df['scores'] < 0]
            
        idx_list = [geneid2idx[i] for i in gene_list]
        idx_top20 = [geneid2idx[i] for i in list(gene_df_pos.index)[0:10]+list(gene_df_neg.index)[0:10]]
        idx_top50 = [geneid2idx[i] for i in list(gene_df_pos.index)[0:25]+list(gene_df_neg.index)[0:25]]
        idx_top100 = [geneid2idx[i] for i in list(gene_df_pos.index)[0:50]+list(gene_df_neg.index)[0:50]]
        idx_top200 = [geneid2idx[i] for i in list(gene_df_pos.index)[0:100]+list(gene_df_neg.index)[0:100]]
        de_idx_map = {'top20': idx_top20,
                        'top50': idx_top50,
                        'top100': idx_top100,
                        'top200': idx_top200,
                        'DE_sig': idx_list,
                        }
        name = 'DE_'
        for prefix in list(de_idx_map.keys()):
            de_idx = de_idx_map[prefix]
            
            m = 'direction_accuracy'
                
            direc_change = np.abs(np.sign(test_res['pred'][p_idx].mean(0)[de_idx_map[prefix]] - ctrl[de_idx_map[prefix]]) - np.sign(test_res['truth'][p_idx].mean(0)[de_idx_map[prefix]] - ctrl[de_idx_map[prefix]]))            
            frac_correct_direction = len(np.where(direc_change == 0)[0])/len(de_idx_map[prefix])
            pert_metric[pert][name + m + f'_{prefix}'] = frac_correct_direction

        
        # - 3. non-de & high express
        nonsig_mean = np.mean(list(gene_df_nonsig['expr']))
        
        if len(gene_df_nonsig) < 200: # if most genes are de genes
            gene_df_exp = gene_df_sig.sort_values(by='expr', ascending=False)
            idx_top20 = [geneid2idx[i] for i in list(gene_df_exp.index)[0:20]]
            idx_top50 = [geneid2idx[i] for i in list(gene_df_exp.index)[0:50]]
            idx_top100 = [geneid2idx[i] for i in list(gene_df_exp.index)[0:100]]
            idx_top200 = [geneid2idx[i] for i in list(gene_df_exp.index)[0:200]]
        else:
            idx_top20 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[0:20]]
            idx_top50 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[0:50]]
            idx_top100 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[0:100]]
            idx_top200 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[0:200]]
        
        if len(gene_df_nonsig) == 0:
            gene_df_exp = gene_df_sig.sort_values(by='expr', ascending=False)
            idx_nonsig_high = [geneid2idx[i] for i in list(gene_df_exp.index)[0:20]]
        else:
            idx_nonsig_high = [geneid2idx[i] for i in list(gene_df_nonsig[gene_df_nonsig['expr'] >= nonsig_mean].index)]
            
        de_idx_map = {'top20': idx_top20,
                      'top50': idx_top50,
                      'top100': idx_top100,
                      'top200': idx_top200,
                      'nonDE_high': idx_nonsig_high
                     }
        metric2fct = {
            'pearson': pearsonr,
            'mse': mse,
            'mae': mae,
            'change_ratio': get_change_ratio,
            'spearman': spearmanr,  
        }
        name = 'NonDE-high_'
        for prefix in list(de_idx_map.keys()):
            de_idx = de_idx_map[prefix]
            for m, fct in metric2fct.items():
                if m == 'pearson' or m == 'spearman':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_delta_{prefix}'] = val


                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                elif m == 'change_ratio':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                else:
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val

        
        # - 4. non-de & low express
        if len(gene_df_nonsig) < 200: # if most genes are de genes
            gene_df_exp = gene_df_sig.sort_values(by='expr', ascending=False)
            idx_top20 = [geneid2idx[i] for i in list(gene_df_exp.index)[::-1][0:20]]
            idx_top50 = [geneid2idx[i] for i in list(gene_df_exp.index)[::-1][0:50]]
            idx_top100 = [geneid2idx[i] for i in list(gene_df_exp.index)[::-1][0:100]]
            idx_top200 = [geneid2idx[i] for i in list(gene_df_exp.index)[::-1][0:200]]
        else:
            idx_top20 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[::-1][0:20]]
            idx_top50 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[::-1][0:50]]
            idx_top100 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[::-1][0:100]]
            idx_top200 = [geneid2idx[i] for i in list(gene_df_nonsig.index)[::-1][0:200]]
            
        if len(gene_df_nonsig) == 0:
            gene_df_exp = gene_df_sig.sort_values(by='expr', ascending=False)
            idx_nonsig_low = [geneid2idx[i] for i in list(gene_df_exp.index)[::-1][0:20]]
        else:
            idx_nonsig_low = [geneid2idx[i] for i in list(gene_df_nonsig[gene_df_nonsig['expr'] < nonsig_mean].index)]

        de_idx_map = {'top20': idx_top20,
                      'top50': idx_top50,
                      'top100': idx_top100,
                      'top200': idx_top200,
                      'nonDE_low': idx_nonsig_low
                     }
        metric2fct = {
            'pearson': pearsonr,
            'mse': mse,
            'mae': mae,
            'change_ratio': get_change_ratio,
            'spearman': spearmanr,  
        }
        name = 'NonDE-low_'
        for prefix in list(de_idx_map.keys()):
            de_idx = de_idx_map[prefix]
            for m, fct in metric2fct.items():
                if m == 'pearson' or m == 'spearman':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_delta_{prefix}'] = val


                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])[0]
                    if np.isnan(val):
                        val = 0
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                elif m == 'change_ratio':
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx], test_res['truth'][p_idx].mean(0)[de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val
                else:
                    val = fct(test_res['pred'][p_idx].mean(0)[de_idx] - ctrl[de_idx], test_res['truth'][p_idx].mean(0)[de_idx]-ctrl[de_idx])
                    pert_metric[pert][name + m + f'_{prefix}'] = val
        
    return pert_metric

def get_common_pert(pert_data, common_genes):
    # - get the common_genes **************
    pert_data.var_genes = list(common_genes)
    pert_data.adata_split = pert_data.adata_split[:, pert_data.var_genes].copy()
    pert_idx_dict = {}
    for pert, tmp_list in pert_data.adata_split.uns['rank_genes_groups'].items():
        idx_list = []
        for i, gene in enumerate(tmp_list):
            if gene in pert_data.adata_split.var_names:
                idx_list.append(i)
        pert_idx_dict[pert] = idx_list
    for key in pert_data.adata_split.uns.keys():
        print(key)
        ele = pert_data.adata_split.uns[key]
        for pert in ele.keys():
            ele[pert] = list(np.array(ele[pert])[pert_idx_dict[pert]])

