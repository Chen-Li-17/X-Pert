## 构建一个完整的类 Byte_Pert_Data

from typing import Any
import scanpy as sc
import pandas as pd
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import pickle
import sys
import requests
from .utils import *

from types import MethodType

from scperturb import *
import anndata as ad

class Byte_Pert_Data:
    def __init__(self,
                 data_dir = '/nfs/public/lichen/data/single_cell/perturb_data/scPerturb/raw/scPerturb_rna/statistic_20240520',
                 prefix = 'ReplogleWeissman2022_K562_essential',
                 pert_cell_filter = 200,
                 seed = 2024,
                 split_ratio = [0.7, 0.2, 0.1],
                 split_type = 1,
                 var_num = 500,
                 num_de_genes = 20,
                 bs_train = 32,
                 bs_test = 32):
        self.data_dir = data_dir
        self.prefix = prefix
        self.pert_cell_filter = pert_cell_filter
        self.split_ratio = split_ratio
        self.seed = seed
        self.split_type = split_type
        self.var_num = var_num
        self.num_de_genes = num_de_genes
        self.bs_train = bs_train
        self.bs_test = bs_test
        
        # None
        
    def read_files(self, adata_ori):
        
        # adata_ori.X = adata_ori.X.astype(np.float32)
        
        if isinstance(adata_ori.X, np.ndarray):
            from scipy.sparse import csr_matrix
            adata_ori.X = csr_matrix(adata_ori.X)
        
        # - filter cells, and get common_obs
        sc.pp.filter_cells(adata_ori, min_genes=200)
        
        obs_df = adata_ori.obs
        
        # - cal the total_perturbation_list
        tmp_obs = obs_df[obs_df['perturbation_new']!='control']
        total_perturbation_list = list(tmp_obs['perturbation_group'].unique()) # record the perturbation pair

        # - subset the metadata and count
        value_counts = obs_df.groupby(['perturbation_group']).size().reset_index(name='Counts')
        value_counts['prefix'] = self.prefix
        
        self.value_counts = value_counts
        self.obs_df = obs_df
        self.adata_ori = adata_ori
        self.total_perturbation_list = total_perturbation_list
        
        self.adata_ori.obs = obs_df
        
        print('='*10,f'read file finished!')
        
    def filter_perturbation(self):
        """
        This function is used to filter the perturbations with less cells
        """
        # - get the retain_pert_num
        retain_pert_num = self.value_counts.shape[0] - self.value_counts[self.value_counts['Counts']<self.pert_cell_filter].shape[0]
        print(f'retain_pert_num is: {retain_pert_num}')
        print('filtered pert num is: ',self.value_counts[self.value_counts['Counts']<self.pert_cell_filter].shape[0])
        
        # - initial a data_split column
        self.obs_df['data_split'] = 'train'
        self.obs_df['retain'] = 'True' # means retains these cells after filtering
        
        # - filter perturbation
        for pert in tqdm(self.total_perturbation_list):

            # - get total obs_names of the pert
            obs_df_sub_idx = np.array(self.obs_df[self.obs_df['perturbation_group']==pert].index)
            
            # - get the control number
            control_count = len(np.array(self.obs_df[self.obs_df['perturbation_group']==' | '.join(['control',pert.split(' | ')[1]])].index))
            if control_count < self.pert_cell_filter:
                self.obs_df.loc[obs_df_sub_idx,'retain'] = 'False'
                continue
                
            # - filter count<200
            cell_count = len(obs_df_sub_idx)
            if cell_count < self.pert_cell_filter:
                self.obs_df.loc[obs_df_sub_idx,'retain'] = 'False'
                
        self.obs_df_split = self.obs_df[self.obs_df['retain']=='True'].copy()
        
        print('='*10,f'filter perturbation finished!')

    def data_split(self,
                split_type = 0,
                test_perts = None):
        """
        This function is used to filter the perturbations with less cells, and split the data
        """
        # - initial a data_split column
        obs_df_split = self.adata_split.obs
        obs_df_split['data_split'] = 'train'
            
        if split_type == 0:
            
            # - split all the pert to 8:2
            for pert in tqdm(self.filter_perturbation_list):

                # - get total obs_names of the pert
                obs_df_sub_idx = np.array(obs_df_split[obs_df_split['perturbation_group']==pert].index)

                np.random.seed(self.seed)
                np.random.shuffle(obs_df_sub_idx)

                # - data split
                split_point_1 = int(len(obs_df_sub_idx) * self.split_ratio[0])
                split_point_2 = int(len(obs_df_sub_idx) * (self.split_ratio[0]+self.split_ratio[1]))
                train = obs_df_sub_idx[:split_point_1]
                test = obs_df_sub_idx[split_point_1:split_point_2]
                val = obs_df_sub_idx[split_point_2:]

                # - set the test row
                obs_df_split.loc[test,'data_split'] = 'test'
                obs_df_split.loc[val,'data_split'] = 'val'
                
                
        if split_type == 1:
            perts = np.array([i.split(' | ')[0] for i in self.filter_perturbation_list])
            np.random.seed(self.seed)
            np.random.shuffle(perts)

            # - data split
            split_point_1 = int(len(perts) * self.split_ratio[0])
            split_point_2 = int(len(perts) * (self.split_ratio[0]+self.split_ratio[1]))
            train_perts = perts[:split_point_1]
            test_perts = perts[split_point_1:split_point_2]
            val_perts = perts[split_point_2:]
            
            train = np.array(obs_df_split[obs_df_split['perturbation_new'].isin(train_perts)].index)
            test = np.array(obs_df_split[obs_df_split['perturbation_new'].isin(test_perts)].index)
            val = np.array(obs_df_split[obs_df_split['perturbation_new'].isin(val_perts)].index)
            
            obs_df_split.loc[test,'data_split'] = 'test'
            obs_df_split.loc[val,'data_split'] = 'val'
            
        if split_type == -1:
            test = np.array(obs_df_split[obs_df_split['perturbation_new'].isin(test_perts)].index)
            obs_df_split.loc[test,'data_split'] = 'test'

        self.adata_split.obs = obs_df_split.copy()
        self.obs_df_split = obs_df_split
        self.train_perts = train_perts
        self.test_perts = test_perts
        self.val_perts = val_perts
        print('='*10,f'data split finished!')
        
    def set_control_barcode(self):
        """
        this function is used to set control_barcode for each pert
        """
        
        # - set all control_barcode to None
        self.obs_df_split['control_barcode'] = 'None'
        
        # # - get all the control barcodes
        # control_obs = np.array(self.obs_df_split[(self.obs_df_split['perturbation_new']=='control')].index)
        
        for pert in tqdm(self.filter_perturbation_list):
            # - get the pert control
            # - get all the control barcodes
            control_obs = np.array(self.obs_df_split[(self.obs_df_split['perturbation_group']=='control'+' | '+pert.split(' | ')[1])].index)
            
            obs_df_sub_idx = np.array(self.obs_df_split[self.obs_df_split['perturbation_group']==pert].index)
            # - get the paired control
            np.random.seed(self.seed)
            pair_control_obs = np.random.choice(control_obs, len(obs_df_sub_idx), replace=True)
            # - set the control barcode
            self.obs_df_split.loc[obs_df_sub_idx,'control_barcode'] = pair_control_obs
            
        self.adata_split.obs = self.obs_df_split
        print('='*10,f'set control barcodes finished!')
            
    def get_hvg_list(self, sample_cell_num=10000, top_genes=5000):
        # - sample adata and get the top hvgs

        # -- sample data
        np.random.seed(self.seed)
        select_cell_num = min(self.adata_split.shape[0], sample_cell_num)
        adata = self.adata_split[np.random.choice(self.adata_split.obs_names, select_cell_num, replace=False)].copy()

        # -- adata process
        sc.pp.filter_cells(adata, min_genes=200)
        # sc.pp.filter_genes(adata, min_cells=3)
        adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        # -- Normalize gene expression matrix with total UMI count per cell
        sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all')
        sc.pp.log1p(adata)
        # -- sc.pp.highly_variable_genes(adata, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
        sc.pp.highly_variable_genes(adata, n_top_genes=top_genes)

        adata = adata[:, adata.var.highly_variable]

        # - get the subset var
        var_genes = np.union1d(adata.var_names, np.setdiff1d(self.pert_gene_list,self.exclude_var_list))
        print(f'len of pert_gene_list is {len(self.pert_gene_list)}')
        print(f'len of final var_names is {len(var_genes)}')
        print('='*10,f'get var genes finished!')
        
        return var_genes
            
    def get_and_process_adata(self, var_num, var_genes=None):
        """
        this function is used to subset the adata from the adata_ori, and process and select hvg for the adata
        """
        # - get the adata after adata_split
        adata_split = self.adata_ori[list(self.obs_df_split.index)].copy()
        adata_split.obs = self.obs_df_split
        self.adata_split = adata_split

        # - cal all the pert_gene_list
        perts, celltypes = get_pert_celltype(self.obs_df_split['perturbation_group'])
        
        pert_gene_list = np.setdiff1d(perts,['control'])
        # - process pert_gene_list for multi perts
        pert_gene_list = []
        for pert_gene in np.setdiff1d(perts,['control']):
            pert_gene_list.extend(pert_gene.split('; '))
        
        self.pert_gene_list = pert_gene_list

        # - cal the gene list not in the adata.var
        exclude_var_list = []
        for gene in pert_gene_list:
            if gene not in adata_split.var_names:
                exclude_var_list.append(gene)
                # print(f'{gene} not in adata_split.var_names')
        print(f'len of exclude_var_list is {len(exclude_var_list)}')
        self.exclude_var_list = exclude_var_list
        
        # - get the var_genes
        self.var_genes = self.get_hvg_list(10000, var_num)
        
        if var_genes!=None:
            self.var_genes = list(var_genes)
        
        # - subset adata_split and preprocess

        # -- process adata_split
        sc.pp.filter_cells(self.adata_split, min_genes=200)
        # sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_per_cell(self.adata_split, key_n_counts='n_counts_all')
        sc.pp.log1p(self.adata_split)
        
        # -- subset
        # adata_split.raw = adata_split
        self.adata_split = self.adata_split[:, self.var_genes]
        
        print('this is new version')
        
        del self.adata_split.uns
        
        # - cal the filter_perturbation_list
        tmp_obs = self.adata_split[self.adata_split.obs['perturbation_new']!='control'].obs
        self.filter_perturbation_list = list(tmp_obs['perturbation_group'].unique()) # record the perturbation pair
        self.perts, self.celltypes = get_pert_celltype(self.filter_perturbation_list)
            
    # def get_filter_perturbation(self):
    #     # - cal the filter_perturbation_list
    #     tmp_obs = self.adata_split[self.adata_split.obs['perturbation_new']!='control'].obs
    #     self.filter_perturbation_list = list(tmp_obs['perturbation_group'].unique()) # record the perturbation pair
        
    def get_de_genes(self,
                    rankby_abs = True,
                    key_added = 'rank_genes_groups'):
        gene_dict = {}
        for pert in tqdm(self.filter_perturbation_list):
            adata_pert = self.adata_split[self.adata_split.obs['perturbation_group']==pert]
            adata_de = self.adata_split[list(adata_pert.obs_names)+list(adata_pert.obs['control_barcode'])].copy()
            control_group_cov = ' | '.join(['control', pert.split(' | ')[1]])
            #compute DEGs
            sc.tl.rank_genes_groups(
                adata_de,
                groupby='perturbation_group',
                reference=control_group_cov,
                rankby_abs=rankby_abs,
                n_genes=len(adata_de.var),
                use_raw=False,
                method = 'wilcoxon'
            )
            de_genes = pd.DataFrame(adata_de.uns['rank_genes_groups']['names'])
            for group in de_genes:
                gene_dict[group] = de_genes[group].tolist()
        self.adata_split.uns[key_added] = gene_dict
        print('='*10,f'get de genes finished!')
        
    def get_edis(self,
                n_var_max = 2000,
                subsample_num = 200):
        # - preprocess data, no need to normalize
        adata_tmp = self.adata_split.copy()
        if 'processed' in adata_tmp.uns.keys():
            print('The dataset is already processed. Skipping processing...')
        else:
            adata_tmp.layers['counts'] = adata_tmp.X.copy()
            if subsample_num != None:
                adata_tmp = equal_subsampling(adata_tmp, 'perturbation_group', N_min=subsample_num)
            # select HVGs
            sc.pp.highly_variable_genes(adata_tmp, n_top_genes=n_var_max, subset=False, flavor='seurat_v3', layer='counts')
            sc.pp.pca(adata_tmp, use_highly_variable=True)
            sc.pp.neighbors(adata_tmp)

            adata_tmp.uns['processed'] = True
            
        # - Compute E-distances
        estats = edist(adata_tmp, obs_key='perturbation_group', obsm_key='X_pca', dist='sqeuclidean')

        # - get filter_pert_edis
        filter_pert_edis = {}
        for pert in self.filter_perturbation_list:
            ctrl = ' | '.join(['control', pert.split(' | ')[1]])
            if pert not in estats or ctrl not in estats:
                continue
            filter_pert_edis[pert] = estats.loc[pert, ctrl]

        # - get the sort pert edis pd    
        df_pert_edis = pd.DataFrame({'pert':filter_pert_edis.keys(),
                    'edis':filter_pert_edis.values()})
        df_pert_edis = df_pert_edis.sort_values(by='edis', ascending=False)
        
        self.filter_pert_edis = filter_pert_edis
        self.df_pert_edis = df_pert_edis
        
        print('='*10,f'get edis finished!')
        
    def get_edis_2(self,
                n_var_max = 2000,
                subsample_num = 200):
        '''
        this is for each pair of pert to ctrl
        '''
        filter_pert_edis = {}
        for pert in tqdm(self.filter_perturbation_list):
            adata_pert = self.adata_split[self.adata_split.obs['perturbation_group']==pert].copy()
            adata_ctrl = self.adata_split[adata_pert.obs['control_barcode']].copy()
            
            adata_cat = ad.concat([adata_pert, adata_ctrl], axis=0)
            
            adata_cat.layers['counts'] = adata_cat.X.copy()
            # select HVGs
            sc.pp.highly_variable_genes(adata_cat, n_top_genes=n_var_max, subset=False, flavor='seurat_v3', layer='counts')
            sc.pp.pca(adata_cat, use_highly_variable=True)
            sc.pp.neighbors(adata_cat)

            adata_cat.uns['processed'] = True
            
            #关闭print的输出
            # sys.stdout = open(os.devnull, 'w')
            # - Compute E-distances
            estats = edist(adata_cat, obs_key='perturbation_group', obsm_key='X_pca', dist='sqeuclidean', verbose=False)
            #打开print的输出
            # sys.stdout = sys.__stdout__
            
            
            ctrl = ' | '.join(['control', pert.split(' | ')[1]])
            # if pert not in estats or ctrl not in estats:
            #     continue
            filter_pert_edis[pert] = estats.loc[pert, ctrl]
            
            # - get the sort pert edis pd    
            df_pert_edis = pd.DataFrame({'pert':filter_pert_edis.keys(),
                        'edis':filter_pert_edis.values()})
            df_pert_edis = df_pert_edis.sort_values(by='edis', ascending=False)

            self.filter_pert_edis = filter_pert_edis
            self.df_pert_edis = df_pert_edis
            
    def filter_sgRNA(self):
        
        # - for each pert, get the sgRNA num
        pert_sgRNA_num = {}
        for pert in self.filter_perturbation_list:
            adata_sub = self.adata_split[self.adata_split.obs['perturbation_group']==pert]
            pert_sgRNA_num[pert] = len(adata_sub.obs['sgRNA_new'].unique())
            
        df_pert_sgRNA_num = pd.DataFrame({'pert':list(pert_sgRNA_num.keys()),
                                        'sgRNA_num':list(pert_sgRNA_num.values())})
        df_pert_sgRNA_num = df_pert_sgRNA_num.sort_values(by='sgRNA_num', ascending=False)

        # - filter the pert_list that more than 1 sgRNA
        multi_sgRNA_pert_list = list(df_pert_sgRNA_num[df_pert_sgRNA_num['sgRNA_num']>1]['pert'])
        len(multi_sgRNA_pert_list)

        # - add 'sgRNA_ID' column
        split_obs_df = self.adata_split.obs

        split_obs_df['sgRNA_ID'] = 'control'
        for pert in multi_sgRNA_pert_list:
            adata_sub = self.adata_split[split_obs_df['perturbation_group']==pert]
            sgRNA_list = adata_sub.obs['sgRNA_new'].unique()
            
            for i,sgRNA in enumerate(sgRNA_list):
                tmp_idx = np.array(split_obs_df[split_obs_df['sgRNA_new']==sgRNA].index)
                split_obs_df.loc[tmp_idx,'sgRNA_ID'] = 'sgRNA_' + str(i+1)
        self.adata_split.obs = split_obs_df

        # - get 'pert_sgRNA' column, the format is 'pert | celltype | sgRNA_ID'
        split_obs_df = self.adata_split.obs
        self.adata_split.obs['pert_sgRNA'] = [' | '.join([split_obs_df['perturbation_new'][i],
                                                            split_obs_df['celltype_new'][i],
                                                            split_obs_df['sgRNA_ID'][i]]) for i in range(split_obs_df.shape[0])]

        # - cal the edis and efficacy for each sgRNA in each pert
        filter_pert_edis = {}
        df_sgRNA_edis_dict = {}
        n_var_max = 2000
        for pert in tqdm(multi_sgRNA_pert_list):
            adata_pert = self.adata_split[self.adata_split.obs['perturbation_group']==pert].copy()
            adata_ctrl = self.adata_split[adata_pert.obs['control_barcode']].copy()
            
            adata_cat = ad.concat([adata_pert, adata_ctrl], axis=0)
            
            adata_cat.layers['counts'] = adata_cat.X.copy()
            # select HVGs
            sc.pp.highly_variable_genes(adata_cat, n_top_genes=n_var_max, subset=False, flavor='seurat_v3', layer='counts')
            sc.pp.pca(adata_cat, use_highly_variable=True)
            sc.pp.neighbors(adata_cat)

            adata_cat.uns['processed'] = True

            # - Compute E-distances
            estats = edist(adata_cat, obs_key='pert_sgRNA', obsm_key='X_pca', dist='sqeuclidean')
            
            sgRNA_list = np.sort(adata_pert.obs['pert_sgRNA'].unique())
            sgRNA_edis = {}

            
            
            ctrl = ' | '.join(['control', pert.split(' | ')[1], 'control'])
            # if pert not in estats or ctrl not in estats:
            #     continue
            for sgRNA in sgRNA_list:
                sgRNA_edis[sgRNA] = estats.loc[sgRNA, ctrl]
                
            
            
            gene = pert.split(' | ')[0]
            adata_ctrl = self.adata_split[self.adata_split.obs['perturbation_group']=='control'+' | '+pert.split(' | ')[1]]
            if gene in adata_ctrl.var_names:
                # print(f'{gene} is in var')
                gene_ctrl_expr = np.mean(adata_ctrl[:,[gene]].X.toarray())

                gene_sgRNA_expr_list = []
                cell_num_list = []
                for sgRNA in sgRNA_list:
                    adata_sub = adata_pert[adata_pert.obs['pert_sgRNA']==sgRNA]
                    cell_num_list.append(len(adata_sub))
                    gene_sgRNA_expr_list.append(np.mean(adata_sub[:,[gene]].X.toarray()))
                    
                    
                # - get the sort sgRNA edis pd    
                df_sgRNA_edis = pd.DataFrame({'pert':sgRNA_edis.keys(),
                            'cell_num':cell_num_list,
                            'edis':sgRNA_edis.values(),
                            'ctrl_gene_expr':[gene_ctrl_expr]*len(gene_sgRNA_expr_list),
                            'pert_gene_expr':gene_sgRNA_expr_list,
                            'pert_expr_ratio':(np.array(gene_sgRNA_expr_list)-gene_ctrl_expr)/gene_ctrl_expr})
                
                
                df_sgRNA_edis_dict[pert] = df_sgRNA_edis
            else:
                print(f'{gene} not in var!')

        # - filter sgRNA
        filter_count = 0
        filter_pert_sgRNA = []
        for pert in list(df_sgRNA_edis_dict.keys()):
            df = df_sgRNA_edis_dict[pert]
            print('='*10,pert)
            df.index = df['pert']
            
            pert_sgRNA_list = df['pert'].unique()
            max_edis = np.max(df['edis'])
            max_edis_cellnum = df[df['edis']==max_edis]['cell_num'][0]
            edis_array = df['edis'].copy()
            edis_array[edis_array<0.1] = 0.1
            
            filter_sgRNA_list = []
            if max_edis > 10 and max_edis_cellnum > 50:
                filter_count += 1
                for pert_sgRNA in pert_sgRNA_list:
                    if edis_array.loc[pert_sgRNA] < 10 and max_edis/edis_array.loc[pert_sgRNA] > 10:
                        filter_sgRNA_list.append(pert_sgRNA)
                        filter_pert_sgRNA.append(pert_sgRNA)
                print(f'filter_sgRNA_list is: {filter_sgRNA_list}')
            else:
                print('max_eids < 10, no need to filter sgRNA')
                
        self.df_pert_sgRNA_num = df_pert_sgRNA_num # df, sgRNA_num
        self.multi_sgRNA_pert_list = multi_sgRNA_pert_list # perts with more than 1 sgRNA
        self.df_sgRNA_edis_dict = df_sgRNA_edis_dict # detailed info of each pert's sgRNA
        self.filter_pert_sgRNA = filter_pert_sgRNA # the pert_sgRNA that is supposed to be filtered
        
        # - filter the sgRNAs
        self.adata_split = self.adata_split[~self.adata_split.obs['pert_sgRNA'].isin(self.filter_pert_sgRNA)].copy()
        self.obs_df_split = self.adata_split.copy()
        
        
    def plot_umap(self,
                pert,
                point_size = 10,
                color = ['perturbation_group'],
                return_adata = False):
        # - get the pert adata
        adata_pert = self.adata_split[self.adata_split.obs['perturbation_group']==pert].copy()
        adata_ctrl = self.adata_split[adata_pert.obs['control_barcode']].copy()

        # - concat data
        adata_cat = ad.concat([adata_pert, adata_ctrl], axis=0)

        # - process data to plot umap
        min_mean       =0.0125
        max_mean       =3
        min_disp       =0.5
        n_neighbors    =10
        n_pcs          =30

        sc.pp.highly_variable_genes(adata_cat, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
        # sc.pp.highly_variable_genes(adata, n_top_genes=3000)

        adata_cat.raw = adata_cat
        adata_cat = adata_cat[:, adata_cat.var.highly_variable]
        sc.pp.scale(adata_cat)
        # pca
        sc.tl.pca(adata_cat, svd_solver='arpack',random_state=2022)
        # neighbors
        sc.pp.neighbors(adata_cat, n_neighbors=n_neighbors, n_pcs=n_pcs,random_state=2022)
        # check umap
        sc.tl.umap(adata_cat,random_state=2022)

        sc.pl.umap(adata_cat, color=color,size=point_size,legend_fontsize=5,
                # color_map='magma',
                ncols=1, wspace=0.4,legend_loc='right margin',
                cmap='coolwarm'
                )

        if return_adata:
            return adata_cat
        
    def get_Data_gears(self,
                    dataset_name = ['train', 'test'],
                    num_de_genes = 20,
                    test_mode = False,
                    add_control = True,
                    only_control = False # if only_control is true, only save ctrl in trainloader
                    ):
        """
        this is for constructing datasets of gears format
        """

        self.dataset_perturbation_list = {} # save perturbation_list for train/test
        self.dataset_pert_cell_graphs = {} # save the pert_cell_graphs
        for dataset in dataset_name:
            # - get the perturbation list
            tmp_obs = self.adata_split[self.adata_split.obs['data_split']==dataset].obs
            tmp_obs = tmp_obs[tmp_obs['perturbation_new']!='control']
            tmp_perturbation_list = list(tmp_obs['perturbation_group'].unique()) # record the perturbation pair
            tmp_adata_split = self.adata_split[tmp_obs.index]
            
            # - get total cell graphs for each dataset
            pert_cell_graphs = {}

            if only_control and dataset=='train':
                pert = 'control'
                adata_pert = self.adata_split[self.adata_split.obs['perturbation_new']==pert]
                Xs = adata_pert.X # ctrl value
                # ys = self.adata_split[adata_pert.obs['control_barcode']].X # perturb value
                if not isinstance(Xs, np.ndarray):
                    Xs = Xs.toarray()
                
                self.de = False
                # num_de_genes = 1
                    
                if self.de:
                    de_idx = np.where(adata_pert.var_names.isin(
                    np.array(de_genes[pert][:num_de_genes])))[0]
                else:
                    de_idx = [-1]*num_de_genes
                    
                cell_graphs = []
                # Create cell graphs
                for X, y in zip(Xs, Xs):
                    # feature_mat = torch.Tensor(X).T
                    # if pert_idx is None:
                    pert_idx = [-1]
                    gears_pert = 'ctrl'
                    tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                                y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert)
                    cell_graphs.append(tmp_Data)
                    
                pert_cell_graphs[pert] = cell_graphs

            else:

                for pert in tqdm(tmp_perturbation_list):
                    adata_pert = tmp_adata_split[tmp_adata_split.obs['perturbation_group']==pert]
                    Xs = adata_pert.X # ctrl value
                    ys = self.adata_split[adata_pert.obs['control_barcode']].X # perturb value
                    
                    # - get the de_idx for pert
                    if 'rank_genes_groups' in adata_pert.uns:
                        de_genes = adata_pert.uns['rank_genes_groups']
                        self.de = True
                    else:
                        self.de = False
                        num_de_genes = 1
                        
                    if self.de:
                        de_idx = np.where(adata_pert.var_names.isin(
                        np.array(de_genes[pert][:num_de_genes])))[0]
                    else:
                        de_idx = [-1] * num_de_genes
                        
                    # - get the pert_idx [TODO]
                    pert_idx = self.get_pert_idx(pert)
                    
                    if not isinstance(Xs, np.ndarray):
                        Xs = Xs.toarray()
                    if not isinstance(ys, np.ndarray):
                        ys = ys.toarray()

                    cell_graphs = []
                    # Create cell graphs
                    for X, y in zip(Xs, ys):
                        # feature_mat = torch.Tensor(X).T
                        if pert_idx is None:
                            pert_idx = [-1]
                            
                        # - if multiple perts
                        if '; ' in pert:
                            gears_pert = '+'.join(pert.split(' | ')[0].split('; ')) # change to gears pert
                        else:
                            gears_pert = '+'.join(pert.split(' | ')[0].split('; ')+['ctrl']) # change to gears pert
                            
                        tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                                    y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert)
                        cell_graphs.append(tmp_Data)
                        
                    pert_cell_graphs[pert] = cell_graphs
                    
                    if test_mode:
                        break
                    
                # - add control to training set
                if add_control and dataset == 'train':
                    pert = 'control'
                    adata_pert = self.adata_split[self.adata_split.obs['perturbation_new']==pert]
                    Xs = adata_pert.X # ctrl value
                    # ys = self.adata_split[adata_pert.obs['control_barcode']].X # perturb value
                    if not isinstance(Xs, np.ndarray):
                        Xs = Xs.toarray()
                    
                    self.de = False
                    # num_de_genes = 1
                        
                    if self.de:
                        de_idx = np.where(adata_pert.var_names.isin(
                        np.array(de_genes[pert][:num_de_genes])))[0]
                    else:
                        de_idx = [-1]*num_de_genes
                        
                    cell_graphs = []
                    # Create cell graphs
                    for X, y in zip(Xs, Xs):
                        # feature_mat = torch.Tensor(X).T
                        # if pert_idx is None:
                        pert_idx = [-1]
                        gears_pert = 'ctrl'
                        tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                                    y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert)
                        cell_graphs.append(tmp_Data)
                        
                    pert_cell_graphs[pert] = cell_graphs
                
            
            # - get the result        
            self.dataset_perturbation_list[dataset] = tmp_perturbation_list
            self.dataset_pert_cell_graphs[dataset] = pert_cell_graphs
            
            if test_mode:
                self.dataset_perturbation_list[dataset] = list(pert_cell_graphs.keys())
                self.dataset_pert_cell_graphs[dataset] = pert_cell_graphs
        print('='*10,f'get Data_gears finished!')
        
            
    def get_dataloader(self,
                    mode = 'all',
                    sub_dict = None, # used to filter dataloader by pert or else
                    bs_train = 32,
                    bs_test = 32,
                    shuffle_train = True,
                    shuffle_test = False):
        self.node_map = {x: it for it, x in enumerate(self.adata_split.var_names)}
        if mode == 'all':

            cell_graphs = {}
            for dataset in self.dataset_perturbation_list:
                cell_graphs[dataset] = []
                for pert in self.dataset_perturbation_list[dataset]:
                    cell_graphs[dataset].extend(self.dataset_pert_cell_graphs[dataset][pert])
            train_loader = DataLoader(cell_graphs['train'],
                                    batch_size=bs_train, shuffle=shuffle_train, drop_last = False)
            test_loader = DataLoader(cell_graphs['test'],
                                    batch_size=bs_test, shuffle=shuffle_test)
            val_loader = DataLoader(cell_graphs['val'],
                                    batch_size=bs_test, shuffle=shuffle_test)
            
            self.dataloader = {'train_loader': train_loader,
                            'test_loader': test_loader,
                            'val_loader': test_loader}
            
            
            return train_loader, test_loader, val_loader
        
    def get_pert_idx(self, pert):
        """
        Get perturbation index for a given perturbation category

        Parameters
        ----------
        pert_category: str
            Perturbation category

        Returns
        -------
        list
            List of perturbation indices

        """
        try:
            pert_idx = [np.where(p == self.pert_names)[0][0]
                    for p in pert.split(' | ')[0].split('; ') if p != 'control']
        except:
            # print(f'{pert} not in pert_names')
            pert_idx = None
            
        return pert_idx
    
    def get_gene2go(self):
        self.data_path = './gears_data/'
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path,
                            os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)

    def set_pert_genes(self,
                    gene_set_path = None,
                    default_pert_graph = True):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """
        self.gene_set_path = gene_set_path
        self.default_pert_graph = default_pert_graph
        if self.gene_set_path is not None:
            # If gene set specified for perturbation graph, use that
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        elif self.default_pert_graph is False:
            # Use a smaller perturbation graph 
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            
        else:
            # Otherwise, use a large set of genes to create perturbation graph
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path,
                                        'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)

        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
        
    def modify_gears(self,
                    split = 'simulation',
                    train_gene_set_size = 0.75,
                    without_subgroup = False,
                    ):
        """
        add some necessary part for gears
        """
        # - add adata
        self.adata = self.adata_split
        # - change adata.X to csr_matrix
        from scipy.sparse import csr_matrix
        if isinstance(self.adata.X, np.ndarray):
            self.adata.X = csr_matrix(self.adata.X)
        print('add adata finished')
        
        # - add node_map
        self.node_map = {x: it for it, x in enumerate(self.adata.var_names)}
        
        # - add else
        self.dataset_name = self.prefix
        self.split = split
        self.train_gene_set_size = train_gene_set_size
        self.gene_names = self.adata_split.var_names
        
        # - add condition, which is perturbation in gears [TODO] this is only 
        def get_adata_condition(x):
            if ';' in x['perturbation_new']:
                return '+'.join(x['perturbation_new'].split('; '))
            
            if x['perturbation_new'] == 'control':
                return 'ctrl'
            else:
                return '+'.join([x['perturbation_new'],'ctrl'])
            
        adata_condition = self.adata.obs.apply(get_adata_condition, axis=1)
        self.adata.obs['condition'] = adata_condition
        
        # - condition_name : this variable should be changed if 
        self.adata.obs.loc[:, 'condition_name'] =  self.adata.obs.apply(lambda x: ' | '.join([x.perturbation_new, x.celltype_new]), axis = 1)
        print('add condition finished')
        
        # - get the subgroup to evaluate
        subgroup = {
            'test_subgroup': {
                # 'seen_single':[],
                # 'combo_seen0': [],
                # 'combo_seen1': [],
                # 'combo_seen2': [],
                # 'unseen_single': []
                },
            'val_subgroup': {
                # 'seen_single':[],
                # 'combo_seen0': [],
                # 'combo_seen1': [],
                # 'combo_seen2': [],
                # 'unseen_single': []
            }
        }
        def convert_gears_pert(my_pert):
            if ';' in my_pert:
                return '+'.join(my_pert.split(' | ')[0].split('; '))
            else:
                return '+'.join(my_pert.split(' | ')[0].split('; ')+['ctrl'])
            
        if not without_subgroup:
        
            tmp_list = [convert_gears_pert(i) for i in self.dataset_perturbation_list['test']]
            subgroup['test_subgroup']['unseen_single'] = list(tmp_list)
            
            tmp_list = [convert_gears_pert(i) for i in self.dataset_perturbation_list['val']]
            subgroup['val_subgroup']['unseen_single'] = list(tmp_list)
            
            self.subgroup = subgroup
        
        # - add uns for adata
        get_dropout_non_zero_genes(self.adata)
        self.adata.var['gene_name'] = self.adata.var_names
        self.adata.uns['rank_genes_groups_cov_all'] = self.adata.uns['rank_genes_groups']
        
        # - get set2conditions
        set2conditions = dict(self.adata.obs.groupby('data_split').agg({'condition':
                                                                lambda x: x}).condition)
        set2conditions = {i: list(np.unique(j)) for i,j in set2conditions.items()} 
        # set2conditions['val'] = set2conditions['test']
        self.set2conditions = set2conditions
        print('add set2conditions finished')
        
    def get_Data_scgpt(self,
                    dataset_name = ['train', 'test', 'val'],
                    num_de_genes = 20,
                    test_mode = False,
                    add_control = True,
                    only_control = False,
                    split_col = 'data_split'):
        """
        this is for constructing datasets of gears format
        """

        self.dataset_perturbation_list = {} # save perturbation_list for train/test
        self.dataset_pert_cell_graphs = {} # save the pert_cell_graphs

        self.var_idx_dict = dict(zip(self.var_genes, np.arange(len(self.var_genes))))
        
        for dataset in dataset_name:
            # - get the perturbation list
            tmp_obs = self.adata_split[self.adata_split.obs[split_col]==dataset].obs
            # tmp_obs = tmp_obs[tmp_obs['perturbation_new']!='control']
            tmp_perturbation_list = list(tmp_obs['perturbation_group'].unique()) # record the perturbation pair
            tmp_adata_split = self.adata_split[tmp_obs.index]

            if dataset == 'train':
                control_perturbation = [i for i in tmp_perturbation_list if 'control' in i][0]
                tmp_perturbation_list = [i for i in tmp_perturbation_list if i!=control_perturbation]
            
            # - get total cell graphs for each dataset
            pert_cell_graphs = {}

            if only_control and dataset=='train':
                pert = 'control'
                adata_pert = self.adata_split[self.adata_split.obs['perturbation_new']==pert]
                Xs = adata_pert.X # ctrl value
                # ys = self.adata_split[adata_pert.obs['control_barcode']].X # perturb value
                pert_flags = torch.zeros(Xs.shape[1])
                if not isinstance(Xs, np.ndarray):
                    Xs = Xs.toarray()
                
                self.de = False
                # num_de_genes = 1
                    
                if self.de:
                    de_idx = np.where(adata_pert.var_names.isin(
                    np.array(de_genes[pert][:num_de_genes])))[0]
                else:
                    de_idx = [-1]*num_de_genes
                    
                cell_graphs = []
                # Create cell graphs
                for X, y in zip(Xs, Xs):
                    # feature_mat = torch.Tensor(X).T
                    # if pert_idx is None:
                    pert_idx = [-1]
                    gears_pert = 'ctrl'
                    tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                                y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert,
                                pert_flags=pert_flags.reshape(1,-1))
                    cell_graphs.append(tmp_Data)
                    
                pert_cell_graphs[pert] = cell_graphs
                tmp_perturbation_list = [pert]

            else:

                for pert in tqdm(tmp_perturbation_list):
                    adata_pert = tmp_adata_split[tmp_adata_split.obs['perturbation_group']==pert]
                    Xs = adata_pert.X # ctrl value
                    ys = self.adata_split[adata_pert.obs['control_barcode']].X # perturb value
                    Xs, ys = ys, Xs # the input must be ctrl!

                    # - get the de_idx for pert
                    if 'rank_genes_groups' in adata_pert.uns:
                        de_genes = adata_pert.uns['rank_genes_groups']
                        self.de = True
                    else:
                        self.de = False
                        # num_de_genes = 1
                        
                    if self.de:
                        de_idx = np.where(adata_pert.var_names.isin(
                        np.array(de_genes[pert][:num_de_genes])))[0]
                    else:
                        de_idx = [-1] * num_de_genes
                        
                    # - get the pert_idx [TODO]
                    pert_idx = self.get_pert_idx(pert)
                    
                    if not isinstance(Xs, np.ndarray):
                        Xs = Xs.toarray()
                    if not isinstance(ys, np.ndarray):
                        ys = ys.toarray()

                    # - pert_flags for multi perts
                    pert_flags = torch.zeros(Xs.shape[1])
                    for gene in pert.split(' | ')[0].split('; '):
                        if gene not in self.var_idx_dict:
                            continue
                        else:
                            pert_flags[self.var_idx_dict[gene]] = 1
                    cell_graphs = []
                    # Create cell graphs
                    for X, y in zip(Xs, ys):
                        # feature_mat = torch.Tensor(X).T
                        if pert_idx is None:
                            pert_idx = [-1]

                        # - if multiple perts
                        if '; ' in pert:
                            gears_pert = '+'.join(pert.split(' | ')[0].split('; ')) # change to gears pert
                        else:
                            gears_pert = '+'.join(pert.split(' | ')[0].split('; ')+['ctrl']) # change to gears pert

                        tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                                    y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert,
                                    pert_flags=pert_flags.reshape(1,-1))
                        cell_graphs.append(tmp_Data)
                        
                    pert_cell_graphs[pert] = cell_graphs
                    
                    if test_mode:
                        break

                # - add control to training set
                if add_control and dataset == 'train':
                    pert = control_perturbation
                    tmp_perturbation_list.append(control_perturbation)
                    adata_pert = self.adata_split[self.adata_split.obs['perturbation_group']==pert]
                    Xs = adata_pert.X # ctrl value
                    # ys = self.adata_split[adata_pert.obs['control_barcode']].X # perturb value
                    pert_flags = torch.zeros(Xs.shape[1])
                    if not isinstance(Xs, np.ndarray):
                        Xs = Xs.toarray()
                    
                    self.de = False
                    # num_de_genes = 1
                        
                    if self.de:
                        de_idx = np.where(adata_pert.var_names.isin(
                        np.array(de_genes[pert][:num_de_genes])))[0]
                    else:
                        de_idx = [-1]*num_de_genes
                        
                    cell_graphs = []
                    # Create cell graphs
                    for X, y in zip(Xs, Xs):
                        # feature_mat = torch.Tensor(X).T
                        # if pert_idx is None:
                        pert_idx = [-1]
                        gears_pert = 'ctrl'
                        tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                                    y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert,
                                    pert_flags=pert_flags.reshape(1,-1))
                        cell_graphs.append(tmp_Data)
                        
                    pert_cell_graphs[pert] = cell_graphs
            
            # - get the result        
            self.dataset_perturbation_list[dataset] = tmp_perturbation_list
            self.dataset_pert_cell_graphs[dataset] = pert_cell_graphs
                
            if test_mode:
                self.dataset_perturbation_list[dataset] = list(pert_cell_graphs.keys())
                self.dataset_pert_cell_graphs[dataset] = pert_cell_graphs
        print('='*10,f'get Data_scGPT finished!')

    def get_Data_scgpt_2(self,
                    dataset_name = ['train', 'test', 'val'],
                    num_de_genes = 20,
                    test_mode = False,
                    add_control = True,
                    only_control = False,
                    split_col = 'data_split', 
                    pert_flag_mode = False,
                    drug_embed_dir = None,
                    pert_type_mode = False):
        """
        this is for constructing datasets of gears format
        
        change the pert, the original is gears pert
        """

        self.dataset_perturbation_list = {} # save perturbation_list for train/test
        self.dataset_pert_cell_graphs = {} # save the pert_cell_graphs

        if pert_flag_mode:
            drug_gene_df = pd.read_csv(os.path.join(drug_embed_dir, f'drug_gene_df_MgraphDTA.csv'))

        self.var_idx_dict = dict(zip(self.var_genes, np.arange(len(self.var_genes))))
        
        for dataset in dataset_name:
            # - get the perturbation list
            tmp_obs = self.adata_split[self.adata_split.obs[split_col]==dataset].obs
            # tmp_obs = tmp_obs[tmp_obs['perturbation_new']!='control']
            tmp_perturbation_list = list(tmp_obs['perturbation_group'].unique()) # record the perturbation pair
            tmp_adata_split = self.adata_split[tmp_obs.index]

            # if dataset == 'train':
            control_perturbation = [i for i in tmp_perturbation_list if 'control' in i]
            tmp_perturbation_list = [i for i in tmp_perturbation_list if i not in control_perturbation]
            
            # - get total cell graphs for each dataset
            pert_cell_graphs = {}

            if only_control and dataset=='train':
                pert = 'control'
                adata_pert = self.adata_split[self.adata_split.obs['perturbation_new']==pert]
                Xs = adata_pert.X # ctrl value
                # ys = self.adata_split[adata_pert.obs['control_barcode']].X # perturb value
                pert_flags = torch.zeros(Xs.shape[1])
                if not isinstance(Xs, np.ndarray):
                    Xs = Xs.toarray()
                
                self.de = False
                # num_de_genes = 1
                    
                if self.de:
                    de_idx = np.where(adata_pert.var_names.isin(
                    np.array(de_genes[pert][:num_de_genes])))[0]
                else:
                    de_idx = [-1]*num_de_genes
                    
                cell_graphs = []
                # Create cell graphs
                for X, y in zip(Xs, Xs):
                    # feature_mat = torch.Tensor(X).T
                    # if pert_idx is None:
                    pert_idx = [-1]
                    gears_pert = 'ctrl'
                    tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                                y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert,
                                pert_flags=pert_flags.reshape(1,-1))
                    cell_graphs.append(tmp_Data)
                    
                pert_cell_graphs[pert] = cell_graphs
                tmp_perturbation_list = [pert]

            else:
                pert_groups = tmp_adata_split.obs.groupby('perturbation_group')
                pert_groups_dict = {name: list(group.index) for name, group in pert_groups}
                for pert in tqdm(tmp_perturbation_list):
                    pert_indices = pert_groups_dict[pert]
                    adata_pert = tmp_adata_split[pert_indices]  # 直接使用索引列表
                    Xs = adata_pert.X # ctrl value
                    ys = self.adata_split[list(adata_pert.obs['control_barcode'])].X # perturb value
                    Xs, ys = ys, Xs # the input must be ctrl!

                    if pert_type_mode:
                        pert_type = adata_pert.obs.pert_type[0]

                    # - get the de_idx for pert
                    if 'rank_genes_groups' in adata_pert.uns:
                        de_genes = adata_pert.uns['rank_genes_groups']
                        self.de = True
                    else:
                        self.de = False
                        # num_de_genes = 1
                        
                    if self.de:
                        de_idx = np.where(adata_pert.var_names.isin(
                        np.array(de_genes[pert][:num_de_genes])))[0]
                    else:
                        de_idx = [-1] * num_de_genes
                        
                    # - get the pert_idx [TODO]
                    # pert_idx = self.get_pert_idx(pert)
                    pert_idx = [-1]
                    
                    if not isinstance(Xs, np.ndarray):
                        Xs = Xs.toarray()
                    if not isinstance(ys, np.ndarray):
                        ys = ys.toarray()

                    if not pert_flag_mode:
                        # - pert_flags for multi perts
                        pert_flags = torch.zeros(Xs.shape[1])
                        for gene in pert.split(' | ')[0].split('; '):
                            if gene not in self.var_idx_dict:
                                continue
                            else:
                                pert_flags[self.var_idx_dict[gene]] = 1
                    else:
                        pert_flags = np.array((drug_gene_df.loc[:, pert.split(' | ')[0]]))
                        
                    cell_graphs = []
                    # Create cell graphs
                    for X, y in zip(Xs, ys):
                        # feature_mat = torch.Tensor(X).T
                        if pert_idx is None:
                            pert_idx = [-1]

                        # - if multiple perts
                        # if '; ' in pert:
                        #     gears_pert = '+'.join(pert.split(' | ')[0].split('; ')) # change to gears pert
                        # else:
                        #     gears_pert = '+'.join(pert.split(' | ')[0].split('; ')+['ctrl']) # change to gears pert
                            
                        gears_pert = pert
                        if not pert_type_mode:
                            tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                                        y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert,
                                        pert_flags=torch.Tensor(pert_flags.reshape(1,-1)))
                        else:
                            tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                                            y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert,
                                            pert_flags=torch.Tensor(pert_flags.reshape(1,-1)), pert_type = pert_type)
                        cell_graphs.append(tmp_Data)
                        
                    pert_cell_graphs[pert] = cell_graphs
                    
                    if test_mode:
                        break

                # - add control to training set
                if add_control and dataset == 'train':
                    pert = control_perturbation
                    tmp_perturbation_list.append(control_perturbation)
                    adata_pert = self.adata_split[self.adata_split.obs['perturbation_group']==pert]
                    Xs = adata_pert.X # ctrl value
                    # ys = self.adata_split[adata_pert.obs['control_barcode']].X # perturb value
                    pert_flags = torch.zeros(Xs.shape[1])
                    if not isinstance(Xs, np.ndarray):
                        Xs = Xs.toarray()
                    
                    self.de = False
                    # num_de_genes = 1
                        
                    if self.de:
                        de_idx = np.where(adata_pert.var_names.isin(
                        np.array(de_genes[pert][:num_de_genes])))[0]
                    else:
                        de_idx = [-1]*num_de_genes
                        
                    cell_graphs = []
                    # Create cell graphs
                    for X, y in zip(Xs, Xs):
                        # feature_mat = torch.Tensor(X).T
                        # if pert_idx is None:
                        pert_idx = [-1]
                        gears_pert = 'ctrl'
                        tmp_Data = Data(x=torch.Tensor(X.reshape(1,-1)), pert_idx=pert_idx,
                                    y=torch.Tensor(y.reshape(1,-1)), de_idx=de_idx, pert=gears_pert,
                                    pert_flags=pert_flags.reshape(1,-1))
                        cell_graphs.append(tmp_Data)
                        
                    pert_cell_graphs[pert] = cell_graphs
            
            # - get the result        
            self.dataset_perturbation_list[dataset] = tmp_perturbation_list
            self.dataset_pert_cell_graphs[dataset] = pert_cell_graphs
                
            if test_mode:
                self.dataset_perturbation_list[dataset] = list(pert_cell_graphs.keys())
                self.dataset_pert_cell_graphs[dataset] = pert_cell_graphs
        print('='*10,f'get Data_scGPT finished!')

    def data_split_2(self,
                split_type = 0,
                test_perts = None):
        """
        This function is used to filter the perturbations with less cells, and split the data
        """
        # - initial a data_split column
        obs_df_split = self.adata_split.obs
        split_col = f'data_split_{split_type}'
        obs_df_split[split_col] = 'train'
            
        # if split_type == 0:
            
        #     # - split all the pert to 8:2
        #     for pert in tqdm(self.filter_perturbation_list):

        #         # - get total obs_names of the pert
        #         obs_df_sub_idx = np.array(obs_df_split[obs_df_split['perturbation_group']==pert].index)

        #         np.random.seed(self.seed)
        #         np.random.shuffle(obs_df_sub_idx)

        #         # - data split
        #         split_point_1 = int(len(obs_df_sub_idx) * self.split_ratio[0])
        #         split_point_2 = int(len(obs_df_sub_idx) * (self.split_ratio[0]+self.split_ratio[1]))
        #         train = obs_df_sub_idx[:split_point_1]
        #         test = obs_df_sub_idx[split_point_1:split_point_2]
        #         val = obs_df_sub_idx[split_point_2:]

        #         # - set the test row
        #         obs_df_split.loc[test,split_col] = 'test'
        #         obs_df_split.loc[val,split_col] = 'val'
        
        if split_type == 0:
            perts = np.unique(self.filter_perturbation_list)
            np.random.seed(self.seed)
            np.random.shuffle(perts)

            # - data split
            split_point_1 = int(len(perts) * self.split_ratio[0])
            split_point_2 = int(len(perts) * (self.split_ratio[0]+self.split_ratio[1]))
            train_perts = perts[:split_point_1]
            test_perts = perts[split_point_1:split_point_2]
            val_perts = perts[split_point_2:]
            
            train = np.array(obs_df_split[obs_df_split['perturbation_group'].isin(train_perts)].index)
            test = np.array(obs_df_split[obs_df_split['perturbation_group'].isin(test_perts)].index)
            val = np.array(obs_df_split[obs_df_split['perturbation_group'].isin(val_perts)].index)
            
            obs_df_split.loc[test,split_col] = 'test'
            obs_df_split.loc[val,split_col] = 'val'
                
                
        elif split_type == 1:
            perts = np.unique([i.split(' | ')[0] for i in self.filter_perturbation_list])
            np.random.seed(self.seed)
            np.random.shuffle(perts)

            # - data split
            split_point_1 = int(len(perts) * self.split_ratio[0])
            split_point_2 = int(len(perts) * (self.split_ratio[0]+self.split_ratio[1]))
            train_perts = perts[:split_point_1]
            test_perts = perts[split_point_1:split_point_2]
            val_perts = perts[split_point_2:]
            
            train = np.array(obs_df_split[obs_df_split['perturbation_new'].isin(train_perts)].index)
            test = np.array(obs_df_split[obs_df_split['perturbation_new'].isin(test_perts)].index)
            val = np.array(obs_df_split[obs_df_split['perturbation_new'].isin(val_perts)].index)
            
            obs_df_split.loc[test,split_col] = 'test'
            obs_df_split.loc[val,split_col] = 'val'
            
        elif split_type == 2:
            perts = np.unique([i.split(' | ')[-1] for i in self.filter_perturbation_list])
            np.random.seed(self.seed)
            np.random.shuffle(perts)

            # - data split
            split_point_1 = int(len(perts) * self.split_ratio[0])
            split_point_2 = int(len(perts) * (self.split_ratio[0]+self.split_ratio[1]))
            train_perts = perts[:split_point_1]
            test_perts = perts[split_point_1:split_point_2]
            val_perts = perts[split_point_2:]
            
            train = np.array(obs_df_split[obs_df_split['celltype_new'].isin(train_perts)].index)
            test = np.array(obs_df_split[obs_df_split['celltype_new'].isin(test_perts)].index)
            val = np.array(obs_df_split[obs_df_split['celltype_new'].isin(val_perts)].index)
            
            obs_df_split.loc[test,split_col] = 'test'
            obs_df_split.loc[val,split_col] = 'val'
            
        elif split_type == -1:
            test = np.array(obs_df_split[obs_df_split['perturbation_new'].isin(test_perts)].index)
            obs_df_split.loc[test,split_col] = 'test'
            
        else:
            raise ValueError("error of split_type")

        self.adata_split.obs = obs_df_split.copy()
        self.obs_df_split = obs_df_split
        # self.train_perts = train_perts
        # self.test_perts = test_perts
        # self.val_perts = val_perts
        
        # self.train_pert_groups = [pert for pert in obs_df_split[obs_df_split[split_col] == 'train']['perturbation_group'].unique() if 'control' not in pert]
        # self.test_pert_groups = obs_df_split[obs_df_split[split_col] == 'test']['perturbation_group'].unique()
        # self.val_pert_groups = obs_df_split[obs_df_split[split_col] == 'val']['perturbation_group'].unique()
        print('='*10,f'data split finished!')
