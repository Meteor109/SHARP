import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import remove_self_loops

warnings.simplefilter("ignore")


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8, num_splits: int = 10):

    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    trains, vals, tests = [], [], []

    for _ in range(num_splits):
        indices = torch.randperm(num_samples)

        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        train_mask.fill_(False)
        train_mask[indices[:train_size]] = True

        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask.fill_(False)
        test_mask[indices[train_size: test_size + train_size]] = True

        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask.fill_(False)
        val_mask[indices[test_size + train_size:]] = True

        trains.append(train_mask.unsqueeze(1))
        vals.append(val_mask.unsqueeze(1))
        tests.append(test_mask.unsqueeze(1))

    train_mask_all = torch.cat(trains, 1)
    val_mask_all = torch.cat(vals, 1)
    test_mask_all = torch.cat(tests, 1)

    return train_mask_all, val_mask_all, test_mask_all

def get_structural_encoding(edges, nnodes, str_enc_dim=16):

    row = edges[0, :].numpy()
    col = edges[1, :].numpy()
    data = np.ones_like(row)

    A = sp.csr_matrix((data, (row, col)), shape=(nnodes, nnodes))
    D = (np.array(A.sum(1)).squeeze()) ** -1.0

    Dinv = sp.diags(D)
    RW = A * Dinv
    M = RW

    SE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(str_enc_dim - 1):
        M_power = M_power * M
        SE.append(torch.from_numpy(M_power.diagonal()).float())
    SE = torch.stack(SE, dim=-1)
    return SE

def load_candies_E18(path, k_neighbors=30):
    print(f"Loading E18 data from {path}...")
    # 1. 读取数据
    # 假设文件名为 E18_adata_rna.h5ad，请确保路径正确
    file_path = os.path.join(path, 'E18_adata_rna.h5ad')
    adata = sc.read_h5ad(file_path)
    adata.var_names_make_unique()

    # 2. 设置空间坐标 (参照 notebook cell 4)
    # 提取 array_col 和 array_row 作为空间坐标
    spatial_coords = np.array(adata.obs[['array_col', 'array_row']])
    adata.obsm['spatial'] = spatial_coords

    # 3. 预处理 (参照 notebook cell 8)
    # 过滤基因
    sc.pp.filter_genes(adata, min_cells=10)
    # 选择高变基因
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    # 归一化
    sc.pp.normalize_total(adata, target_sum=1e4)
    # Log 变换
    sc.pp.log1p(adata)
    # 标准化 (Scale)
    sc.pp.scale(adata)
    
    # PCA 降维 (作为节点特征输入 GREET)
    sc.tl.pca(adata, n_comps=60)
    features = torch.FloatTensor(adata.obsm['X_pca'].copy())

    # 4. 构建图结构 (Spatial Neighbor Graph)
    # Notebook 中使用了 construct_neighbor_graph，通常 ST 数据使用空间 KNN 图
    # 这里我们基于 spatial 坐标构建 k=6 或 k=10 的近邻图
    k_neighbors = int(k_neighbors)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    
    # 构建 edge_index (Source, Target)
    src = []
    dst = []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]): # 跳过自身 (index 0)
            src.append(i)
            dst.append(indices[i, j])
            # 如果需要无向图，通常 GREET 会处理，或者这里添加双向边
            src.append(indices[i, j])
            dst.append(i)
            
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = remove_self_loops(edge_index)[0]

    # 5. 处理标签 (Labels)
    # Notebook 中使用 'Combined_Clusters' 作为 Ground Truth
    if 'Combined_Clusters' in adata.obs:
        # 将字符串标签转换为整数 ID
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs['Combined_Clusters'])
        labels = torch.LongTensor(labels)
    else:
        print("Warning: 'Combined_Clusters' not found. Using dummy labels.")
        labels = torch.zeros(adata.n_obs, dtype=torch.long)

    # 6. 获取基本信息
    nnodes = features.shape[0]
    nfeats = features.shape[1]
    
    # 7. 划分训练/验证/测试集
    # 使用 GREET 自带的划分函数
    train_mask, val_mask, test_mask = get_split(nnodes)
    
    # 8. 计算结构编码 (Structural Encoding)
    # 注意：这会保存到 '../data/se/E18_mouse_brain'，请确保文件夹存在或代码能自动创建
    se_path = './data/se/E18_mouse_brain'
    if not os.path.exists(se_path):
        os.makedirs(se_path)
    
    file_name = se_path + '/{}_{}.pt'.format('E18_mouse_brain', 16)
    if os.path.exists(file_name):
        se = torch.load(file_name)
    else:
        print('Computing structural encoding for E18...')
        se = get_structural_encoding(edge_index, nnodes)
        torch.save(se, file_name)
        print('Done. The structural encoding is saved as: {}.'.format(file_name))

    if 'layer_guess' in adata.obs and 'le' in locals():
        # 如果定义了 LabelEncoder，就获取名字列表
        class_names = le.classes_
    else:
        # 如果没有标签，或者是全 -1
        class_names = None

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, class_names, nnodes, nfeats

def load_BRCA(path, k_neighbors=30):
    print(f"Loading BRCA data from {path} ")

    data_dir = os.path.join(path, 'Breast_Caner')
    h5_path = os.path.join(data_dir, 'filtered_feature_bc_matrix.h5')
    pos_path = os.path.join(data_dir, 'spatial', 'tissue_positions_list.csv')
    label_path = os.path.join(data_dir, 'spatial', 'brca_ground_truth_annotation.csv')

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Could not find {h5_path}")
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"Could not find {pos_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Could not find {label_path}")

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    positions = pd.read_csv(pos_path, header=None).iloc[:, :6].copy()
    positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'py', 'px']
    positions['barcode'] = positions['barcode'].astype(str)
    positions.set_index('barcode', inplace=True)

    labels_df = pd.read_csv(label_path)
    labels_df['spot_id'] = labels_df['spot_id'].astype(str)
    labels_df.set_index('spot_id', inplace=True)

    adata.obs.index = adata.obs.index.astype(str)
    adata.obs = adata.obs.join(positions, how='left')
    adata.obs = adata.obs.join(labels_df[['ground_truth_annotation']], how='left')

    if adata.obs['in_tissue'].isna().all():
        raise ValueError("Failed to align BRCA tissue_positions_list.csv with 10x barcodes.")
    if adata.obs['ground_truth_annotation'].isna().all():
        raise ValueError("Failed to align BRCA ground-truth annotation with 10x barcodes.")

    adata = adata[(adata.obs['in_tissue'] == 1) & (~adata.obs['ground_truth_annotation'].isna())].copy()
    adata.obsm['spatial'] = adata.obs[['px', 'py']].astype(float).values

    print("Running E18-style preprocessing for BRCA...")
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    sc.tl.pca(adata, n_comps=60)
    features = torch.FloatTensor(adata.obsm['X_pca'].copy())

    k_neighbors = int(k_neighbors)
    spatial_coords = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)

    src, dst = [], []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):
            src.append(i)
            dst.append(indices[i, j])
            src.append(indices[i, j])
            dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = remove_self_loops(edge_index)[0]

    le = LabelEncoder()
    labels = torch.LongTensor(le.fit_transform(adata.obs['ground_truth_annotation'].astype(str)))

    nnodes = features.shape[0]
    nfeats = features.shape[1]
    train_mask, val_mask, test_mask = get_split(nnodes)

    se_path = './data/se/BRCA'
    if not os.path.exists(se_path):
        os.makedirs(se_path)

    file_name = os.path.join(se_path, f'BRCA_k{k_neighbors}_16.pt')
    if os.path.exists(file_name):
        se = torch.load(file_name)
    else:
        print('Computing structural encoding for BRCA...')
        se = get_structural_encoding(edge_index, nnodes)
        torch.save(se, file_name)
        print(f'Done. The structural encoding is saved as: {file_name}.')

    class_names = le.classes_
    return features, edge_index, se, train_mask, val_mask, test_mask, labels, class_names, nnodes, nfeats

def load_Stereo_AB(path=None, annotation_col='Annotation', k_neighbors=20, pca_dim=80):

    if path is None:
        path = './data/Stereo/Stereo_AB.h5ad'

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stereo_AB.h5ad not found at: {path}\n"
            "Please place it at ./data/Stereo/Stereo_AB.h5ad (default) "
            "or pass a valid path to load_Stereo_AB(path=...)."
        )

    print(f"Loading Stereo_AB data from {path} (Label: {annotation_col}) ...")

    # 2) load h5ad
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    # 3) spatial coords
    if 'spatial' not in adata.obsm:
        # fallback: try obs columns
        if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
            adata.obsm['spatial'] = adata.obs[['x', 'y']].values
        elif 'spatial_x' in adata.obs.columns and 'spatial_y' in adata.obs.columns:
            adata.obsm['spatial'] = adata.obs[['spatial_x', 'spatial_y']].values
        elif 'array_col' in adata.obs.columns and 'array_row' in adata.obs.columns:
            adata.obsm['spatial'] = adata.obs[['array_col', 'array_row']].values
        else:
            raise ValueError("Could not find spatial coordinates in adata.obsm['spatial'] or common obs columns.")

    spatial_coords = np.asarray(adata.obsm['spatial'], dtype=float)

    # 4) ground truth
    if annotation_col not in adata.obs.columns:
        raise ValueError(f"Annotation column '{annotation_col}' not found in adata.obs. Available: {list(adata.obs.columns)}")

    adata.obs['ground_truth'] = adata.obs[annotation_col].astype(str)

    # 5) preprocessing (lightweight, close to your reference)
    # Use raw counts if provided as a layer; otherwise fall back to X.
    if 'count' not in adata.layers:
        if 'counts' in adata.layers:
            adata.layers['count'] = adata.layers['counts']
        else:
            adata.layers['count'] = adata.X.copy() if not sp.issparse(adata.X) else adata.X

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_genes(adata, min_counts=10)

    # HVG on counts (Seurat v3 expects counts-like)
    try:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=3000)
        adata = adata[:, adata.var['highly_variable']].copy()
    except Exception as e:
        print(f"!! Warning: HVG selection failed ({e}). Proceeding without HVG filtering.")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    # PCA as node features
    sc.tl.pca(adata, n_comps=pca_dim, svd_solver='arpack')
    features = torch.FloatTensor(adata.obsm['X_pca'].copy())

    # 6) construct kNN graph on spatial coords (n=4 in your reference)
    nnodes = features.shape[0]
    nfeats = features.shape[1]

    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree').fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)

    src, dst = [], []
    for i in range(nnodes):
        for j in range(1, k_neighbors + 1):
            src.append(i); dst.append(indices[i, j])
            src.append(indices[i, j]); dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)

    # 7) labels + masks (60/20/20, repeated to 10 splits shape)
    le = LabelEncoder()
    labels = torch.LongTensor(le.fit_transform(adata.obs['ground_truth'].astype(str)))

    idx = np.random.permutation(nnodes)
    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)

    train_mask[idx[:int(0.6 * nnodes)]] = True
    val_mask[idx[int(0.6 * nnodes):int(0.8 * nnodes)]] = True
    test_mask[idx[int(0.8 * nnodes):]] = True

    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    # 8) structural encoding (cache)
    se_path = './data/se/Stereo_AB'
    if not os.path.exists(se_path):
        os.makedirs(se_path)

    se_file = os.path.join(se_path, f'Stereo_AB_k{k_neighbors}_16.pt')
    if os.path.exists(se_file):
        se = torch.load(se_file)
    else:
        print("Computing structural encoding...")
        se = get_structural_encoding(edge_index, nnodes, str_enc_dim=16)
        torch.save(se, se_file)

    if 'layer_guess' in adata.obs and 'le' in locals():
        # 如果定义了 LabelEncoder，就获取名字列表
        class_names = le.classes_
    else:
        # 如果没有标签，或者是全 -1
        class_names = None

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, class_names, nnodes, nfeats

def load_Stereo_DPI(path=None, annotation_col=None, k_neighbors=20, pca_dim=60,
                    min_cells=10, min_counts=10, n_top_genes=2000, str_enc_dim=16):

    # ---- 1) resolve file path ----
    if path is None:
        path = './data/Stereo/Stereo_DPI.h5ad'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stereo_DPI.h5ad not found at: {path}\n"
            "Please place it at ./data/Stereo/Stereo_DPI.h5ad (default) "
            "or pass a valid path to load_Stereo_AB(path=...)."
        )

    print(f"Loading Stereo_DPI data from {path} (Label: {annotation_col}) ...")

    # ---- 2) load AnnData ----
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()

    # ---- 3) spatial coordinates ----
    if 'spatial' in adata.obsm:
        spatial = adata.obsm['spatial']
    elif 'x_array' in adata.obs.columns and 'y_array' in adata.obs.columns:
        spatial = adata.obs[['x_array', 'y_array']].values
    elif 'x' in adata.obs.columns and 'y' in adata.obs.columns:
        spatial = adata.obs[['x', 'y']].values
    else:
        raise ValueError("Could not find spatial coordinates in obsm['spatial'] or obs[['x_array','y_array']] / obs[['x','y']].")
    adata.obsm['spatial'] = spatial.astype(float)

    # ---- labels: choose a valid column, then map to 'layer_guess' ----
    if annotation_col is not None and annotation_col in adata.obs.columns:
        label_src = annotation_col
    elif 'ground_truth' in adata.obs.columns:
        label_src = 'ground_truth'
    elif 'Annotation' in adata.obs.columns:
        label_src = 'Annotation'
    else:
        raise ValueError(
            f"No label column found. Tried: {annotation_col}, 'ground_truth', 'Annotation'. "
            f"Available obs columns: {list(adata.obs.columns)}"
        )

    adata.obs['layer_guess'] = adata.obs[label_src]

    # 统一转成 string 再填 NA，避免 categorical 报错
    adata.obs['layer_guess'] = adata.obs['layer_guess'].astype('string').fillna('Unknown').astype(str)

    print(f"  -> Using labels from obs['{label_src}'] (n_classes={pd.Series(adata.obs['layer_guess']).nunique()})")



    # ---- 5) preprocessing (Scanpy, SEDR-like but lighter defaults) ----
    print("Preprocessing (filter/normalize/HVG/scale/PCA)...")

    # keep a count layer for HVG
    if 'counts' in adata.layers:
        # keep sparse if present
        adata.layers['count'] = adata.layers['counts']
    else:
        adata.layers['count'] = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_genes(adata, min_counts=min_counts)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable'] == True].copy()
    sc.pp.scale(adata)

    pca = PCA(n_components=pca_dim, random_state=2023)
    X_mat = adata.X if not sp.issparse(adata.X) else adata.X.toarray()
    feat_pca = pca.fit_transform(X_mat)
    features = torch.FloatTensor(feat_pca)

    # ---- 6) graph construction: spatial kNN ----
    print(f"Constructing Spatial Graph (k={k_neighbors})...")
    cell_position = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='ball_tree').fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)

    src, dst = [], []
    for i in range(cell_position.shape[0]):
        for j in range(1, k_neighbors + 1):
            nb = int(indices[i, j])
            src.append(i); dst.append(nb)
            src.append(nb); dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)

    # ---- 7) label encoding + masks ----
    le = LabelEncoder()
    labels_np = le.fit_transform(adata.obs['ground_truth'].astype(str))
    labels = torch.LongTensor(labels_np)

    nnodes = features.shape[0]
    nfeats = features.shape[1]

    valid_mask = (adata.obs['layer_guess'].astype(str) != 'Unknown')

    valid_indices = np.where(valid_mask.values)[0]
    np.random.shuffle(valid_indices)

    n_valid = len(valid_indices)
    train_end = int(0.6 * n_valid)
    val_end = int(0.8 * n_valid)

    train_mask = torch.zeros(nnodes, dtype=torch.bool)
    val_mask = torch.zeros(nnodes, dtype=torch.bool)
    test_mask = torch.zeros(nnodes, dtype=torch.bool)

    train_mask[valid_indices[:train_end]] = True
    val_mask[valid_indices[train_end:val_end]] = True
    test_mask[valid_indices[val_end:]] = True

    train_mask = train_mask.unsqueeze(1).repeat(1, 10)
    val_mask = val_mask.unsqueeze(1).repeat(1, 10)
    test_mask = test_mask.unsqueeze(1).repeat(1, 10)

    # ---- 8) structural encoding (SE) ----
    se_path = './data/se/Stereo_DPI'
    if not os.path.exists(se_path):
        os.makedirs(se_path)

    file_name = os.path.join(se_path, f'Stereo_DPI_k{k_neighbors}_{str_enc_dim}.pt')
    if os.path.exists(file_name):
        se = torch.load(file_name)
    else:
        print('Computing structural encoding...')
        se = get_structural_encoding(edge_index, nnodes, str_enc_dim=str_enc_dim)
        torch.save(se, file_name)
        print(f'Done. Saved SE to {file_name}')

    return features, edge_index, se, train_mask, val_mask, test_mask, labels, nnodes, nfeats


def load_data(dataset_name, graph_k=None):
    """Load one of the four public SHARP benchmarks."""
    if dataset_name == "E18_mouse_brain":
        return load_candies_E18("./data/E18.5_mouse_brain", k_neighbors=10 if graph_k is None else graph_k)
    if dataset_name == "BRCA":
        return load_BRCA("./data/BRCA", k_neighbors=30 if graph_k is None else graph_k)
    if dataset_name == "Stereo_AB":
        return load_Stereo_AB(k_neighbors=20 if graph_k is None else graph_k)
    if dataset_name == "Stereo_DPI":
        return load_Stereo_DPI("./data/Stereo/Stereo_DPI.h5ad", k_neighbors=20 if graph_k is None else graph_k)
    raise ValueError(
        f"Unsupported dataset: {dataset_name}. "
        "Choose from BRCA, E18_mouse_brain, Stereo_AB, Stereo_DPI."
    )
