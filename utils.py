from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import torch
from rpy2.robjects import numpy2ri
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

EOS = 1e-10


def normalize_adj(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value
        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())

def get_adj_from_edges(edges, weights, nnodes):
    adj = torch.zeros(nnodes, nnodes).cuda()
    adj[edges[0], edges[1]] = weights
    return adj

def mclust_R(embedding, num_cluster, modelNames='EEE', random_seed=2023):
    """
    使用 R 语言 mclust 进行聚类 (带 PCA 加速)
    """
    numpy2ri.activate()
    robjects.r['set.seed'](random_seed)
    
    if torch.is_tensor(embedding):
        embedding = embedding.detach().cpu().numpy()
        
    # 强制 PCA 降维到 30 (如果维度过高)
    if embedding.shape[1] > 30:
        pca = PCA(n_components=30, random_state=random_seed)
        embedding = pca.fit_transform(embedding)
    
    try:
        robjects.r('suppressPackageStartupMessages(library("mclust"))')
    except Exception as e:
        print(f"!! Error loading 'mclust': {e}")
        return None

    robjects.r('''
    cluster_mclust <- function(data, n_clusters, modelNames) {
      dimnames(data) <- NULL
      if (is.null(modelNames)) {
          result <- mclust::Mclust(data, G = n_clusters, verbose = FALSE)
      } else {
          result <- mclust::Mclust(data, G = n_clusters, modelNames = modelNames, verbose = FALSE)
      }
      return(result)
    }
    ''')
    rmclust = robjects.r['cluster_mclust']

    data = np.ascontiguousarray(embedding, dtype=np.float64)
    r_modelNames = modelNames if modelNames is not None else robjects.r("NULL")

    try:
        res = rmclust(data, num_cluster, r_modelNames)
        if 'classification' in res.names:
            mclust_res = np.array(res.rx2('classification')).astype('int')
        else:
            mclust_res = np.array(res[-2]).astype('int')
        return mclust_res - 1
    except Exception as e:
        print(f"!! Mclust Execution Error: {e}")
        return None

def refine_label_spatial(labels, spatial, k=20, verbose=True):
    """
    策略2: 标签空间修正 (多数投票)
    逻辑: 如果我的邻居大部分是类别 A，那我也应该是 A
    """
    if spatial is None:
        return labels
        
    if verbose:
        print(f"  [Refine] Refining labels (k={k})...")
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(spatial)
    _, indices = nbrs.kneighbors(spatial)
    
    new_labels = labels.copy()
    for i in range(len(labels)):
        neighbor_labels = labels[indices[i]]
        # 找到出现次数最多的标签
        most_common = Counter(neighbor_labels).most_common(1)[0][0]
        new_labels[i] = most_common
        
    return new_labels

def plot_spatial_hex_tiles(
    spatial,
    pred_labels,
    save_name="spatial_hex.png",
    title=None,
    tile_scale=1.03,
    figsize=(6, 6),
    dpi=300,
):
    """Draw a contiguous Visium-style hex tile map without gaps."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba

    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.detach().cpu().numpy()
    if torch.is_tensor(spatial):
        spatial = spatial.detach().cpu().numpy()

    spatial = np.asarray(spatial, dtype=float)
    pred_labels = np.asarray(pred_labels).reshape(-1)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError(f"spatial must be (N,2), got {spatial.shape}")
    if spatial.shape[0] != pred_labels.shape[0]:
        raise ValueError(f"spatial/pred length mismatch: {spatial.shape[0]} vs {pred_labels.shape[0]}")

    nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(spatial)
    d, _ = nbrs.kneighbors(spatial)
    nn = np.median(d[:, 1])
    radius = (nn / np.sqrt(3.0)) * tile_scale

    # Pointy-top hexagons match the 10x Visium lattice:
    # horizontal center spacing = sqrt(3) * radius; diagonal dy = 1.5 * radius.
    angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    unit_hex = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    verts = spatial[:, None, :] + unit_hex[None, :, :]

    palette = [
        "#2c7fb8", "#7bccc4", "#d9d100", "#009e73", "#b2184b", "#e31a1c",
        "#f0a202", "#003f5c", "#00a6a6", "#087f5b", "#08306b", "#8c2d04",
        "#54278f", "#c51b8a", "#41ab5d", "#feb24c", "#08519c", "#de2d26",
        "#006d2c", "#756bb1",
    ]
    cats = sorted(pd.unique(pred_labels), key=lambda v: int(v) if str(v).lstrip("-").isdigit() else str(v))
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
    facecolors = [to_rgba(color_map[c]) for c in pred_labels]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pc = PolyCollection(
        verts,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0,
        antialiaseds=False,
    )
    ax.add_collection(pc)
    ax.set_xlim(spatial[:, 0].min(), spatial[:, 0].max())
    ax.set_ylim(spatial[:, 1].max(), spatial[:, 1].min())
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=16, pad=8)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
        plt.savefig(save_name, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    else:
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(save_name, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"  [Plot] Saved hex spatial map to {save_name}")

def align_labels(pred_labels, true_labels):
    """
    使用匈牙利算法重新排列 pred_labels，使其与 true_labels 尽可能一致。
    这样每次运行，相同的解剖结构都会被赋予相同的 Cluster ID。
    """
    # 确保是 numpy 数组
    if torch.is_tensor(pred_labels): pred_labels = pred_labels.detach().cpu().numpy()
    if torch.is_tensor(true_labels): true_labels = true_labels.detach().cpu().numpy()

    # 过滤掉无效值 (如 -1)
    mask = (true_labels >= 0) & (pred_labels >= 0)
    y_true = true_labels[mask]
    y_pred = pred_labels[mask]
    
    if len(y_true) == 0:
        return pred_labels

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # 计算混淆矩阵 (Overlap)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
        
    # 使用匈牙利算法寻找最大权重的匹配 (Maximize Accuracy)
    # linear_sum_assignment 寻找最小 Cost，所以我们取负号 -w
    row_ind, col_ind = linear_sum_assignment(-w)
    
    # row_ind 是原来的 pred 标签，col_ind 是它应该映射到的 true 标签
    # 创建映射字典
    reorder_map = {row: col for row, col in zip(row_ind, col_ind)}
    
    # 没匹配上的标签保持原样 (通常不会发生，除非类别数不一致)
    new_pred_labels = np.zeros_like(pred_labels)
    for i in range(len(pred_labels)):
        old_lbl = pred_labels[i]
        if old_lbl in reorder_map:
            new_pred_labels[i] = reorder_map[old_lbl]
        else:
            new_pred_labels[i] = old_lbl
            
    return new_pred_labels
