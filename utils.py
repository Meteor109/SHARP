import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import dgl
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scanpy as sc
import pandas as pd
import anndata
# R 语言接口 (用于 mclust)
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from matplotlib.lines import Line2D
import os
EOS = 1e-10
import seaborn as sns

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

def generate_random_node_pairs(nnodes, nedges, backup=300):
    rand_edges = np.random.choice(nnodes, size=(nedges + backup) * 2, replace=True)
    rand_edges = rand_edges.reshape((2, nedges + backup))
    rand_edges = torch.from_numpy(rand_edges)
    rand_edges = rand_edges[:, rand_edges[0,:] != rand_edges[1,:]]
    rand_edges = rand_edges[:, 0: nedges]
    return rand_edges.cuda()

def eval_test_mode(embedding, labels, train_mask, val_mask, test_mask):
    X = embedding.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy()
    X = normalize(X, norm='l2')

    X_train = X[train_mask.cpu()]
    X_val = X[val_mask.cpu()]
    X_test = X[test_mask.cpu()]
    y_train = Y[train_mask.cpu()]
    y_val = Y[val_mask.cpu()]
    y_test = Y[test_mask.cpu()]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    y_pred_val = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_pred_val)

    return acc_test * 100, acc_val * 100

# --- 新增函数 ---

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

def eval_clustering(embedding, labels, n_clusters):
    """ K-Means 备用方案 """
    from sklearn.cluster import KMeans
    X = embedding.detach().cpu().numpy() if torch.is_tensor(embedding) else embedding
    Y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    pred_labels = kmeans.fit_predict(X)
    ari = adjusted_rand_score(Y, pred_labels)
    return ari, pred_labels

def plot_tsne(embedding, labels, save_name='tsne_plot.png'):
    """ t-SNE 可视化并保存 """
    X = embedding.detach().cpu().numpy() if torch.is_tensor(embedding) else embedding
    
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='jet', s=10, alpha=0.6)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"  [Plot] Saved to {save_name}")


def smooth_embedding_spatial(embedding, spatial, k=15, alpha=0.5):

    if spatial is None:
        return embedding
    
    # 确保是 Numpy
    if torch.is_tensor(embedding):
        embedding = embedding.detach().cpu().numpy()
        
    print(f"  [Refine] Smoothing embedding (k={k}, alpha={alpha})...")
    
    # 拟合空间坐标
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(spatial)
    _, indices = nbrs.kneighbors(spatial)
    
    emb_smooth = np.zeros_like(embedding)
    for i in range(len(embedding)):
        # 计算邻居的平均 Embedding
        neighbor_emb = np.mean(embedding[indices[i]], axis=0)
        # 加权融合
        emb_smooth[i] = alpha * embedding[i] + (1 - alpha) * neighbor_emb
        
    return emb_smooth

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


def plot_spatial(
    spatial,
    pred_labels,
    true_labels=None,
    title="Prediction",
    save_name="spatial.png",
    emb=None,
    spot_size=200,
    class_names=None,
    all_spatial_plot=None,
    spatial_uns=None,
    library_id=None,
    img_key="hires",
    xlim=None, 
    ylim=None,
    single_panel=False,
    show_background=True,
    show_legend=True,
    show_title=True,
    save_umap=True,
):
    """Scanpy-like spatial + (optional) UMAP plotting.

    Key features for Visium-style datasets:
    - If `spatial_uns` + `library_id` + `img_key` are provided, draw H&E image as background
      and align scatter using scalefactors_json (fullres pixel coords * scalef).
    - If `all_spatial_plot` is provided, draw empty-ring dots for *all* spots (including in_tissue=0),
      then overlay colored tissue spots.

    Notes:
    - `spatial` should be tissue-spot coordinates in **fullres pixel** space: (x=pxl_col_in_fullres, y=pxl_row_in_fullres).
    - `all_spatial_plot` should be all-spot coordinates in the same fullres pixel space.
    - This function only changes the *presentation* (background/origin/legend layout); it does NOT change model outputs.
    """
    if spatial is None:
        return

    # --- to numpy ---
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.detach().cpu().numpy()
    if true_labels is not None and torch.is_tensor(true_labels):
        true_labels = true_labels.detach().cpu().numpy()
    if torch.is_tensor(spatial):
        spatial = spatial.detach().cpu().numpy()
    if emb is not None and torch.is_tensor(emb):
        emb = emb.detach().cpu().numpy()
    if all_spatial_plot is not None and torch.is_tensor(all_spatial_plot):
        all_spatial_plot = all_spatial_plot.detach().cpu().numpy()

    spatial = np.asarray(spatial)
    if all_spatial_plot is not None:
        all_spatial_plot = np.asarray(all_spatial_plot)

    # --- build categorical labels (for stable color/legend) ---
    # Manual annotation
    if true_labels is None:
        manual = pd.Categorical(["Unknown"] * spatial.shape[0])
    else:
        if class_names is not None:
            mapped = []
            for v in true_labels:
                if v == -1:
                    mapped.append("Unknown")
                elif int(v) < len(class_names):
                    mapped.append(str(class_names[int(v)]))
                else:
                    mapped.append(str(v))
            manual = pd.Categorical(mapped)
        else:
            manual = pd.Categorical([str(v) for v in true_labels])

    # Prediction
    pred = pd.Categorical([f"Cluster {int(v)}" if str(v).lstrip('-').isdigit() else str(v) for v in pred_labels])

    def _cat_color_map(cat: pd.Categorical):
        cats = list(cat.categories)
        cmap = plt.get_cmap("tab20")
        return {c: cmap(i % 20) for i, c in enumerate(cats)}

    manual_cmap = _cat_color_map(manual)
    pred_cmap = _cat_color_map(pred)

    # --- background image (optional) ---
    img = None
    sf = 1.0
    if show_background and spatial_uns is not None and library_id is not None:
        try:
            lib = spatial_uns[library_id]
            img = lib.get("images", {}).get(img_key, None)
            sfs = lib.get("scalefactors", {})
            # 10x keys: tissue_hires_scalef / tissue_lowres_scalef
            sf = float(sfs.get(f"tissue_{img_key}_scalef", 1.0))
        except Exception:
            img = None
            sf = 1.0

    # scale coords to match chosen image
    spatial_plot = spatial * sf
    grid = None
    if all_spatial_plot is not None:
        # Guard: make sure it's (N,2)
        grid = np.asarray(all_spatial_plot)
        if grid.ndim != 2 or grid.shape[1] != 2:
            raise ValueError(f"all_spatial_plot must be (N,2), got {grid.shape}")
        grid = grid * sf

    def _auto_scatter_s(ax, xy, spot_size, base=200.0, frac=0.85):
        """
        把“scanpy风格的 spot_size(你习惯用200)”换算成 matplotlib.scatter 的 s(points^2),
        目标：点的直径约等于最近邻距离的 frac 倍，从而不糊成块。
        """
        xy = np.asarray(xy)
        if xy.ndim != 2 or xy.shape[0] < 5:
            return float(spot_size)

        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(xy)
        d, _ = nbrs.kneighbors(xy)
        nn = np.median(d[:, 1])
        if not np.isfinite(nn) or nn <= 0:
            return float(spot_size)

        # 你传的 spot_size=200 视为 “1.0倍”
        scale = float(spot_size) / float(base)
        diam_data = frac * nn * scale  # 希望的点直径（数据坐标单位）

        fig = ax.figure
        p0 = ax.transData.transform((0, 0))
        p1 = ax.transData.transform((diam_data, 0))
        diam_px = abs(p1[0] - p0[0])
        diam_pt = diam_px * 72.0 / fig.dpi
        return max(1.0, diam_pt ** 2)


    # --- plotting helpers ---
    def _draw_panel(ax, coords, cat, cmap_map, panel_title):
        # background image
        if img is not None:
            ax.imshow(img, origin="upper", zorder=0)
            h, w = img.shape[0], img.shape[1]
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # 与scanpy一致：左上为原点
            base_xy = grid if grid is not None else coords
        else:
            base_xy = grid if grid is not None else coords
            xmin, xmax = base_xy[:, 0].min(), base_xy[:, 0].max()
            ymin, ymax = base_xy[:, 1].min(), base_xy[:, 1].max()
            padx = (xmax - xmin) * 0.02
            pady = (ymax - ymin) * 0.02
            ax.set_xlim(xmin - padx, xmax + padx)
            ax.set_ylim(ymax + pady, ymin - pady)  # 保持 y 向下更接近ST图习惯

        # ---- 这里算一个“不会糊”的点大小（仍然由你传入的 spot_size 控制）----
        base_xy = grid if grid is not None else coords
        s_plot = _auto_scatter_s(ax, base_xy, spot_size)
        # empty rings for all spots
        if grid is not None:
            ax.scatter(
                grid[:, 0], grid[:, 1],
                s=s_plot, facecolors="none",
                edgecolors=(0, 0, 0, 0.25),
                linewidths=0.25, zorder=1
            )
        # colored tissue spots (draw per-category for stable legend/colors)
        handles = []
        labels = []
        for c in cat.categories:
            mask = (np.asarray(cat) == c)
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=s_plot, c=[cmap_map[c]],
                edgecolors="none", linewidths=0, zorder=2
            )

            handles.append(plt.Line2D([0], [0], marker='o', linestyle='', markersize=7,
                                     markerfacecolor=cmap_map[c], markeredgecolor='none'))
            labels.append(str(c))

        if show_title:
            ax.set_title(panel_title, fontsize=14)
        ax.set_aspect("equal")

        ax.axis("off")


        # legend at right margin
        if show_legend:
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      frameon=False, fontsize="small", handletextpad=0.4)

        # If image exists, clamp view to image extent so it looks like scanpy
        if img is not None:
            h, w = img.shape[0], img.shape[1]
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # keep origin at top
            
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        
        if ylim is not None:
            # 智能处理 Y 轴方向：
            # 空间转录组图通常 Y 轴是反的（0 在顶部，大值在底部）。
            # 为了防止用户填反导致图片倒过来，我们强制让 大值在底部(bottom)，小值在顶部(top)
            bottom, top = max(ylim), min(ylim)
            ax.set_ylim(bottom, top)


    if single_panel:
        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_panel(ax, spatial_plot, pred, pred_cmap, title)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(save_name, dpi=300, bbox_inches="tight", pad_inches=0)
    else:
        # --- Spatial: 2 panels ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        _draw_panel(axes[0], spatial_plot, manual, manual_cmap, "Manual Annotation")
        _draw_panel(axes[1], spatial_plot, pred, pred_cmap, title)
        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved Spatial map to {save_name}")

    # --- Optional UMAP from emb ---
    if save_umap and emb is not None:
        try:
            ad = anndata.AnnData(X=emb)
            ad.obs["Manual Annotation"] = manual
            ad.obs["Prediction"] = pred
            sc.pp.neighbors(ad, use_rep="X", metric="cosine")
            sc.tl.umap(ad, random_state=0)

            umap_save = save_name.replace(".png", "_UMAP.png")
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sc.pl.umap(
                ad,
                color="Manual Annotation",
                ax=axes[0],
                show=False,
                title="Manual Annotation (UMAP)",
                s=spot_size,
                palette="tab20",
                frameon=False,
                legend_loc="right margin",
                legend_fontsize="small",
            )
            sc.pl.umap(
                ad,
                color="Prediction",
                ax=axes[1],
                show=False,
                title=title.replace("Prediction", "UMAP"),
                s=spot_size,
                palette="tab20",
                frameon=False,
                legend_loc="right margin",
                legend_fontsize="small",
            )
            plt.tight_layout()
            plt.savefig(umap_save, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  [Plot] Saved UMAP projection to {umap_save}")
        except Exception as e:
            print(f"  !! UMAP plot failed: {e}")


    
def run_deg_analysis_151673(pred_labels, output_dir='./results/151673_analysis'):
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import torch
    
    # 尝试导入您的自定义绘图函数
    try:
        from utils import plot_deg_visualizations, plot_custom_volcano, plot_custom_split_violin, plot_spatial
    except ImportError:
        pass 

    print("\n[Analysis] Starting DEG/SVG Analysis for 151673...")
    
    # ==============================================================================
    # 1. 读取数据 (带图片!)
    # ==============================================================================
    data_root = './data/151673'
    if not os.path.exists(os.path.join(data_root, 'spatial')):
        print(f"!! Error: 'spatial' folder not found in {data_root}.")
        return

    print(f"  -> Loading Visium data (matrix + images) from {data_root}...")
    try:
        # 这一步非常关键，它会把图片读到 adata.uns['spatial'] 里
        adata = sc.read_visium(path=data_root, 
                               count_file='filtered_feature_bc_matrix.h5',
                               load_images=True) 
    except Exception as e:
        print(f"!! Error loading Visium data: {e}")
        return
        
    adata.var_names_make_unique()
    if 'in_tissue' in adata.obs:
        adata = adata[adata.obs['in_tissue'] == 1].copy()
    
    # 2. 预处理
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)

    # 3. 注入标签
    if torch.is_tensor(pred_labels):
        pred_labels = pred_labels.detach().cpu().numpy()
        
    if len(pred_labels) != adata.n_obs:
        print(f"!! Warning: Length mismatch! Labels({len(pred_labels)}) vs Data({adata.n_obs})")
    
    adata.obs['GREET_Cluster'] = pd.Categorical([f"Cluster {i}" for i in pred_labels])
    available_clusters = adata.obs['GREET_Cluster'].unique().tolist()
    print(f"  -> Available clusters: {available_clusters}")

    # 4. 计算全局差异基因 (用于概览)
    print("  -> Calculating marker genes...")
    sc.tl.rank_genes_groups(adata, 'GREET_Cluster', method='wilcoxon')
    try:
        plot_deg_visualizations(adata, output_dir)
    except: pass

    # =================================================================
    # 【已恢复并修改】自定义对比 Volcano + Split Violin
    #  按照您的要求：Cluster 4 vs 6, 4 vs 1, 1 vs 6
    # =================================================================
    
    comparisons = [
        ('Cluster 4', 'Cluster 6'), 
        ('Cluster 4', 'Cluster 1'),
        ('Cluster 1', 'Cluster 6')
    ]

    print(f"  -> Generating requested pairwise comparisons: {comparisons}")

    for group_test, group_ref in comparisons:
        # 1. 安全检查：确保这两个 Cluster 在当前结果中都存在
        # (如果某次跑出来的结果里没有 Cluster 6，代码会自动跳过，不会报错)
        if group_test not in available_clusters or group_ref not in available_clusters:
            print(f"     Skipping {group_test} vs {group_ref} (Cluster missing in this run)")
            continue

        print(f"     Processing {group_test} vs {group_ref}...")

        try:
            # 2. 计算这一对的差异基因 (为了找 Violin 的 Top 基因)
            sc.tl.rank_genes_groups(adata, groupby='GREET_Cluster', 
                                    groups=[group_test], 
                                    reference=group_ref, 
                                    method='wilcoxon', 
                                    key_added='temp_deg')
            
            # 3. 自动提取差异最大的 Top 6 基因 (3个上调，3个下调)
            df = sc.get.rank_genes_groups_df(adata, group=group_test, key='temp_deg')
            
            # 挑 group_test 高表达的 (logFC > 0)
            top_up = df[df['logfoldchanges'] > 0].sort_values('logfoldchanges', ascending=False).head(3)['names'].tolist()
            # 挑 group_ref 高表达的 (logFC < 0)
            top_down = df[df['logfoldchanges'] < 0].sort_values('logfoldchanges', ascending=True).head(3)['names'].tolist()
            
            violin_genes = top_up + top_down # 合并基因列表

            # 4. 画火山图 (Volcano)
            plot_custom_volcano(adata, group_test, group_ref, output_dir)
            
            # 5. 画分裂小提琴图 (Split Violin)
            plot_custom_split_violin(adata, group_test, group_ref, violin_genes, output_dir)
            
        except Exception as e:
            print(f"     !! Failed to process {group_test} vs {group_ref}: {e}")

    # =================================================================
    # 5. 保存 Top Markers 与 空间图
    # =================================================================
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(10).to_csv(os.path.join(output_dir, '151673_Top_Markers.csv'))
    
    # 获取图片数据
    spatial_uns = adata.uns.get('spatial', None)
    if spatial_uns is None:
        print("!! Warning: No image data found. Plot will be unaligned.")
    
    # 设置裁剪范围 (根据您的之前的设置)
    my_xlim = (4000, 12000) 
    my_ylim = (7000, 14000)

    try:
        print("  -> Plotting Aligned & Cropped Domain Map...")
        plot_spatial(
            spatial=adata.obsm['spatial'],
            pred_labels=pred_labels,
            true_labels=pred_labels,
            title="GREET Clusters (Cropped)",
            save_name=os.path.join(output_dir, '151673_Domain_Map_Cropped.png'),
            spot_size=300,           
            spatial_uns=spatial_uns,
            img_key="hires",
            xlim=my_xlim,
            ylim=my_ylim
        )
        print(f"  -> Saved cropped map.")
    except Exception as e:
        print(f"!! plot_spatial failed: {e}")
        # 兜底
        plt.figure(figsize=(8,8))
        sc.pl.spatial(adata, color='GREET_Cluster', img_key='hires', show=False)
        plt.savefig(os.path.join(output_dir, '151673_Domain_Map_Backup.png'))
        plt.close()

    print(f"[Done] Results saved to {output_dir}")



def plot_deg_visualizations(adata, output_dir, top_n=5):
    """
    【简化版】只绘制全局概览图：
    1. Matrixplot (全局热图)
    2. Stacked Violin Plot (全局堆叠小提琴)
    
    注意：具体的“二元对比” (Volcano/Split-Violin) 现在由 run_deg_analysis 函数中手动调用
    plot_custom_volcano 和 plot_custom_split_violin 来完成。
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import os

    print(f"  -> Generating global overview DEG plots in {output_dir}...")
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    if 'rank_genes_groups' not in adata.uns:
        print("  !! Error: 'rank_genes_groups' not found. Run rank_genes_groups first.")
        return

    # 准备 Marker Genes 列表 (用于全局展示)
    feature_names = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(top_n)
    marker_genes_all = feature_names.values.flatten().tolist()
    marker_genes_all = list(dict.fromkeys(marker_genes_all)) 
    marker_genes_all = [g for g in marker_genes_all if not g.startswith('MT-')]

    # ========================== 1. Matrixplot (全局热图) ==========================
    print("  -> Plotting Global Matrixplot...")
    plt.figure(figsize=(10, 8))
    # dendrogram=True 会展示 Cluster 之间的相似性聚类树，很有用
    sc.pl.matrixplot(adata, marker_genes_all, groupby='GREET_Cluster', 
                     standard_scale='var', cmap='viridis', 
                     show=False, dendrogram=True)
    plt.savefig(os.path.join(output_dir, 'DEG_Global_Matrixplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ========================== 2. Stacked Violin (全局概览) ==========================
    print("  -> Plotting Global Stacked Violin Plot...")
    plt.figure(figsize=(10, 8))
    sc.pl.stacked_violin(adata, marker_genes_all[:15], groupby='GREET_Cluster', 
                         show=False)
    plt.savefig(os.path.join(output_dir, 'DEG_Global_Stacked_Violin.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("  -> Global overview plots saved.")


def plot_custom_volcano(adata, group_test, group_reference, output_dir):
    """
    自定义火山图：Group Test vs Group Reference
    例如：Cluster 0 (Layer 5) vs Cluster 3 (White Matter)
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    print(f"  -> Plotting Custom Volcano: {group_test} vs {group_reference}...")
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1. 计算两个指定组之间的差异基因
    # method='wilcoxon' 是最常用的
    sc.tl.rank_genes_groups(adata, groupby='GREET_Cluster', 
                            groups=[group_test], 
                            reference=group_reference, 
                            method='wilcoxon', 
                            key_added='custom_deg')

    # 2. 提取数据
    result = sc.get.rank_genes_groups_df(adata, group=group_test, key='custom_deg')
    
    # 3. 处理数据 (P值和FC)
    # 给 P 值加个底数防止 log(0)
    result['log10p'] = -np.log10(result['pvals_adj'] + 1e-50)
    
    # 标记上调/下调
    # 阈值可以自己调：比如 FC > 2 (log2 > 1) 且 P < 0.01
    result['diff'] = 'normal'
    result.loc[(result['logfoldchanges'] > 1) & (result['pvals_adj'] < 0.05), 'diff'] = 'up'
    result.loc[(result['logfoldchanges'] < -1) & (result['pvals_adj'] < 0.05), 'diff'] = 'down'

    # 4. 绘图
    plt.figure(figsize=(8, 6)) # 图画大一点
    
    # 画背景点 (灰色)
    sns.scatterplot(data=result[result['diff']=='normal'], x='logfoldchanges', y='log10p', 
                    color='lightgrey', s=10, alpha=0.5, edgecolor=None)
    
    # 画显著点 (红色/蓝色)
    # up: 代表在 group_test 中高表达
    # down: 代表在 group_reference 中高表达
    sns.scatterplot(data=result[result['diff']!='normal'], x='logfoldchanges', y='log10p', 
                    hue='diff', palette={'up': '#d62728', 'down': '#1f77b4'},
                    s=30, alpha=0.8, edgecolor=None)

    # 5. 标注最显著的 Top 基因名字 (让图看起来更高级)
    # 挑选最显著的前 5 个上调基因和前 5 个下调基因
    top_up = result[result['diff']=='up'].sort_values('log10p', ascending=False).head(5)
    top_down = result[result['diff']=='down'].sort_values('log10p', ascending=False).head(5)
    
    import adjustText # 如果没有这个库，可以用 plt.text 简单代替
    texts = []
    for idx, row in top_up.iterrows():
        texts.append(plt.text(row['logfoldchanges'], row['log10p'], row['names'], fontsize=9, color='darkred'))
    for idx, row in top_down.iterrows():
        texts.append(plt.text(row['logfoldchanges'], row['log10p'], row['names'], fontsize=9, color='darkblue'))
    
    # 简单防重叠 (如果没有 adjustText 库，这段可能会报错，可以注释掉)
    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    except:
        pass # 如果没装 adjustText 就算了，不标或者重叠也行

    # 6. 装饰
    plt.title(f'Volcano: {group_test} (Red) vs {group_reference} (Blue)')
    plt.xlabel(f'log2 Fold Change\n(>0: High in {group_test}, <0: High in {group_reference})')
    plt.ylabel('-log10(adj. p-value)')
    plt.axvline(1, ls='--', c='k', lw=0.5); plt.axvline(-1, ls='--', c='k', lw=0.5)
    plt.axhline(1.3, ls='--', c='k', lw=0.5) # p=0.05
    
    plt.xlim(-5, 5)

    # 顺便让 Y 轴自动适应当前视野内的数据高度，不然图会很空
    # 这一步是为了配合缩短后的 X 轴，让图看起来更饱满
    current_view_genes = result[
        (result['logfoldchanges'] >= -5) & 
        (result['logfoldchanges'] <= 5)
    ]
    if len(current_view_genes) > 0:
        max_y_in_view = np.max(current_view_genes['log10p'].replace([np.inf, -np.inf], 0))
        plt.ylim(0, max_y_in_view * 1.1)
    

    # 保存
    save_name = f'Volcano_{group_test}_vs_{group_reference}.png'.replace(' ', '_')
    plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     -> Saved to {save_name}")


from scipy.optimize import linear_sum_assignment

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

def plot_custom_split_violin(adata, group_test, group_reference, genes, output_dir):
    """
    自定义分裂小提琴图：只对比两个特定的 Cluster (例如 Cluster 0 vs Cluster 3)
    左边：group_test (蓝色)
    右边：group_reference (橙色)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc
    import os

    # 1. 检查基因
    valid_genes = [g for g in genes if g in adata.var_names]
    if len(valid_genes) == 0:
        print(f"  !! No valid genes found for split violin ({group_test} vs {group_reference})")
        return

    print(f"  -> Plotting Split Violin: {group_test} vs {group_reference}...")

    # 2. 关键步骤：只提取这两个 Cluster 的数据
    # 我们创建一个 mask，只保留这两个组的细胞
    subset_mask = adata.obs['GREET_Cluster'].isin([group_test, group_reference])
    subset_adata = adata[subset_mask].copy()

    # 3. 准备绘图数据
    # 获取表达量矩阵
    obs_df = sc.get.obs_df(subset_adata, keys=valid_genes + ['GREET_Cluster'])
    
    # 将数据转换为长格式 (Long format) 以适配 Seaborn
    df_melted = obs_df.melt(id_vars=['GREET_Cluster'], 
                            value_vars=valid_genes,
                            var_name='Gene', 
                            value_name='Expression')

    # 4. 绘图
    plt.figure(figsize=(max(6, len(valid_genes)*1.2), 6))
    
    # 定义颜色：为了和火山图呼应，我们可以用 蓝 vs 红，或者默认的 蓝 vs 橙
    # 这里我们用自定义调色板，确保颜色固定
    my_palette = {group_test: '#d62728', group_reference: '#1f77b4'} # 红 vs 蓝 (对应火山图的上/下)
    
    # hue_order 确保 group_test (红) 在左边还是右边固定
    ax = sns.violinplot(data=df_melted, x='Gene', y='Expression', hue='GREET_Cluster',
                        hue_order=[group_test, group_reference], # 指定顺序
                        split=True, inner='quart',
                        palette=my_palette,
                        cut=0) # cut=0 不外推数据

    # (可选) 画散点 - 如果细胞太多就采样，防止画图太慢
    if df_melted.shape[0] > 3000:
        plot_data = df_melted.sample(n=3000, random_state=1)
    else:
        plot_data = df_melted
        
    sns.stripplot(data=plot_data, x='Gene', y='Expression', hue='GREET_Cluster',
                  hue_order=[group_test, group_reference],
                  dodge=True, jitter=True, color='black', size=2, alpha=0.3,
                  ax=ax, legend=False)

    # 5. 装饰
    plt.title(f'Split Violin: {group_test} (Red) vs {group_reference} (Blue)')
    plt.xlabel('Marker Genes')
    plt.ylabel('Log1p Expression')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Cluster')
    plt.tight_layout()

    # 保存
    save_name = f"SplitViolin_{group_test}_vs_{group_reference}.png".replace(' ', '_')
    plt.savefig(os.path.join(output_dir, save_name), dpi=300, bbox_inches='tight')
    plt.close()

    import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def _to_1d_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().flatten().cpu().numpy()
    return np.asarray(x).reshape(-1)
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_edge_similarity_histogram(
    embedding: torch.Tensor,     # (N, d) on GPU/CPU
    edges: torch.Tensor,         # (2, E) on same device as embedding
    weights_lp: torch.Tensor = None,  # (E,)
    weights_hp: torch.Tensor = None,  # (E,)
    save_path: str = "./results/edge_sim_hist.png",
    bins: int = 60,
    max_edges: int = 200000,
    seed: int = 0,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    E = edges.shape[1]
    if E > max_edges:
        g = torch.Generator(device=edges.device)
        g.manual_seed(seed)
        idx = torch.randperm(E, generator=g, device=edges.device)[:max_edges]
        e = edges[:, idx]
        wlp = weights_lp[idx] if weights_lp is not None else None
        whp = weights_hp[idx] if weights_hp is not None else None
    else:
        e, wlp, whp = edges, weights_lp, weights_hp

    u, v = e[0], e[1]
    sim_all = F.cosine_similarity(embedding[u], embedding[v], dim=1).detach().cpu().numpy()

    # 可选：随机负例分布（对照）
    ru = torch.randint(0, embedding.shape[0], (u.numel(),), device=embedding.device)
    rv = torch.randint(0, embedding.shape[0], (u.numel(),), device=embedding.device)
    sim_rand = F.cosine_similarity(embedding[ru], embedding[rv], dim=1).detach().cpu().numpy()

    plt.figure()
    plt.hist(sim_all, bins=bins, alpha=0.6, label="edges")
    plt.hist(sim_rand, bins=bins, alpha=0.6, label="random pairs")

    # 可选：按 LP/HP 权重拆分
    if wlp is not None:
        m = (wlp > 0.5).detach().cpu().numpy()
        if m.any():
            plt.hist(sim_all[m], bins=bins, alpha=0.6, label="LP edges (w_lp>0.5)")
    if whp is not None:
        m = (whp > 0.5).detach().cpu().numpy()
        if m.any():
            plt.hist(sim_all[m], bins=bins, alpha=0.6, label="HP edges (w_hp>0.5)")

    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_polynomial_filter_shapes(
    k: int,
    alpha_hp: float,
    coef_logits: torch.Tensor = None,   # cl_model.coef_logits
    save_path: str = "./results/poly_filter.png",
    num_points: int = 400,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lam = np.linspace(-1.0, 1.0, num_points)  # 归一化邻接特征值常见在 [-1,1]
    g_lp = lam ** k
    g_hp = (1.0 - alpha_hp * lam) ** k

    plt.figure()
    plt.plot(lam, g_lp, label=f"LP: λ^{k}")
    plt.plot(lam, g_hp, label=f"HP: (1-α·λ)^{k}, α={alpha_hp}")

    if coef_logits is not None:
        coef = torch.softmax(coef_logits.detach().cpu(), dim=0).numpy()  # :contentReference[oaicite:13]{index=13}
        g_mix = coef[0] * g_lp + coef[1] * g_hp
        plt.plot(lam, g_mix, label=f"mix: {coef[0]:.2f}·LP + {coef[1]:.2f}·HP")

    plt.xlabel("eigenvalue λ (normalized adjacency)")
    plt.ylabel("filter response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()






# def louvain_clustering(embedding, n_clusters, n_neighbors=12, max_steps=20):
#     """
#     带二分查找的 Louvain 聚类：尝试寻找最佳 resolution 以匹配 n_clusters
#     """
#     if torch.is_tensor(embedding):
#         embedding = embedding.detach().cpu().numpy()
        
#     # 1. 构建 AnnData 和邻居图
#     adata = anndata.AnnData(X=embedding)
#     sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    
#     # 2. 二分查找最佳 Resolution
#     # Louvain 的 resolution 越大，簇越多；越小，簇越少。
#     res_min = 0.1
#     res_max = 3.0
#     res_curr = 1.0
    
#     best_labels = None
#     best_diff = float('inf')
    
#     print(f"  [Louvain] Searching resolution for k={n_clusters}...", end='', flush=True)
    
#     for step in range(max_steps):
#         # 运行 Louvain
#         try:
#             sc.tl.louvain(adata, resolution=res_curr, key_added='louvain')
#         except ImportError:
#             # 如果没装 louvain 包，尝试用 leiden (它是 louvain 的升级版，用法一样)
#             print("\n  !! Louvain not installed. Trying Leiden instead...")
#             sc.tl.leiden(adata, resolution=res_curr, key_added='louvain')
            
#         labels = adata.obs['louvain'].astype(int).values
#         n_found = len(np.unique(labels))
        
#         # 记录最接近的一次
#         diff = abs(n_found - n_clusters)
#         if diff < best_diff:
#             best_diff = diff
#             best_labels = labels
            
#         # 完美匹配
#         if n_found == n_clusters:
#             print(f" Found! (res={res_curr:.3f})")
#             return labels
        
#         # 调整二分查找范围
#         if n_found > n_clusters:
#             res_max = res_curr
#             res_curr = (res_min + res_curr) / 2
#         else:
#             res_min = res_curr
#             res_curr = (res_max + res_curr) / 2
            
#     print(f" Done. (Best match: {len(np.unique(best_labels))} clusters, res={res_curr:.3f})")
#     return best_labels


# def visualize_scanpy_style(features, spatial, true_labels, pred_labels, ari, save_name_prefix):
#     """
#     使用 Scanpy 风格进行可视化：
#     1. 空间分布图 (左: GT, 右: 预测)
#     2. UMAP 投影图 (左: GT, 右: 预测)
#     """
#     # 1. 数据封装为 AnnData
#     if torch.is_tensor(features): features = features.detach().cpu().numpy()
#     if torch.is_tensor(true_labels): true_labels = true_labels.detach().cpu().numpy()
#     if isinstance(spatial, torch.Tensor): spatial = spatial.detach().cpu().numpy()
    
#     # 创建 AnnData 对象
#     adata = anndata.AnnData(X=features)
#     adata.obsm['spatial'] = spatial
    
#     # 存储标签 (转为字符串以保证是离散颜色)
#     # 处理 -1 (Unknown) 的情况，避免画图报错
#     true_labels_str = true_labels.astype(str)
#     pred_labels_str = pred_labels.astype(str)
    
#     adata.obs['Manual Annotation'] = pd.Categorical(true_labels_str)
#     adata.obs['Clustering'] = pd.Categorical(pred_labels_str)
    
#     # 2. 计算 UMAP (参考你的代码: metric='cosine')
#     # print("  [Vis] Computing UMAP...")
#     sc.pp.neighbors(adata, use_rep='X', metric='cosine')
#     sc.tl.umap(adata)

#     # -------------------------------------------------------
#     # 图 1: Spatial Plot (空间分布)
#     # -------------------------------------------------------
#     fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
    
#     # 如果点太多，减小点的大小 (spot_size)
#     spot_size = 30 if adata.n_obs > 5000 else 100
    
#     # 左图：真实标签
#     sc.pl.spatial(adata, color='Manual Annotation', ax=axes1[0], show=False, 
#                   spot_size=spot_size, title='Manual Annotation', frameon=False)
    
#     # 右图：预测标签 (标题带 ARI)
#     sc.pl.spatial(adata, color='Clustering', ax=axes1[1], show=False, 
#                   spot_size=spot_size, title=f'Clustering (ARI={ari:.4f})', frameon=False)
    
#     plt.tight_layout()
#     plt.savefig(f"{save_name_prefix}_Spatial.png", dpi=300)
#     plt.close()

#     # -------------------------------------------------------
#     # 图 2: UMAP Plot (流形投影)
#     # -------------------------------------------------------
#     fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    
#     # 左图：真实标签
#     sc.pl.umap(adata, color='Manual Annotation', ax=axes2[0], show=False, 
#                title='Manual Annotation', frameon=False)
    
#     # 右图：预测标签
#     sc.pl.umap(adata, color='Clustering', ax=axes2[1], show=False, 
#                title='Clustering', frameon=False)
    
#     # 设置长宽比 (参考你的代码)
#     for ax in axes2:
#         ax.set_aspect(1)
        
#     plt.tight_layout()
#     plt.savefig(f"{save_name_prefix}_UMAP.png", dpi=300)
#     plt.close()
    
    # print(f"  [Vis] Saved plots to {save_name_prefix}_*.png")


# def run_deg_analysis_151673(pred_labels, output_dir='./results/151673_analysis'):

    
#     # 尝试导入您的自定义绘图函数
#     try:
#         from utils import plot_deg_visualizations, plot_custom_volcano, plot_custom_split_violin, plot_spatial
#     except ImportError:
#         pass 

#     print("\n[Analysis] Starting DEG/SVG Analysis for 151673...")
    
#     # ==============================================================================
#     # 1. 读取数据 (带图片!)
#     # ==============================================================================
#     data_root = './data/151673'
#     if not os.path.exists(os.path.join(data_root, 'spatial')):
#         print(f"!! Error: 'spatial' folder not found in {data_root}.")
#         return

#     print(f"  -> Loading Visium data (matrix + images) from {data_root}...")
#     try:
#         # 这一步非常关键，它会把图片读到 adata.uns['spatial'] 里
#         adata = sc.read_visium(path=data_root, 
#                                count_file='filtered_feature_bc_matrix.h5',
#                                load_images=True) 
#     except Exception as e:
#         print(f"!! Error loading Visium data: {e}")
#         return
        
#     adata.var_names_make_unique()
#     if 'in_tissue' in adata.obs:
#         adata = adata[adata.obs['in_tissue'] == 1].copy()
    
#     # 2. 预处理
#     sc.pp.normalize_total(adata, target_sum=1e6)
#     sc.pp.log1p(adata)

#     # 3. 注入标签
#     if torch.is_tensor(pred_labels):
#         pred_labels = pred_labels.detach().cpu().numpy()
        
#     if len(pred_labels) != adata.n_obs:
#         print(f"!! Warning: Length mismatch! Labels({len(pred_labels)}) vs Data({adata.n_obs})")
    
#     adata.obs['GREET_Cluster'] = pd.Categorical([f"Cluster {i}" for i in pred_labels])
#     available_clusters = adata.obs['GREET_Cluster'].unique().tolist()

#     # 4. 计算差异基因 (略过详细代码，保持原样即可)
#     print("  -> Calculating marker genes...")
#     # ... (此处省略 rank_genes_groups 代码，您保留原来的即可) ...
#     # 为了完整性，简单写一下：
#     sc.tl.rank_genes_groups(adata, 'GREET_Cluster', method='wilcoxon')
#     try:
#         plot_deg_visualizations(adata, output_dir)
#     except: pass

#     # ... (此处省略 Volcano/Violin 的对比代码，您保留原来的即可) ...

#     # 5. 保存 Top Markers
#     if not os.path.exists(output_dir): os.makedirs(output_dir)
#     pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(10).to_csv(os.path.join(output_dir, '151673_Top_Markers.csv'))
    
#     # 获取图片数据 (如果 sc.read_visium 成功，这里一定有数据)
#     spatial_uns = adata.uns.get('spatial', None)
    
#     if spatial_uns is None:
#         print("!! Warning: No image data found in adata.uns['spatial']. Plot will be unaligned.")
    
#     # 调用您的自定义函数
#     # 注意：我们将 true_labels 也设为 pred_labels，这样左右两张图都是预测结果，方便看背景对齐
#     my_xlim = (4000, 12000) 
#     my_ylim = (7000, 14000)

#     try:
#         plot_spatial(
#             spatial=adata.obsm['spatial'],
#             pred_labels=pred_labels,
#             true_labels=pred_labels,
#             title="GREET Clusters (Cropped)",
#             save_name=os.path.join(output_dir, '151673_Domain_Map_Cropped.png'),
#             spot_size=300,           # 放大后，点可以调大一点让它更明显
#             spatial_uns=spatial_uns,
#             img_key="hires",
#             xlim=my_xlim,
#             ylim=my_ylim
#         )
#         print(f"  -> Saved cropped map.")
#     except Exception as e:
#         print(f"!! plot_spatial failed: {e}")
#         # 如果 plot_spatial 失败，用 sc.pl.spatial 兜底
#         plt.figure(figsize=(8,8))
#         sc.pl.spatial(adata, color='GREET_Cluster', img_key='hires', show=False)
#         plt.savefig(os.path.join(output_dir, '151673_Domain_Map_Backup.png'))
#         plt.close()

#     print(f"[Done] Results saved to {output_dir}")



# def plot_one_split_violin(adata, target_cluster, genes, output_dir):
#     """
#     绘制 Cluster X vs Rest 的分裂小提琴图 (Split Violin Plot)
#     """
#     # 1. 检查基因是否存在
#     valid_genes = [g for g in genes if g in adata.var_names]
#     if len(valid_genes) == 0:
#         return

#     # 2. 创建临时分组列： "Target" vs "Rest"
#     # 我们不修改原始 adata，而是创建一个临时的 DataFrame
#     obs_df = sc.get.obs_df(adata, keys=valid_genes + ['GREET_Cluster'])
    
#     # 定义 hue 列：如果是目标簇就是 'Target'，否则是 'Rest'
#     obs_df['Group'] = obs_df['GREET_Cluster'].apply(
#         lambda x: target_cluster if x == target_cluster else 'Others'
#     )

#     # 3. 数据重塑为长格式 (Melt) 以适配 Seaborn
#     # 格式变成: [Cell_ID, Gene_Name, Expression, Group]
#     df_melted = obs_df.melt(id_vars=['GREET_Cluster', 'Group'], 
#                             value_vars=valid_genes,
#                             var_name='Gene', 
#                             value_name='Expression')

#     # 4. 绘图
#     plt.figure(figsize=(max(6, len(valid_genes)*1.2), 6)) # 动态调整宽度
    
#     # (A) 画分裂小提琴
#     # split=True 是核心，它让两个组共用一个小提琴
#     # inner='quart' 会画出四分位线（您图中的那些黑线）
#     ax = sns.violinplot(data=df_melted, x='Gene', y='Expression', hue='Group',
#                         split=True, inner='quart',
#                         palette={target_cluster: '#1f77b4', 'Others': '#ff7f0e'}, # 蓝 vs 橙
#                         cut=0) # cut=0 让小提琴在数据范围内截断，不外推

#     # (B) 画散点 (Stripplot) - 模拟您图中的黑点
#     # 注意：如果细胞太多，画散点会很慢且糊成一团。我们做一个降采样保护。
#     if df_melted.shape[0] > 5000:
#         # 如果点太多，只随机画 3000 个点示意一下
#         plot_data = df_melted.sample(n=3000, random_state=1)
#     else:
#         plot_data = df_melted
        
#     sns.stripplot(data=plot_data, x='Gene', y='Expression', hue='Group',
#                   dodge=True, # 让点也分开在两边
#                   jitter=True, # 抖动
#                   color='black', # 黑色点
#                   size=2, # 点的大小
#                   alpha=0.3, # 透明度
#                   ax=ax, legend=False) # 不显示散点的图例

#     # 美化
#     plt.title(f'Split Violin: {target_cluster} vs Rest')
#     plt.xlabel('Marker Genes')
#     plt.ylabel('Log1p Expression')
#     plt.xticks(rotation=45)
    
#     # 调整图例位置
#     plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.tight_layout()
    
#     # 保存
#     safe_name = target_cluster.replace(' ', '_')
#     save_path = f"{output_dir}/DEG_SplitViolin_{safe_name}.png"
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"     -> Saved Split Violin for {target_cluster}")
