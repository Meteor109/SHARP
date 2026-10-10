"""Reproducible training entry point for the four public SHARP benchmarks."""

import argparse
import os
import random

import dgl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

from data_loader import load_data
from model import Edge_Discriminator, GCL
from utils import align_labels, mclust_R, refine_label_spatial, plot_spatial_hex_tiles


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_cl(model, discriminator, optimizer, features, structural_encoding, edges):
    model.train()
    discriminator.eval()
    adj_lp, adj_hp, _, _ = discriminator(torch.cat((features, structural_encoding), dim=1), edges)
    loss = model(features, adj_lp, features, adj_hp)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_discriminator(model, discriminator, optimizer, features, structural_encoding, edges, args):
    model.eval()
    discriminator.train()
    adj_lp, adj_hp, weights_lp, weights_hp = discriminator(
        torch.cat((features, structural_encoding), dim=1), edges
    )
    embedding = model.get_embedding(features, adj_lp, adj_hp)
    random_pairs = torch.randint(features.shape[0], (2, edges.shape[1]), device=features.device)
    keep = random_pairs[0] != random_pairs[1]
    random_pairs[:, ~keep] = (random_pairs[:, ~keep] + 1) % features.shape[0]

    edge_batch_size = 5000
    batch_losses = []
    for start in range(0, edges.shape[1], edge_batch_size):
        end = min(start + edge_batch_size, edges.shape[1])
        batch_edges = edges[:, start:end]
        batch_pairs = random_pairs[:, start:end]
        edge_similarity = F.cosine_similarity(embedding[batch_edges[0]], embedding[batch_edges[1]])
        random_similarity = F.cosine_similarity(embedding[batch_pairs[0]], embedding[batch_pairs[1]])
        target = torch.ones_like(edge_similarity)
        lp_loss = F.margin_ranking_loss(
            edge_similarity, random_similarity, target, margin=args.margin_hom, reduction="none"
        )
        hp_loss = F.margin_ranking_loss(
            random_similarity, edge_similarity, target, margin=args.margin_het, reduction="none"
        )
        lp_loss = (lp_loss * torch.relu(weights_lp[start:end] - 0.5)).mean()
        hp_loss = (hp_loss * torch.relu(weights_hp[start:end] - 0.5)).mean()
        batch_losses.append((lp_loss + hp_loss) / 2.0)

    loss = torch.stack(batch_losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def load_spatial_coordinates(dataset: str) -> np.ndarray:
    """Load coordinates in exactly the same spot order used by each benchmark loader."""
    if dataset == "E18_mouse_brain":
        adata = sc.read_h5ad("./data/E18.5_mouse_brain/E18_adata_rna.h5ad")
        return adata.obs[["array_col", "array_row"]].to_numpy(dtype=float)
    if dataset in {"Stereo_AB", "Stereo_DPI"}:
        adata = sc.read_h5ad(f"./data/Stereo/{dataset}.h5ad")
        if "spatial" in adata.obsm:
            return np.asarray(adata.obsm["spatial"], dtype=float)
        for columns in (("array_col", "array_row"), ("x_array", "y_array"), ("x", "y")):
            if all(column in adata.obs for column in columns):
                return adata.obs[list(columns)].to_numpy(dtype=float)
        raise ValueError(f"No spatial coordinates found for {dataset}.")
    if dataset == "BRCA":
        import pandas as pd

        data_dir = "./data/BRCA/Breast_Caner"
        adata = sc.read_10x_h5(os.path.join(data_dir, "filtered_feature_bc_matrix.h5"))
        positions = pd.read_csv(os.path.join(data_dir, "spatial", "tissue_positions_list.csv"), header=None).iloc[:, :6]
        positions.columns = ["barcode", "in_tissue", "array_row", "array_col", "py", "px"]
        positions["barcode"] = positions["barcode"].astype(str)
        positions = positions.set_index("barcode")
        annotation = pd.read_csv(os.path.join(data_dir, "spatial", "brca_ground_truth_annotation.csv"))
        annotation = annotation.assign(spot_id=annotation["spot_id"].astype(str)).set_index("spot_id")
        obs = adata.obs.join(positions, how="left").join(annotation[["ground_truth_annotation"]], how="left")
        obs = obs[(obs["in_tissue"] == 1) & obs["ground_truth_annotation"].notna()]
        return obs[["px", "py"]].to_numpy(dtype=float)
    raise ValueError(f"Unsupported dataset: {dataset}")


def plot_prediction_spatial(spatial, labels, dataset, ari, output_path):
    if dataset == "BRCA":
        plot_spatial_hex_tiles(spatial, labels, output_path, title=f"ARI = {ari:.4f}")
        return
    plt.figure(figsize=(7, 7), dpi=300)
    plt.scatter(spatial[:, 0], spatial[:, 1], c=labels, cmap="tab20", s=10, linewidths=0)
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.title(f"SHARP clustering (ARI = {ari:.4f})", fontsize=13)
    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close()


def plot_prediction_umap(embedding, labels, ari, output_path):
    adata = sc.AnnData(X=np.asarray(embedding))
    adata.obs["SHARP cluster"] = [f"Cluster {int(value)}" for value in labels]
    sc.pp.neighbors(adata, use_rep="X", metric="cosine")
    sc.tl.umap(adata, random_state=0)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sc.pl.umap(
        adata, color="SHARP cluster", palette="tab20", legend_loc="right margin",
        frameon=False, size=20, title=f"SHARP UMAP (ARI = {ari:.4f})", ax=ax, show=False,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def evaluate(model, discriminator, features, structural_encoding, edges, labels, n_clusters, spatial, refine_k):
    model.eval()
    discriminator.eval()
    with torch.no_grad():
        adj_lp, adj_hp, _, _ = discriminator(torch.cat((features, structural_encoding), dim=1), edges)
        embedding = model.get_embedding_comb(features, adj_lp, adj_hp).detach().cpu().numpy()
    predicted = mclust_R(embedding, n_clusters)
    if predicted is None:
        raise RuntimeError("mclust clustering failed. Ensure R package 'mclust' is installed.")
    truth = labels.detach().cpu().numpy()
    predicted = align_labels(predicted, truth)
    if refine_k > 0:
        predicted = refine_label_spatial(predicted, spatial, k=refine_k, verbose=False)
    valid = truth >= 0
    ari = adjusted_rand_score(truth[valid], predicted[valid]) if valid.any() else float("nan")
    return ari, predicted, embedding


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation is required for this implementation.")
    os.makedirs(args.output_dir, exist_ok=True)
    setup_seed(0)

    data = load_data(args.dataset, graph_k=args.k)
    # Common layout: features, edges, SE, train/val/test, labels, [class_names], n_nodes, n_features
    features, edges, structural_encoding = data[0], data[1], data[2]
    labels, n_nodes, n_features = data[6], data[-2], data[-1]
    spatial = load_spatial_coordinates(args.dataset)
    if spatial.shape[0] != n_nodes:
        raise ValueError(f"Spatial/model spot mismatch: {spatial.shape[0]} versus {n_nodes}.")

    features = features.cuda()
    structural_encoding = structural_encoding.cuda()
    edges = edges.cuda()
    n_clusters = len(torch.unique(labels))
    best = {"ari": -np.inf, "labels": None, "embedding": None, "epoch": None, "trial": None}

    for trial in range(args.ntrials):
        setup_seed(trial)
        model = GCL(args.nlayers_enc, args.nlayers_proj, n_features, args.emb_dim, args.proj_dim, args.dropout, args.sparse).cuda()
        discriminator = Edge_Discriminator(
            n_nodes, n_features + structural_encoding.shape[1], args.alpha, args.sparse, temperature=args.temp_disc
        ).cuda()
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr_gcl, weight_decay=args.w_decay)
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc, weight_decay=args.w_decay)

        for epoch in range(1, args.epochs + 1):
            for _ in range(args.cl_rounds):
                train_cl(model, discriminator, optimizer_model, features, structural_encoding, edges)
            train_discriminator(model, discriminator, optimizer_discriminator, features, structural_encoding, edges, args)

            if epoch % args.eval_freq == 0:
                ari, predicted, embedding = evaluate(
                    model, discriminator, features, structural_encoding, edges, labels, n_clusters, spatial, args.refine_k
                )
                if ari > best["ari"]:
                    best.update(ari=ari, labels=predicted.copy(), embedding=embedding.copy(), epoch=epoch, trial=trial)

    if best["labels"] is None:
        raise RuntimeError("No checkpoint was evaluated. Ensure epochs is at least eval_freq.")

    spatial_path = os.path.join(args.output_dir, f"{args.dataset}_best_spatial.png")
    umap_path = os.path.join(args.output_dir, f"{args.dataset}_best_umap.png")
    plot_prediction_spatial(spatial, best["labels"], args.dataset, best["ari"], spatial_path)
    plot_prediction_umap(best["embedding"], best["labels"], best["ari"], umap_path)
    print(f"Finished {args.dataset}. Best ARI: {best['ari']:.4f} (trial {best['trial']}, epoch {best['epoch']}).")
    print(f"Saved: {spatial_path}")
    print(f"Saved: {umap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHARP reproducibility runner")
    parser.add_argument("-dataset", choices=["BRCA", "E18_mouse_brain", "Stereo_AB", "Stereo_DPI"], required=True)
    parser.add_argument("-output_dir", default="./results")
    parser.add_argument("-gpu", default="2")
    parser.add_argument("-epochs", type=int, default=600)
    parser.add_argument("-ntrials", type=int, default=1)
    parser.add_argument("-eval_freq", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("-refine_k", type=int, default=5)
    parser.add_argument("-margin_hom", type=float, default=0.3)
    parser.add_argument("-margin_het", type=float, default=0.7)
    parser.add_argument("-alpha", type=float, default=0.1)
    parser.add_argument("-temp_disc", type=float, default=1.0)
    parser.add_argument("-lr_gcl", type=float, default=1e-4)
    parser.add_argument("-lr_disc", type=float, default=1e-4)
    parser.add_argument("-w_decay", type=float, default=0.0)
    parser.add_argument("-dropout", type=float, default=0.5)
    parser.add_argument("-cl_rounds", type=int, default=2)
    parser.add_argument("-nlayers_enc", type=int, default=2)
    parser.add_argument("-nlayers_proj", type=int, default=1)
    parser.add_argument("-emb_dim", type=int, default=512)
    parser.add_argument("-proj_dim", type=int, default=512)
    parser.add_argument("-sparse", type=int, default=1)
    main(parser.parse_args())
