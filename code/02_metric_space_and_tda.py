import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from scipy.spatial.distance import pdist, squareform

SEED = 42


def frob_unit(M: np.ndarray) -> np.ndarray:
    fn = np.linalg.norm(M, ord="fro")
    return M / fn if fn > 0 else M


def build_structured_metric_space(df: pd.DataFrame, seed: int = SEED):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    B_icon = ohe.fit_transform(df[["iconography_class"]])
    B_mat = ohe.fit_transform(df[["material_clean"]])
    B_reg = ohe.fit_transform(df[["macro_region"]])

    z_yr = StandardScaler().fit_transform(df[["year_ce_imputed"]])
    flag_dated = df[["is_dated"]].astype(float).to_numpy()
    B_chr = np.hstack([z_yr, flag_dated])

    X_raw = np.hstack([
        0.40 * frob_unit(B_icon),
        0.25 * frob_unit(B_mat),
        0.20 * frob_unit(B_reg),
        0.15 * frob_unit(B_chr),
    ])
    Z = normalize(X_raw, norm="l2")

    rng = np.random.default_rng(seed)
    D = squareform(pdist(Z, metric="euclidean"))
    E = rng.normal(0.0, 1e-8, size=D.shape)
    E_sym = 0.5 * (E + E.T)
    np.fill_diagonal(E_sym, 0.0)
    D = np.clip(D + E_sym, 0.0, None)
    np.fill_diagonal(D, 0.0)
    return Z, D


def betti_curve(intervals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(grid), dtype=int)
    for b, d in intervals:
        death = grid[-1] if np.isinf(d) else d
        counts += ((grid >= b) & (grid < death)).astype(int)
    return counts


def run_tda(proc_dir="../data/processed_data", tables_dir="../tables", figures_dir="../figures"):
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(proc_dir, "structured_corpus_audit.csv"))
    Z, D = build_structured_metric_space(df, seed=SEED)
    np.save(os.path.join(proc_dir, "pairwise_distance_matrix.npy"), D)

    rip = ripser(D, distance_matrix=True, maxdim=1, do_cocycles=True)
    dgms = rip["dgms"]
    cocycles_h1 = rip["cocycles"][1]
    h0, h1 = dgms[0], dgms[1]

    pd.DataFrame(h0, columns=["birth", "death"]).to_csv(os.path.join(proc_dir, "persistence_h0.csv"), index=False)
    pd.DataFrame(h1, columns=["birth", "death"]).to_csv(os.path.join(proc_dir, "persistence_h1.csv"), index=False)

    valid_mask = np.isfinite(h1[:, 1])
    valid_h1 = h1[valid_mask]
    valid_cocycles = [cocycles_h1[i] for i, ok in enumerate(valid_mask) if ok]
    h1_pers = valid_h1[:, 1] - valid_h1[:, 0]
    tau_90 = float(np.percentile(h1_pers, 90))

    order = np.argsort(-h1_pers)
    t2_rows = []
    for rank, idx in enumerate(order, start=1):
        coc = valid_cocycles[idx]
        involved_nodes = sorted(set(int(x) for edge in coc for x in edge[:2])) if len(coc) > 0 else []
        members = [
            f"{df.iloc[nid]['partition']}:{df.iloc[nid]['artifact_id']} ({df.iloc[nid]['title'][:28]})"
            for nid in involved_nodes[:12]
        ]
        t2_rows.append({
            "Loop_ID": f"H1-Loop_{rank}",
            "Birth": round(float(valid_h1[idx, 0]), 4),
            "Death": round(float(valid_h1[idx, 1]), 4),
            "Persistence": round(float(h1_pers[idx]), 4),
            "Is_Prominent_Top10pct": bool(h1_pers[idx] >= tau_90),
            "Involved_Nodes": len(involved_nodes),
            "Cocycle_Members": " | ".join(members),
        })

    df_t2 = pd.DataFrame(t2_rows)
    df_t2[df_t2["Is_Prominent_Top10pct"]].to_csv(os.path.join(tables_dir, "Table2_H1_Loops.csv"), index=False)

    parts = df["partition"].to_numpy()
    is_dha = (parts == "DHARMA")
    is_mus = np.isin(parts, ["VA", "CMA"])
    d_min_cross = float(D[np.ix_(is_dha, is_mus)].min())
    eps_bridge = round(d_min_cross + 0.05, 4)

    # Figure 1: Persistence Diagram
    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_axes([0.12, 0.14, 0.80, 0.77])
    plot_diagrams(dgms, ax=ax, show=False)
    ax.set_title(f"Cross-Cultural Persistence Diagram (N = {len(df)}, d = {Z.shape[1]})", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Proximity Scale Birth ($\\epsilon$)", fontsize=10)
    ax.set_ylabel("Persistence Scale Death ($\\epsilon$)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(figures_dir, "Fig1_Persistence_Profile.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: Betti Curves
    grid_s = np.linspace(0.0, float(np.percentile(D, 98)), 400)
    b0_c = betti_curve(h0, grid_s)
    b1_c = betti_curve(h1, grid_s)

    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_axes([0.10, 0.14, 0.82, 0.77])
    ax.plot(grid_s, b0_c, label="Connected Components ($H_0$)", color="#1f618d", lw=2.4)
    ax.plot(grid_s, b1_c, label="Transmission Loops ($H_1$)", color="#c0392b", lw=2.4)
    ax.axvline(eps_bridge, color="#27ae60", linestyle=":", lw=1.8, label=f"Bridging Horizon ($\\epsilon_{{bridge}} = {eps_bridge:.4f}$)")
    ax.set_title(f"Multi-Scale Cultural Integration Profile (N = {len(df)})", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Proximity Scale ($\\epsilon$)", fontsize=10)
    ax.set_ylabel("Betti Count", fontsize=10)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#bdc3c7")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(figures_dir, "Fig2_Transmission_Profile.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[02_metric_space_and_tda] d={Z.shape[1]}, D_min={d_min_cross:.4f}, H0={len(h0)}, H1={len(valid_h1)}, tau_90%={tau_90:.4f}")


if __name__ == "__main__":
    run_tda()
