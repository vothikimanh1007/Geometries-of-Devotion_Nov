import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

SEED = 42


def run_network_and_ablations(proc_dir="../data/processed_data", tables_dir="../tables", figures_dir="../figures"):
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    df = pd.read_csv(os.path.join(proc_dir, "structured_corpus_audit.csv"))
    D = np.load(os.path.join(proc_dir, "pairwise_distance_matrix.npy"))

    parts = df["partition"].to_numpy()
    is_dha = (parts == "DHARMA")
    is_va = (parts == "VA")
    is_cma = (parts == "CMA")
    is_mus = (is_va | is_cma)

    d_min_cross = float(D[np.ix_(is_dha, is_mus)].min())
    eps_bridge = round(d_min_cross + 0.05, 4)
    eps_global = round(float(np.percentile(D[np.triu_indices(len(df), k=1)], 65)), 4)

    def eval_horizon(eps: float, label: str):
        n = len(df)
        adj = ((D <= eps) & (~np.eye(n, dtype=bool))).astype(int)
        G = nx.from_numpy_array(adj)
        deg = np.array([G.degree(i) for i in range(n)], dtype=float)
        cb = np.array(list(nx.betweenness_centrality(G, normalized=True).values()), dtype=float)
        cl = np.array(list(nx.clustering(G).values()), dtype=float)

        rows_c = []
        for p, pname in [("DHARMA", "DHARMA SEA Epigraphy"), ("VA", "V&A Portable Bronzes"), ("CMA", "CMA Asian Sculptures")]:
            m = (parts == p)
            boots = [rng.choice(cb[m], size=m.sum(), replace=True).mean() for _ in range(1000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows_c.append({
                "Horizon": label,
                "Epsilon": eps,
                "Edges": G.number_of_edges(),
                "Partition": pname,
                "Degree_Mean": round(float(deg[m].mean()), 2),
                "Degree_SD": round(float(deg[m].std()), 2),
                "CB_Mean": round(float(cb[m].mean()), 6),
                "CB_SD": round(float(cb[m].std()), 6),
                "CB_CI95_Low": round(float(lo), 6),
                "CB_CI95_High": round(float(hi), 6),
                "Clustering_Mean": round(float(cl[m].mean()), 4),
            })

        obs_gap = float(cb[is_mus].mean() - cb[is_dha].mean())
        n_bins = min(10, len(np.unique(deg)))
        deg_bins = pd.qcut(deg, q=n_bins, duplicates="drop").codes if n_bins > 1 else np.zeros(n, dtype=int)
        null_gaps = []
        for _ in range(1000):
            perm_cb = cb.copy()
            for b in np.unique(deg_bins):
                idx_b = np.where(deg_bins == b)[0]
                perm_cb[idx_b] = rng.permutation(perm_cb[idx_b])
            null_gaps.append(float(perm_cb[is_mus].mean() - perm_cb[is_dha].mean()))
        p_deg_null = float(np.mean(np.array(null_gaps) >= obs_gap))
        _, p_mwu = mannwhitneyu(cb[is_mus], cb[is_dha], alternative="two-sided")

        return pd.DataFrame(rows_c), cb, {
            "Test_Ablation": f"Degree_Null_{label}",
            "Scale_Epsilon": eps,
            "Nodes_N": n,
            "Observed": round(obs_gap, 6),
            "Null_Mean": round(float(np.mean(null_gaps)), 6),
            "Degree_Null_p": round(p_deg_null, 4),
            "MWU_p": f"{p_mwu:.4e}",
        }

    df_t3_b, cb_bridge, stat_b = eval_horizon(eps_bridge, "Bridging")
    df_t3_g, _, stat_g = eval_horizon(eps_global, "Global")
    pd.concat([df_t3_b, df_t3_g], ignore_index=True).to_csv(os.path.join(tables_dir, "Table3_Network_Centrality.csv"), index=False)

    regs = df["macro_region"].to_numpy()
    ablation_rows = [stat_b, stat_g]

    def run_geo_ablation(mask: np.ndarray, test_name: str):
        idx = np.where(mask)[0]
        D_s = D[np.ix_(idx, idx)]
        adj_s = ((D_s <= eps_bridge) & (~np.eye(len(idx), dtype=bool))).astype(int)
        G_s = nx.from_numpy_array(adj_s)
        cb_s = np.array(list(nx.betweenness_centrality(G_s, normalized=True).values()), dtype=float)
        dm, mm = is_dha[idx], is_mus[idx]
        dmin_s = float(D_s[np.ix_(dm, mm)].min())
        _, p_mwu_s = mannwhitneyu(cb_s[mm], cb_s[dm], alternative="two-sided")
        return {
            "Test_Ablation": test_name,
            "Scale_Epsilon": eps_bridge,
            "Nodes_N": len(idx),
            "Observed": round(float(cb_s[mm].mean() - cb_s[dm].mean()), 6),
            "Null_Mean": np.nan,
            "Degree_Null_p": np.nan,
            "MWU_p": f"{p_mwu_s:.4e} (D_min={dmin_s:.4f})",
        }

    ablation_rows.append(run_geo_ablation(is_dha | (is_mus & (regs == "Maritime_Mainland_SEA")), "Geo_Test_1_Museum_in_SEA"))
    ablation_rows.append(run_geo_ablation(is_mus | (is_dha & np.isin(regs, list(set(regs[is_mus])))), "Geo_Test_2_Shared_Regions"))

    perm_dmins = [float(D[np.ix_(m := (rng.permutation(parts) == "DHARMA"), ~m)].min()) for _ in range(1000)]
    p_perm_dmin = float(np.mean(np.array(perm_dmins) <= d_min_cross))
    ablation_rows.append({
        "Test_Ablation": "Label_Permutation_Null_Dmin",
        "Scale_Epsilon": eps_bridge,
        "Nodes_N": len(df),
        "Observed": round(d_min_cross, 4),
        "Null_Mean": round(float(np.mean(perm_dmins)), 4),
        "Degree_Null_p": round(p_perm_dmin, 4),
        "MWU_p": "D_min permutation check",
    })
    pd.DataFrame(ablation_rows).to_csv(os.path.join(tables_dir, "Table4_Ablation_Null_Tests.csv"), index=False)

    # Table 5: Closest Cross-Corpus Boundary Pairs and Controls
    iu, ju = np.triu_indices(len(df), k=1)
    cross_mask = ((parts[iu] == "DHARMA") & is_mus[ju]) | ((parts[ju] == "DHARMA") & is_mus[iu])
    c_iu, c_ju = iu[cross_mask], ju[cross_mask]
    d_idx = np.where(parts[c_iu] == "DHARMA", c_iu, c_ju)
    m_idx = np.where(parts[c_iu] == "DHARMA", c_ju, c_iu)

    df_cases = pd.DataFrame({
        "DHARMA_ID": df["artifact_id"].values[d_idx],
        "DHARMA_Title": df["title"].values[d_idx],
        "Museum": parts[m_idx],
        "Museum_ID": df["artifact_id"].values[m_idx],
        "Museum_Title": df["title"].values[m_idx],
        "Iconography_Class": df["iconography_class"].values[m_idx],
        "Distance_Dij": np.round(D[c_iu, c_ju], 4),
        "Museum_CB_Bridge": np.round(cb_bridge[m_idx], 6),
    }).sort_values(by=["Distance_Dij", "Museum_CB_Bridge", "DHARMA_ID", "Museum_ID"],
                   ascending=[True, False, True, True]).reset_index(drop=True)
    df_cases.drop_duplicates(subset=["Museum_ID"]).head(20).to_csv(os.path.join(tables_dir, "Table5_Case_Studies_And_Controls.csv"), index=False)

    # Figure 3: Illustrative Tripartite Sample (n=45)
    sel = np.concatenate([
        df[df["partition"] == "DHARMA"].index[:15],
        df[df["partition"] == "VA"].index[:15],
        df[df["partition"] == "CMA"].index[:15],
    ])
    sub_df = df.iloc[sel].reset_index(drop=True)
    sub_D = D[np.ix_(sel, sel)]

    G3 = nx.Graph()
    cmap = {"DHARMA": "#2980b9", "VA": "#d35400", "CMA": "#27ae60"}
    for idx, r in sub_df.iterrows():
        G3.add_node(f"[{idx+1}] {r['title'][:11]}", color=cmap.get(r["partition"], "#7f8c8d"))
    for i in range(len(sub_df)):
        for nb in np.argsort(sub_D[i])[1:4]:
            G3.add_edge(f"[{i+1}] {sub_df.iloc[i]['title'][:11]}", f"[{nb+1}] {sub_df.iloc[nb]['title'][:11]}")

    pos3 = nx.spring_layout(G3, seed=SEED, k=0.45)
    fig = plt.figure(figsize=(10.5, 9.0))
    ax = fig.add_axes([0.05, 0.06, 0.90, 0.86])
    nx.draw_networkx_nodes(G3, pos3, ax=ax, node_color=[G3.nodes[n]["color"] for n in G3.nodes()],
                           node_size=550, alpha=0.92, edgecolors="#2c3e50", linewidths=0.6)
    nx.draw_networkx_edges(G3, pos3, ax=ax, alpha=0.35, edge_color="#7f8c8d", width=1.3)
    nx.draw_networkx_labels(G3, pos3, ax=ax, font_size=7.5, font_family="sans-serif", font_weight="bold")
    ax.scatter([], [], c="#2980b9", label="Southeast Asian Epigraphy (DHARMA, n=15)", s=90, edgecolors="#2c3e50", linewidths=0.5)
    ax.scatter([], [], c="#d35400", label="Portable Devotional Bronzes (V&A London, n=15)", s=90, edgecolors="#2c3e50", linewidths=0.5)
    ax.scatter([], [], c="#27ae60", label="Himalayan/Gandharan Icons (Cleveland Museum, n=15)", s=90, edgecolors="#2c3e50", linewidths=0.5)
    ax.legend(scatterpoints=1, loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#bdc3c7", fontsize=9.5)
    ax.set_title("Tripartite Simplicial Network Sample (n = 45)", fontsize=11.5, fontweight="bold")
    ax.axis("off")
    plt.savefig(os.path.join(figures_dir, "Fig3_Simplicial_Network.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[03_centrality_ablations_and_cases] Completed full-graph metrics, 4 ablations, Table 5, and Fig 3.")


if __name__ == "__main__":
    run_network_and_ablations()
