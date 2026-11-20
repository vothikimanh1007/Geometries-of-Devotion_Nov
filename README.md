# Geometries of Devotion: Controlled Multi-Modal Topological Auditing of Asian Buddhist Epigraphy and Portable Icons

Reproducible computational repository accompanying the manuscript:
**Auditing Structural Relations Between Asian Buddhist Epigraphy and Portable Icons: A Controlled Multi-Modal Topological Analysis** (Thi Kim Anh Vo).

## Repository Structure

```text
Geometries-of-Devotion/
├── code/
│   ├── 01_ingest_and_audit_corpus.py        # Raw TEI-XML & JSON parsing + Table 1 audit
│   ├── 02_metric_space_and_tda.py           # d=16 Frobenius/L2 space, Ripser H0/H1, cocycles, Figs 1-2
│   ├── 03_centrality_ablations_and_cases.py # Full-graph centrality, 4 null/geo tests, Fig 3
│   ├── run_all.py                           # End-to-end deterministic runner (SEED=42)
│   └── requirements.txt                     # Python package dependencies
├── data/
│   ├── raw_data/                            # 308 DHARMA XMLs, 100 V&A JSONs, 100 CMA JSONs
│   └── processed_data/                      # Audited CSV, distance matrix, persistence CSVs
├── tables/                                  # Recomputed Tables 1-5 (CSV)
├── figures/                                 # High-resolution 300 DPI Figures 1-3 (PNG)
├── CITATION.cff
├── LICENSE
└── README.md
```

## Quick Reproduction

```bash
pip install -r code/requirements.txt
cd code
python run_all.py
```

All random number generators are fixed at `SEED = 42`.
