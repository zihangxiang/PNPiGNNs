# Preserving Node-level Privacy in Graph Neural Networks

Code for the IEEE S&P 2024 paper *Preserving Node-level Privacy in Graph Neural Networks*.
It trains GNNs with node-level differential privacy.

## How it works

1. **Subgraph sampling** (Figure 3, `privacy/sampling.py`). Root nodes are
   Poisson-sampled. Each root is expanded into a small subgraph whose other
   nodes are neighbours kept with probability `M / d_out(u)`, so every node
   appears in only a bounded number of subgraphs per step.
2. **DP-SGD over subgraphs** (`train_scheduler.py`). The GNN classifies the
   root of each subgraph. Per-subgraph gradients are clipped, averaged and
   perturbed with Gaussian noise.
3. **Node-level accounting** (Theorem 2, `privacy/mix.py`). The privacy loss of
   one node is a Gaussian mixture over how often it was sampled. Its Rényi
   divergence is integrated numerically, composed over all training steps, and
   converted to (ε, δ). The noise multiplier σ is the smallest (to within 0.01) that meets the
   target ε.

## Repository layout

```
Preserving_Node_level_Privacy_in_Graph_Neural_Networks/
├── main.py                    # node-level DP GNN (the paper's method)
├── main_NaiveDPSGD.py         # baseline: DP-SGD MLP on node features only
├── train_scheduler.py         # DP-SGD training loops for both
├── utils.py                   # CLI arguments, metrics, result files
├── run_*.sh                   # experiment sweeps, one per dataset
├── datasets/
│   ├── SETUP.py               # data root, seeding, device
│   ├── utils.py               # dataset registry, node split, loader construction
│   └── model.py               # GNN operating on one padded subgraph
└── privacy/
    ├── sampling.py            # subgraph sampler + Poisson batch sampler (Figure 3)
    ├── mix.py                 # node-level accountant (Theorem 2)
    └── accounting_analysis.py # standard RDP / PRV accounting (baseline)
```

## Setup

```bash
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
```

Datasets are downloaded on first use into `GRAPH_DATA/` at the repository
root. Derived data is cached there too: node degrees, neighbour lists, and
noise multipliers in `privacy_node_dp/`. A cache entry is a deterministic
function of its file name (dataset, setting, seed), so it is safe to reuse
across runs. The baseline caches its noise multipliers in `privacy/stds.pt`.

## Running experiments

Run from the project directory:

```bash
cd Preserving_Node_level_Privacy_in_Graph_Neural_Networks
bash run_facebook.sh        # also: run_twitch.sh, run_pubmed.sh, run_amazon.sh, run_reddit.sh
bash run_NaiveDPSGD.sh      # baseline on all datasets
```

A single run:

```bash
python main.py --dataset facebook --expected_batchsize 4096 --epoch 9 --lr 0.01 \
    --priv_epsilon 8 --num_neighbors 3 --num_neighbors_test 7 --graph_setting transductive --seed 1
```

| Argument | Meaning |
| --- | --- |
| `--dataset` | e.g. `facebook`, `twitch_DE`, `PubMed`, `Amazon_Computers`, `Reddit` (full list in `datasets/utils.py`) |
| `--expected_batchsize` | expected number of roots per Poisson-sampled batch |
| `--epoch` | epochs; each has `ceil(N / expected_batchsize)` noisy steps |
| `--priv_epsilon` | target ε; δ is set to `1 / N^1.1` (N = number of training nodes) |
| `--num_neighbors` | neighbour budget `M` in training (enters the accountant) |
| `--num_neighbors_test` | maximum number of neighbours per test subgraph |
| `--graph_setting` | `transductive` (neighbours may be any node) or `inductive` (training nodes only) |
| `--C` | clipping threshold |
| `--K` | number of GNN layers / neighbour-sampling rounds |
| `--lr`, `--seed`, `--worker_num`, `--log_dir` | learning rate, seed (also fixes the 80/1/19 node split), DataLoader workers, log directory |

Results are written next to the scripts:

- `logs/log.txt`: full log of every run.
- `logs/weighted_recall.csv`: per-epoch train/val/test accuracy.
- `data_records/jd_<dataset>_<setting>_eps<ε>.json`: one entry per run, with all arguments (including the noise multiplier `std`) and the accuracy curves.

## Using the node-level accountant directly

```python
from privacy.mix import NodeDPAccountant

acc = NodeDPAccountant(q=0.2, num_steps=45, D_out=20000, M_train=1)
eps, alpha = acc.eps_from_noise(sigma=1.65, delta=1e-5)   # privacy of a given noise level
sigma = acc.noise_from_eps(2, delta=1e-5)                 # noise needed for a target epsilon
```

The same example runs with `python -m privacy.mix` from the project directory.

## Reference
Please cite our work if you find it useful:
```tex
@inproceedings{DBLP:conf/sp/XiangWW24,
  author       = {Zihang Xiang and
                  Tianhao Wang and
                  Di Wang},
  title        = {Preserving Node-level Privacy in Graph Neural Networks},
  booktitle    = {{IEEE} Symposium on Security and Privacy, {SP} 2024, San Francisco,
                  CA, USA, May 19-23, 2024},
  pages        = {4714--4732},
  publisher    = {{IEEE}},
  year         = {2024},
  url          = {https://doi.org/10.1109/SP54263.2024.00270},
  doi          = {10.1109/SP54263.2024.00270},
  timestamp    = {Sun, 06 Oct 2024 21:15:04 +0200},
  biburl       = {https://dblp.org/rec/conf/sp/XiangWW24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
