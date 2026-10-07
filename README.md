# A Hybrid Two-Stage Framework for Twitter Bot Detection: Decoupling Graph Representation Learning and Classification

Code accompanying the paper "A Hybrid Two-Stage Framework for Twitter Bot Detection: Decoupling Graph Representation Learning and Classification" (MAPR 2026). The framework trains a GNN (BotRGCN, GCN, or GAT) as a frozen feature extractor, then trains XGBoost independently on the extracted embeddings for final classification.

## Requirements

Install the following packages before running any script.

```
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for GNN training and embedding extraction.

## Data

This project uses the TwiBot-22 dataset. Download it separately and place it in a folder containing at minimum:

```
user.json
label.csv
split.csv
edge.csv
```

Set the following environment variables before running any script.

```
export TWIBOT22_ROOT=/path/to/TwiBot-22
export PROCESSED_DIR=./processed_data
```

`TWIBOT22_ROOT` should point to the folder above. `PROCESSED_DIR` is where preprocessed tensors, model checkpoints, embeddings, and result files are written.

## How to run

Run the following steps in order. All commands assume `TWIBOT22_ROOT` and `PROCESSED_DIR` are set as described above.

### 1. Preprocessing

```
python preprocess.py
```

This produces the tensors required by `Dataset.py`, including `des_tensor.pt` and `tweets_tensor.pt`, inside `PROCESSED_DIR`.

### 2. Train a GNN backbone

```
python train.py 42
```

The argument is the random seed. Repeat for each seed used in the paper (42 through 46). This saves the best checkpoint as `{PROCESSED_DIR}/botrgcn_best_{seed}.pt`.

To train GCN or GAT instead of BotRGCN, use the corresponding model class from `model.py` in place of BotRGCN in `train.py`.

### 3. Extract frozen embeddings

```
python extract_embeddings.py --model botrgcn --layers alllayers --seed 42
```

Supported values for `--model` are `botrgcn`, `botgcn`, and `botgat`. Supported values for `--layers` are `default`, `layer1`, and `alllayers`. Repeat across seeds 42 through 46 before running the analysis scripts below.

### 4. Classifier ablation (Table II)

```
python classifier_ablation.py --seed 42 --model botrgcn
```

### 5. Class weighting and threshold ablation (Table IV)

```
python ablation_weighting_threshold.py
```

### 6. MLP architecture tuning

```
python tune_mlp.py
```

### 7. XGBoost hyperparameter sweep (Figure 2 data)

```
python hyperparam_sweep.py
```

This writes `hyperparam_sweep_results.json` and `hyperparam_sweep_results.csv` to `PROCESSED_DIR`.

### 8. Plots (Figures 2 and 3)

```
python plot_curves.py
```

This reads the sweep results from step 7 and produces the hyperparameter sensitivity plots, the ROC curve, and the precision-recall curve.

## Configuration

All paths are controlled through `config.py` via the `TWIBOT22_ROOT` and `PROCESSED_DIR` environment variables. Default XGBoost hyperparameters used across the ablation scripts are also defined in `config.py` and can be edited there directly.

## How to cite

If you found our work useful, please cite us. For the two-stage decoupled GNN and XGBoost framework on TwiBot-22, please cite:

Khoi Nguyen Pham, Ngoc Thao Vy Tran, Thanh Quan Nguyen, Kim Ngan Tran, Minh Phuong Ha, Hung-Nghiep Tran. "A Hybrid Two-Stage Framework for Twitter Bot Detection: Decoupling Graph Representation Learning and Classification." MAPR, 2026. doi: xxx

```bibtex
@INPROCEEDINGS{11685731,
  author={Pham, Khoi Nguyen and Tran, Ngoc Thao Vy and Nguyen, Thanh Quan and Tran, Kim Ngan and Ha, Minh Phuong and Tran, Hung-Nghiep},
  booktitle={2026 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)}, 
  title={A Hybrid Two-Stage Framework for Twitter Bot Detection: Decoupling Graph Representation Learning and Classification}, 
  year={2026},
  volume={},
  number={},
  pages={394-399},
  keywords={Graph neural networks;Modeling;Training;Social networking (online);Bot (Internet);Chatbots;Signal detection;Equations;Manuals;Learning (artificial intelligence);Twitter bot detection;graph neural networks;decoupled learning;XGBoost;TwiBot-22},
  doi={10.1109/MAPR72750.2026.11685731}}
```

For the TwiBot-22 dataset itself, please also cite the original benchmark paper as referenced in our related work section.
