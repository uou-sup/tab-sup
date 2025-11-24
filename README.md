## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download raw datasets** (optional if already present)
   ```bash
   python dataset/download_dataset.py
   ```

3. **Prepare processed tensors** (choose one or more)
   ```bash
   python -m sup.prepare_shoppers_dataset      # shoppers
   python -m sup.prepare_adult_dataset         # adult
   python -m sup.prepare_default_dataset       # default of credit card clients
   python -m sup.prepare_magic_dataset         # MAGIC gamma
   python -m sup.prepare_beijing_dataset       # Beijing PM2.5
   python -m sup.prepare_news_dataset          # Online News Popularity
   ```
   These scripts write `dataset/<name>/X_*.npy`, `y_*.npy`, and `info.json`.

4. **Train Tab-SEDD**
   ```bash
   python -m sup.train --config sup/configs/shoppers.toml
   ```
   Replace the config path with `sup/configs/adult.toml`, `sup/configs/default.toml`, etc., to train on other datasets. Set `WANDB_MODE=disabled` if network access is restricted.

5. **Sample / Impute** (after a checkpoint exists)
   ```bash
   python -m sup.generate_samples --config sup/configs/shoppers.toml --checkpoint sup/checkpoints/shoppers/best.pt
   python -m sup.generate_imputations --config sup/configs/shoppers.toml --checkpoint sup/checkpoints/shoppers/best.pt \
       --split test --token-mask-value -1 --decoded-output samples/shoppers_imputed.csv
   ```

## Directory Structure Highlights

```
dataset/        # processed tensors (X_num_*.npy, X_cat_*.npy, y_*.npy, info.json)
data/           # raw CSV/XLS + metadata from download_dataset.py
sup/configs/    # training TOML configs (shoppers, adult, default, magic, beijing, news, ...)
sup/prepare_*   # dataset preparation scripts
```
