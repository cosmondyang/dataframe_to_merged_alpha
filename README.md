# dataframe_to_merged_alpha

This repository contains utilities for expanding large cross-sectional feature tables into alpha combinations.  The main entry point is `alpha_combo_generator_v2.py`, which implements the stochastic workflow described in the project specification.

## Running `alpha_combo_generator_v2.py`

1. **Prepare the feature parquet**  
   The script defaults to the shared demo file located at
   `/remote-home/yyc/gen_ocall_features_demo_v02/saved_features/fiveclass30sfeature2.parquet`.  If you want to use a different file, pass its path with `--parquet-path`.

2. **Launch the search**  
   ```bash
   python alpha_combo_generator_v2.py \
       --iterations 200 \
       --log-path combo_progress.txt \
       --combo-parquet best_combo.parquet \
       --figure-path figures/best_combo.png
   ```
   Additional useful flags:
   - `--parquet-path` – override the default parquet location.
   - `--seed` – set a random seed for reproducibility.
   - `--iterations` – control how many random alphas to explore in one run.
   - `--touchstone-state` / `--touchstone-T` – forwarded to `examine_featuretocsv3.analyze` when computing IC and `lw_ic_v2`.

During the run the script writes:
- `combo_progress.txt`: an append-only log describing every alpha variant that improved the combo (depth, preprocessing, operators, weights, before/after metrics).
- `best_combo.parquet`: overwritten after each improvement and contains only the current best combined alpha values.
- `figures/best_combo.png` (or your custom path): overwritten with the IC / `lw_ic_v2` traces of the strongest combo so far.

## Uploading results to GitHub

The container environment does not automatically push to GitHub.  To publish your work:

1. Configure the remote once (replace the URL with your repository):
   ```bash
   git remote add origin git@github.com:your-account/your-repo.git
   ```

2. Commit your changes locally:
   ```bash
   git add .
   git commit -m "Describe your change"
   ```

3. Push the commit to GitHub:
   ```bash
   git push origin <branch-name>
   ```

If you cloned the repository locally with the remote already configured, only steps 2 and 3 are required after making modifications.

