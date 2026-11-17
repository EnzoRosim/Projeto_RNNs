# Household Power Consumption Forecasting with LSTM

Project 4 for *Introduction to Artificial Neural Networks* (Computer Science).
An LSTM that predicts the household's global active power **1 hour ahead** from the previous 24 hours.

## Task and data

- **Dataset:** [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (UCI), one household, 1-minute measurements from Dec 2006 to Nov 2010.
- **Target:** `Global_active_power` (kW) at t+1 h.
- **Preprocessing:** missing values (`?`) interpolated, series resampled to hourly means, 11 input features (7 electrical measurements + sin/cos of hour and day of week), standardization fitted on the training set only.
- **Split:** chronological 70% train / 15% validation / 15% test.

## Project structure

```
database/data/   # dataset (.txt or .zip) goes here
src/
  config.py      # paths, seed, default hyperparameters
  data.py        # loading, cleaning, windowing, DataLoaders
  model.py       # LSTM model
  train.py       # training, early stopping, hyperparameter search
  evaluate.py    # baselines, metrics, plots
  main.py        # full pipeline
output/          # models, tuning results, figures, metrics
requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

python src/data.py                # check data loading and splits
python src/train.py               # quick test: train the default config
python src/main.py                # full pipeline (EDA, tuning, test evaluation)
python src/main.py --skip-tuning  # only re-evaluate the saved best model
```

Everything is saved to `output/` (tuning tables, learning curves, test metrics, figures).

## Method

- Adam optimizer, MSE loss, gradient clipping, early stopping on validation loss, fixed seed (42).
- Two-stage grid search selected by validation loss: (1) topology (hidden size, layers, dropout, window), (2) rates (learning rate, batch size, weight decay).
- **Final model:** 2 LSTM layers, 64 hidden units, dropout 0.2, window 24 h, lr 1e-3, batch 64, weight decay 1e-4.

## Results (test set, kW)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence | 0.373 | 0.575 | 0.330 |
| Moving average (24 h) | 0.519 | 0.669 | 0.093 |
| Daily seasonal (24 h ago) | 0.505 | 0.752 | -0.146 |
| **LSTM** | **0.317** | **0.459** | **0.573** |

The LSTM reduces MAE by about 15% and RMSE by about 20% compared to the best naive baseline (persistence).

## Author

Enzo Rosim
