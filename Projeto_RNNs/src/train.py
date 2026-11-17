"""Treino com early stopping e busca de hiperparâmetros."""
import copy
import itertools
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as cfg
from data import get_dataloaders
from model import LSTMForecaster

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULTS = {
    "window": cfg.WINDOW, "batch_size": cfg.BATCH_SIZE,
    "hidden_size": cfg.HIDDEN_SIZE, "num_layers": cfg.NUM_LAYERS, "dropout": cfg.DROPOUT,
    "lr": cfg.LR, "weight_decay": cfg.WEIGHT_DECAY,
}

# Etapa 1: topologia | Etapa 2: taxas (usa a melhor topologia)
TOPOLOGY_GRID = {"hidden_size": [32, 64, 128], "num_layers": [1, 2],
                 "dropout": [0.0, 0.2], "window": [24, 48]}
RATES_GRID = {"lr": [1e-2, 1e-3, 1e-4], "batch_size": [32, 64, 128], "weight_decay": [0.0, 1e-4]}


def set_seed(seed=cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, optimizer=None):
    """Uma passada no loader. Com optimizer treina; sem, só avalia. Retorna a loss média."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = loss_fn(model(X), y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # evita explosão do gradiente
                optimizer.step()
            total += loss.item() * len(X)
            n += len(X)
    return total / n


def fit(model, train_dl, val_dl, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
        epochs=cfg.EPOCHS, patience=cfg.PATIENCE, verbose=True):
    """Treina com early stopping na val. O modelo termina com os pesos da melhor época."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    hist = {"train_loss": [], "val_loss": []}
    best, best_state, wait = float("inf"), None, 0

    for ep in range(1, epochs + 1):
        tr = run_epoch(model, train_dl, loss_fn, opt)
        va = run_epoch(model, val_dl, loss_fn)
        hist["train_loss"].append(tr)
        hist["val_loss"].append(va)
        if verbose:
            print(f"época {ep:3d} | train {tr:.4f} | val {va:.4f}")

        if va < best:
            best, best_state, wait = va, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping na época {ep}")
                break

    model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmin(hist["val_loss"])) + 1
    hist["best_val_loss"] = best
    return hist


def train_config(params=None, verbose=False):
    """Treina uma configuração (sobrescreve DEFAULTS). Retorna modelo, histórico, info e params."""
    p = {**DEFAULTS, **(params or {})}
    set_seed()
    tr, va, _, info = get_dataloaders(window=p["window"], batch_size=p["batch_size"])
    model = LSTMForecaster(info["n_features"], p["hidden_size"], p["num_layers"], p["dropout"])
    hist = fit(model, tr, va, p["lr"], p["weight_decay"], verbose=verbose)
    return model, hist, info, p


def tune(grid, name="tuning", n_trials=None, base=None):
    """Busca em grade (ou aleatória, se n_trials). Salva CSV, históricos e melhor modelo em output/."""
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    if n_trials and n_trials < len(combos):
        combos = random.Random(cfg.SEED).sample(combos, n_trials)

    rows, hists, best_loss, best_params = [], {}, float("inf"), None
    for i, params in enumerate(combos, 1):
        t0 = time.time()
        model, hist, _, p = train_config({**(base or {}), **params})
        rows.append({"trial": i, **p, "best_epoch": hist["best_epoch"],
                     "val_loss": hist["best_val_loss"], "time_s": round(time.time() - t0, 1)})
        hists[i] = {**hist, "params": p}
        print(f"[{i}/{len(combos)}] {params} -> val {hist['best_val_loss']:.4f} "
              f"(época {hist['best_epoch']}, {rows[-1]['time_s']}s)")

        if hist["best_val_loss"] < best_loss:
            best_loss, best_params = hist["best_val_loss"], p
            torch.save({"state_dict": model.state_dict(), "params": p},
                       cfg.OUTPUT_DIR / f"{name}_best.pt")

    df = pd.DataFrame(rows).sort_values("val_loss")
    df.to_csv(cfg.OUTPUT_DIR / f"{name}.csv", index=False)
    with open(cfg.OUTPUT_DIR / f"{name}_histories.json", "w") as f:
        json.dump(hists, f)
    return df, best_params


if __name__ == "__main__":
    # Teste rápido: treina a configuração padrão
    print("Dispositivo:", DEVICE)
    import evaluate as ev  # import local para evitar import circular

    model, hist, _, p = train_config(verbose=True)
    print(f"Melhor val_loss: {hist['best_val_loss']:.4f} na época {hist['best_epoch']} | {p}")

    tr, va, te, info = get_dataloaders(window=p["window"], batch_size=p["batch_size"])
    ev.plot_learning_curve(hist, "padrao_curva.png", "Curva de aprendizagem — config. padrão")
    ev.plot_fit(model, {"train": tr, "val": va, "test": te}, info, "padrao_ajuste_serie.png")
    print(f"Gráficos salvos em {cfg.OUTPUT_DIR}")