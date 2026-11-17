"""Leitura, limpeza, split temporal, normalização e janelamento da série."""
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import config as cfg


def load_raw() -> pd.DataFrame:
    """Lê o dataset original (1 min) com índice datetime. Extrai o .zip se faltar o .txt."""
    txts = sorted(cfg.DATA_DIR.glob("*.txt"))
    if not txts:
        zips = sorted(cfg.DATA_DIR.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"Nenhum .txt ou .zip em {cfg.DATA_DIR}")
        with zipfile.ZipFile(zips[0]) as zf:
            nome = next(n for n in zf.namelist() if n.endswith(".txt"))
            zf.extract(nome, cfg.DATA_DIR)
        txts = sorted(cfg.DATA_DIR.glob("*.txt"))

    df = pd.read_csv(txts[0], sep=";", na_values=["?", ""], low_memory=False)
    df.index = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df.index.name = "datetime"
    return df.drop(columns=["Date", "Time"]).astype("float32")


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Reamostra pela média (ignora NaN) e preenche lacunas restantes por interpolação no tempo."""
    out = df.resample(cfg.FREQ).mean()
    return out.interpolate(method="time").ffill().bfill()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica hora e dia da semana de forma cíclica."""
    df = df.copy()
    h, d = df.index.hour, df.index.day_of_week
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)
    return df


@lru_cache(maxsize=1)
def load_series() -> pd.DataFrame:
    """Pipeline completo até a série horária pronta (em cache: lê o .txt uma vez só)."""
    df = to_hourly(load_raw())
    return add_time_features(df) if cfg.USE_TIME_FEATURES else df


def make_windows(arr: np.ndarray, target_idx: int, window: int, horizon: int):
    """X: (N, window, F) com instantes até t; y: (N,) com o alvo em t+horizon."""
    n = len(arr) - window - horizon + 1
    X = sliding_window_view(arr, window, axis=0)[:n].transpose(0, 2, 1)
    y = arr[window + horizon - 1: window + horizon - 1 + n, target_idx]
    return np.ascontiguousarray(X), y


def get_dataloaders(window=None, horizon=None, batch_size=None):
    """Retorna (train, val, test) DataLoaders e um dict `info` com scaler e metadados."""
    window = window or cfg.WINDOW
    horizon = horizon or cfg.HORIZON
    batch_size = batch_size or cfg.BATCH_SIZE

    df = load_series()
    n = len(df)
    i_tr, i_va = int(n * cfg.TRAIN_FRAC), int(n * (cfg.TRAIN_FRAC + cfg.VAL_FRAC))
    bounds = {"train": (0, i_tr), "val": (i_tr, i_va), "test": (i_va, n)}

    # Scaler ajustado só no treino (evita vazamento)
    scaler = StandardScaler().fit(df.iloc[:i_tr].values)
    data = scaler.transform(df.values).astype("float32")
    target_idx = list(df.columns).index(cfg.TARGET)

    loaders, target_index = {}, {}
    for nome, (a, b) in bounds.items():
        # Val/teste usam as `window` horas anteriores como contexto; os alvos ficam dentro do split
        s = max(a - window, 0)
        X, y = make_windows(data[s:b], target_idx, window, horizon)
        target_index[nome] = df.index[s + window + horizon - 1: s + window + horizon - 1 + len(y)]
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
        loaders[nome] = DataLoader(ds, batch_size=batch_size, shuffle=(nome == "train"))

    info = {
        "scaler": scaler,
        "target_idx": target_idx,
        "features": list(df.columns),
        "n_features": df.shape[1],
        "target_index": target_index,  # timestamps dos alvos de cada split
    }
    return loaders["train"], loaders["val"], loaders["test"], info


def inverse_target(y_scaled, info) -> np.ndarray:
    """Volta o alvo para a escala original (kW)."""
    sc, i = info["scaler"], info["target_idx"]
    return np.asarray(y_scaled) * sc.scale_[i] + sc.mean_[i]


if __name__ == "__main__":
    tr, va, te, info = get_dataloaders()
    print("Features:", info["features"])
    for nome, dl in [("train", tr), ("val", va), ("test", te)]:
        X, y = dl.dataset.tensors
        idx = info["target_index"][nome]
        print(f"{nome}: X={tuple(X.shape)} y={tuple(y.shape)} | {idx[0]} -> {idx[-1]}")