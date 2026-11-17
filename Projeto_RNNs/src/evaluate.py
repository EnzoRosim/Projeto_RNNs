"""Baselines, métricas (em kW) e gráficos."""
import matplotlib

matplotlib.use("Agg")  # salva figuras sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

import config as cfg
from data import inverse_target
from train import DEVICE

# Paleta categórica (ordem fixa) e tons de texto/grade
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2,
    "legend.frameon": False,
})


# ---------------- Previsão e métricas ----------------
@torch.no_grad()
def predict(model, loader):
    """Retorna (y_true, y_pred) na escala normalizada."""
    model.to(DEVICE).eval()
    preds, ys = [], []
    for X, y in loader:
        preds.append(model(X.to(DEVICE)).cpu())
        ys.append(y)
    return torch.cat(ys).numpy().ravel(), torch.cat(preds).numpy().ravel()


def metrics(y, p) -> dict:
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p)}


def baselines(loader, info) -> dict:
    """Previsões ingênuas a partir das próprias janelas (escala normalizada)."""
    X = loader.dataset.tensors[0][:, :, info["target_idx"]].numpy()
    b = {"Persistência (valor em t)": X[:, -1],
         f"Média móvel ({X.shape[1]}h)": X.mean(axis=1)}
    if X.shape[1] >= 24:
        b["Sazonal diária (24h antes)"] = X[:, -24]  # t+1-24 = t-23
    return b


def evaluate(model, loader, info):
    """Compara LSTM e baselines em kW. Retorna (tabela, y_true_kW, y_pred_kW)."""
    y, p = predict(model, loader)
    y_kw, p_kw = inverse_target(y, info), inverse_target(p, info)
    rows = [{"modelo": "LSTM", **metrics(y_kw, p_kw)}]
    for nome, b in baselines(loader, info).items():
        rows.append({"modelo": nome, **metrics(y_kw, inverse_target(b, info))})
    return pd.DataFrame(rows).set_index("modelo"), y_kw, p_kw


# ---------------- Gráficos ----------------
def _save(fig, name):
    fig.savefig(cfg.OUTPUT_DIR / name)
    plt.close(fig)


def plot_overview(df):
    """EDA: média diária do alvo e perfil médio por hora do dia."""
    s = df[cfg.TARGET]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].plot(s.resample("1D").mean(), color=COLORS[0], linewidth=1)
    ax[0].set(title="Consumo médio diário", ylabel="Potência ativa (kW)")
    prof = s.groupby(s.index.hour).mean()
    ax[1].bar(prof.index, prof.values, color=COLORS[0], width=0.8)
    ax[1].set(title="Perfil médio por hora", xlabel="Hora do dia", ylabel="kW")
    _save(fig, "eda_overview.png")


def plot_learning_curve(hist, name, title="Curva de aprendizagem"):
    """Loss de treino e validação por época, marcando a melhor época."""
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, hist["train_loss"], color=COLORS[0], label="Treino")
    ax.plot(ep, hist["val_loss"], color=COLORS[1], label="Validação")
    ax.axvline(hist["best_epoch"], color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"melhor época ({hist['best_epoch']})", (hist["best_epoch"], ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points", color=MUTED, fontsize=9)
    ax.set(title=title, xlabel="Época", ylabel="MSE (normalizado)")
    ax.legend()
    _save(fig, name)


def plot_val_curves(histories, name, keys, top=5, title="Validação por configuração"):
    """Curvas de validação das `top` melhores tentativas de uma etapa do tuning."""
    best = sorted(histories.values(), key=lambda h: h["best_val_loss"])[:top]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, h in zip(COLORS, best):
        label = ", ".join(f"{k}={h['params'][k]}" for k in keys)
        ax.plot(np.arange(1, len(h["val_loss"]) + 1), h["val_loss"], color=c, label=label)
    ax.set(title=title, xlabel="Época", ylabel="MSE de validação (normalizado)")
    ax.legend(fontsize=8)
    _save(fig, name)


def plot_predictions(index, y, p, name, days=7):
    """Previsto vs real nos primeiros `days` dias do teste."""
    n = days * 24
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(index[:n], y[:n], color=COLORS[0], label="Real")
    ax.plot(index[:n], p[:n], color=COLORS[1], label="Previsto (LSTM)")
    ax.set(title=f"Previsão 1h à frente — {days} dias do teste", ylabel="Potência ativa (kW)")
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, name)


def predict_splits(model, loaders, info) -> dict:
    """Previsões em kW para cada split, em ordem temporal: {split: DataFrame(real, prev)}."""
    out = {}
    for nome, dl in loaders.items():
        dl = DataLoader(dl.dataset, batch_size=512, shuffle=False)  # treino vem embaralhado
        y, p = predict(model, dl)
        out[nome] = pd.DataFrame({"real": inverse_target(y, info), "prev": inverse_target(p, info)},
                                 index=info["target_index"][nome])
    return out


def plot_fit(model, loaders, info, name="ajuste_serie.png", zoom_days=7):
    """Série original (pontos coloridos por split) x curva aprendida pela LSTM.
    Cima: série inteira (média diária). Baixo: zoom horário na fronteira val/teste."""
    splits = predict_splits(model, loaders, info)
    cores = {"train": COLORS[0], "val": COLORS[1], "test": COLORS[2]}
    nomes = {"train": "Treino (real)", "val": "Validação (real)", "test": "Teste (real)"}
    prev = pd.concat([s["prev"] for s in splits.values()])

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 1]})

    # Série completa (média diária, para caber 4 anos)
    for k, s in splits.items():
        d = s["real"].resample("1D").mean()
        ax[0].scatter(d.index, d.values, s=7, color=cores[k], label=nomes[k], zorder=2)
    pd_ = prev.resample("1D").mean()
    ax[0].plot(pd_.index, pd_.values, color=INK, linewidth=0.8, label="Previsão LSTM", zorder=3)
    ax[0].set(title="Série original x curva aprendida (média diária)", ylabel="Potência ativa (kW)")

    # Zoom horário: metade antes e metade depois do início do teste
    t0, h = splits["test"].index[0], pd.Timedelta(hours=zoom_days * 12)
    for k, s in splits.items():
        z = s.loc[t0 - h: t0 + h]
        ax[1].scatter(z.index, z["real"], s=12, color=cores[k], zorder=2)
    zp = prev.loc[t0 - h: t0 + h]
    ax[1].plot(zp.index, zp.values, color=INK, linewidth=1.2, zorder=3)
    ax[1].set(title=f"Zoom horário ({zoom_days} dias) na passagem validação → teste",
              ylabel="Potência ativa (kW)")

    # Fronteiras entre splits (no zoom, só a do teste)
    for k in ("val", "test"):
        ax[0].axvline(splits[k].index[0], color=MUTED, linestyle="--", linewidth=1, zorder=1)
    ax[1].axvline(t0, color=MUTED, linestyle="--", linewidth=1, zorder=1)
    ax[1].set_xlim(t0 - h, t0 + h)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, markerscale=2)
    _save(fig, name)


def plot_residuals(y, p, name):
    """Histograma dos resíduos (real - previsto). Retorna média e desvio em kW."""
    r = np.asarray(y) - np.asarray(p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(r, bins=60, color=COLORS[0], edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set(title="Resíduos no teste (real − previsto)", xlabel="Resíduo (kW)", ylabel="Frequência")
    _save(fig, name)
    return {"media": float(r.mean()), "desvio": float(r.std())}


def plot_scatter(y, p, name):
    """Dispersão previsto x real com a diagonal ideal."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, p, s=8, alpha=0.3, color=COLORS[0], edgecolors="none")
    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1)
    ax.set(title="Previsto x real (teste)", xlabel="Real (kW)", ylabel="Previsto (kW)")
    _save(fig, name)