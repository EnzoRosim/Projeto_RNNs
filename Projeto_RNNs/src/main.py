"""Pipeline completo: EDA -> tuning (topologia, taxas) -> avaliação no teste.

Uso:
    python src/main.py                # roda tudo
    python src/main.py --skip-tuning  # só avalia o melhor modelo já salvo
"""
import argparse
import json

import torch

import config as cfg
import evaluate as ev
from data import get_dataloaders, load_series
from model import LSTMForecaster
from train import RATES_GRID, TOPOLOGY_GRID, tune

STAGES = {"tuning_topologia": list(TOPOLOGY_GRID), "tuning_taxas": list(RATES_GRID)}


def main(skip_tuning=False):
    # 1) EDA
    ev.plot_overview(load_series())

    # 2) Tuning em duas etapas (seleção pela loss de validação)
    if not skip_tuning:
        _, best_top = tune(TOPOLOGY_GRID, "tuning_topologia")
        print("Melhor topologia:", best_top)
        _, best = tune(RATES_GRID, "tuning_taxas", base=best_top)
        print("Melhor configuração final:", best)

    # 3) Carrega o melhor modelo final
    ckpt = torch.load(cfg.OUTPUT_DIR / "tuning_taxas_best.pt", map_location="cpu")
    p = ckpt["params"]
    train_dl, val_dl, test_dl, info = get_dataloaders(window=p["window"], batch_size=p["batch_size"])
    model = LSTMForecaster(info["n_features"], p["hidden_size"], p["num_layers"], p["dropout"])
    model.load_state_dict(ckpt["state_dict"])

    # 4) Avaliação no teste (LSTM x baselines)
    results, y, yhat = ev.evaluate(model, test_dl, info)
    print("\nResultados no teste (kW):\n", results.round(4))
    results.to_csv(cfg.OUTPUT_DIR / "metricas_teste.csv")
    with open(cfg.OUTPUT_DIR / "melhores_params.json", "w") as f:
        json.dump(p, f, indent=2)

    # 5) Gráficos do tuning e da previsão
    for stage, keys in STAGES.items():
        with open(cfg.OUTPUT_DIR / f"{stage}_histories.json") as f:
            hists = json.load(f)
        ev.plot_val_curves(hists, f"{stage}_val_curves.png", keys)
        best_h = min(hists.values(), key=lambda h: h["best_val_loss"])
        ev.plot_learning_curve(best_h, f"{stage}_melhor_curva.png",
                               f"Curva de aprendizagem — melhor de {stage.split('_')[1]}")
    ev.plot_predictions(info["target_index"]["test"], y, yhat, "previsao_teste.png")
    ev.plot_scatter(y, yhat, "dispersao_teste.png")
    print("Resíduos (kW):", ev.plot_residuals(y, yhat, "residuos_teste.png"))
    ev.plot_fit(model, {"train": train_dl, "val": val_dl, "test": test_dl}, info)
    print(f"\nArquivos salvos em {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tuning", action="store_true", help="pula o tuning e só avalia")
    main(parser.parse_args().skip_tuning)