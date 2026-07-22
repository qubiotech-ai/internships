
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: este script solo necesita guardar los PNG a disco
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


# Rutas

BASE_DIR = Path(__file__).parent.parent  # segmentaciones/ (este script vive en segmentaciones/scripts/)
RESULTS_DIR = BASE_DIR / "resultados_evaluacion"

TEST_PREDICTIONS_CSV_PATH = RESULTS_DIR / "test_predictions.csv"
THRESHOLD_METRICS_CSV_PATH = RESULTS_DIR / "threshold_metrics.csv"
SUMMARY_PATH = RESULTS_DIR / "threshold_optimization_summary.txt"

PRECISION_PLOT_PATH = RESULTS_DIR / "precision_vs_threshold.png"
RECALL_PLOT_PATH = RESULTS_DIR / "recall_vs_threshold.png"
F1_PLOT_PATH = RESULTS_DIR / "f1_vs_threshold.png"
SPECIFICITY_PLOT_PATH = RESULTS_DIR / "specificity_vs_threshold.png"

THRESHOLDS = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)



# 1. Cargar test_predictions.csv

def load_predictions(path: Path = TEST_PREDICTIONS_CSV_PATH) -> pd.DataFrame:
  
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Este script no hace inferencia: "
            "ejecuta antes evaluate.py para generar test_predictions.csv."
        )
    return pd.read_csv(path)



# 2. Métricas para un threshold concreto


def metrics_at_threshold(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float) -> dict:
  
    y_pred = (y_prob_pos >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    youden = recall + specificity - 1

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "youden_j": youden,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def sweep_thresholds(y_true: np.ndarray, y_prob_pos: np.ndarray, thresholds: np.ndarray = THRESHOLDS) -> pd.DataFrame:
    
    rows = [metrics_at_threshold(y_true, y_prob_pos, t) for t in thresholds]
    return pd.DataFrame(rows)



# 3. Selección de los thresholds recomendados


def best_by_f1(metrics_df: pd.DataFrame) -> pd.Series:
    """Fila (threshold + métricas) que maximiza el F1-score."""
    return metrics_df.loc[metrics_df["f1"].idxmax()]


def best_by_youden(metrics_df: pd.DataFrame) -> pd.Series:
    """Fila (threshold + métricas) que maximiza el índice de Youden (recall + specificity - 1)."""
    return metrics_df.loc[metrics_df["youden_j"].idxmax()]



# 4. Gráficas


def _plot_metric_vs_threshold(
    metrics_df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    path: Path,
    best_threshold: float,
) -> None:
    """
    Dibuja una métrica frente al threshold, con una línea vertical
    marcando el threshold recomendado por F1 para poder situar
    visualmente el punto de corte elegido sobre la curva.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(metrics_df["threshold"], metrics_df[column], color="tab:blue")
    ax.axvline(best_threshold, color="tab:red", linestyle="--", label=f"Mejor F1 (threshold={best_threshold:.2f})")

    ax.set_xlabel("Threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(metrics_df: pd.DataFrame, best_f1_threshold: float) -> None:
    _plot_metric_vs_threshold(metrics_df, "precision", "Precision", "Precision vs Threshold", PRECISION_PLOT_PATH, best_f1_threshold)
    _plot_metric_vs_threshold(metrics_df, "recall", "Recall (Sensibilidad)", "Recall vs Threshold", RECALL_PLOT_PATH, best_f1_threshold)
    _plot_metric_vs_threshold(metrics_df, "f1", "F1-score", "F1-score vs Threshold", F1_PLOT_PATH, best_f1_threshold)
    _plot_metric_vs_threshold(metrics_df, "specificity", "Specificity (Especificidad)", "Specificity vs Threshold", SPECIFICITY_PLOT_PATH, best_f1_threshold)


# 5. Resumen en texto


def format_row(label: str, row: pd.Series) -> list[str]:
    return [
        f"{label}: threshold = {row['threshold']:.2f}",
        f"  Accuracy:    {row['accuracy']:.4f}",
        f"  Precision:   {row['precision']:.4f}",
        f"  Recall:      {row['recall']:.4f}",
        f"  Specificity: {row['specificity']:.4f}",
        f"  F1-score:    {row['f1']:.4f}",
        f"  Youden J:    {row['youden_j']:.4f}",
        f"  Matriz de confusión: TN={int(row['tn'])}  FP={int(row['fp'])}  FN={int(row['fn'])}  TP={int(row['tp'])}",
    ]


def save_summary(roc_auc: float, row_f1: pd.Series, row_youden: pd.Series, n_samples: int, path: Path = SUMMARY_PATH) -> None:
    lines = [
        "=" * 60,
        "Optimización de threshold — CVSNet (a partir de test_predictions.csv)",
        "=" * 60,
        f"Nº de muestras de test: {n_samples}",
        f"ROC-AUC (no depende del threshold): {roc_auc:.4f}",
        "",
        *format_row("Threshold recomendado por F1-score", row_f1),
        "",
        *format_row("Threshold recomendado por índice de Youden", row_youden),
        "=" * 60,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")



# Main


def main() -> None:
    df = load_predictions()
    y_true = df["true_label"].to_numpy()
    y_prob_pos = df["prob_cvs_pos"].to_numpy()

    roc_auc = roc_auc_score(y_true, y_prob_pos)

    metrics_df = sweep_thresholds(y_true, y_prob_pos)
    metrics_df.to_csv(THRESHOLD_METRICS_CSV_PATH, index=False)

    row_f1 = best_by_f1(metrics_df)
    row_youden = best_by_youden(metrics_df)

    plot_all(metrics_df, best_f1_threshold=row_f1["threshold"])

    save_summary(roc_auc, row_f1, row_youden, n_samples=len(df))

    print("=" * 60)
    print("Optimización de threshold — CVSNet")
    print("=" * 60)
    print(f"Nº de muestras de test: {len(df)}")
    print(f"ROC-AUC (no depende del threshold): {roc_auc:.4f}")

    print("\n--- Recomendado por F1-score ---")
    for line in format_row("Threshold", row_f1):
        print(line)

    print("\n--- Recomendado por índice de Youden ---")
    for line in format_row("Threshold", row_youden):
        print(line)

    print(f"\nArchivos generados en {RESULTS_DIR.relative_to(BASE_DIR)}/:")
    print(f"  {THRESHOLD_METRICS_CSV_PATH.name}")
    print(f"  {SUMMARY_PATH.name}")
    print(f"  {PRECISION_PLOT_PATH.name}")
    print(f"  {RECALL_PLOT_PATH.name}")
    print(f"  {F1_PLOT_PATH.name}")
    print(f"  {SPECIFICITY_PLOT_PATH.name}")


if __name__ == "__main__":
    main()
