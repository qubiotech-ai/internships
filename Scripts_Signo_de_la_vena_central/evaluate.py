import csv

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: este script solo necesita guardar los PNG a disco
import matplotlib.pyplot as plt
import torch
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from model import CVSNet
from test_monai_dataset import (
    BASE_DIR,
    load_dataset,
    split_dataset,
    to_monai_samples,
    get_transforms,
)
from train import BEST_MODEL_PATH, BATCH_SIZE


# Rutas de salida
RESULTS_DIR = BASE_DIR / "resultados_evaluacion"
RESULTS_DIR.mkdir(exist_ok=True)

TEST_PREDICTIONS_CSV_PATH = RESULTS_DIR / "test_predictions.csv"
EVALUATION_METRICS_PATH = RESULTS_DIR / "evaluation_metrics.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"

CLASS_NAMES = ["CVS-", "CVS+"]  # índice 0 -> CVS-, índice 1 -> CVS+ (mismo orden que "label" en dataset.json)

# 1. Cargar el modelo entrenado

def load_model(device: torch.device, path=BEST_MODEL_PATH) -> CVSNet:
    model = CVSNet()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# 2. DataLoader de test 

def create_test_loader(
    batch_size: int = BATCH_SIZE,
) -> tuple[MonaiDataLoader, dict[str, str]]:
    dataset = load_dataset()
    _, _, test_entries = split_dataset(dataset)

    transforms = get_transforms()
    test_ds = MonaiDataset(data=to_monai_samples(test_entries), transform=transforms)
    test_loader = MonaiDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    lesion_id_to_patient_id = {entry["lesion_id"]: entry["patient_id"] for entry in test_entries}

    return test_loader, lesion_id_to_patient_id



# 3. Evaluación: recorrer test_loader sin aprender nada

def evaluate(
    model: CVSNet,
    test_loader: MonaiDataLoader,
    lesion_id_to_patient_id: dict[str, str],
    device: torch.device,
) -> list[dict]:
    model.eval()

    records = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            lesion_ids = batch["lesion_id"]

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            for i, lesion_id in enumerate(lesion_ids):
                records.append({
                    "lesion_id": lesion_id,
                    "patient_id": lesion_id_to_patient_id[lesion_id],
                    "true_label": labels[i].item(),
                    "predicted_label": predictions[i].item(),
                    "prob_cvs_neg": probabilities[i, 0].item(),
                    "prob_cvs_pos": probabilities[i, 1].item(),
                })

    return records



# 4. Guardar las predicciones muestra a muestra

def save_predictions(records: list[dict], path=TEST_PREDICTIONS_CSV_PATH) -> None:
    
    fieldnames = ["lesion_id", "patient_id", "true_label", "predicted_label", "prob_cvs_neg", "prob_cvs_pos"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)



# 5. Métricas agregadas

def compute_metrics(records: list[dict]) -> dict:
    
    y_true = [r["true_label"] for r in records]
    y_pred = [r["predicted_label"] for r in records]
    y_prob_pos = [r["prob_cvs_pos"] for r in records]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob_pos),
        "confusion_matrix": cm,
        "n_samples": len(records),
    }


def save_metrics(metrics: dict, path=EVALUATION_METRICS_PATH) -> None:
    tn, fp, fn, tp = metrics["confusion_matrix"].ravel()

    lines = [
        "=" * 60,
        "Evaluación final del modelo CVSNet sobre el conjunto de TEST",
        "=" * 60,
        f"Nº de muestras de test: {metrics['n_samples']}",
        "",
        f"Accuracy:    {metrics['accuracy']:.4f}",
        f"Precision:   {metrics['precision']:.4f}",
        f"Recall (Sensibilidad):    {metrics['recall']:.4f}",
        f"Specificity (Especificidad): {metrics['specificity']:.4f}",
        f"F1-score:    {metrics['f1']:.4f}",
        f"ROC-AUC:     {metrics['roc_auc']:.4f}",
        "",
        "Matriz de confusión (filas = etiqueta real, columnas = predicción):",
        f"  TN (CVS- bien clasificado): {tn}",
        f"  FP (CVS- clasificado como CVS+): {fp}",
        f"  FN (CVS+ clasificado como CVS-): {fn}",
        f"  TP (CVS+ bien clasificado): {tp}",
        "=" * 60,
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")



# 6. Figuras

def plot_confusion_matrix(cm, path=CONFUSION_MATRIX_PATH) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de confusión — CVSNet (test)")

    # Anota el número de lesiones en cada celda
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)

    fig.colorbar(im, ax=ax, label="Nº de lesiones")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: list[int], y_prob_pos: list[float], roc_auc: float, path=ROC_CURVE_PATH) -> None:
    
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"CVSNet (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC = 0.500)")

    ax.set_xlabel("1 - Especificidad (FPR)")
    ax.set_ylabel("Sensibilidad (TPR)")
    ax.set_title("Curva ROC — CVSNet (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



# Main
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(device)
    test_loader, lesion_id_to_patient_id = create_test_loader()
    print(f"Test: {len(test_loader.dataset)} muestras")

    records = evaluate(model, test_loader, lesion_id_to_patient_id, device)
    save_predictions(records)

    metrics = compute_metrics(records)
    save_metrics(metrics)

    plot_confusion_matrix(metrics["confusion_matrix"])

    y_true = [r["true_label"] for r in records]
    y_prob_pos = [r["prob_cvs_pos"] for r in records]
    plot_roc_curve(y_true, y_prob_pos, metrics["roc_auc"])

    print("\n" + "=" * 48)
    print("Evaluación final")
    print("=" * 48)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1-score:    {metrics['f1']:.4f}")
    print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    print("=" * 48)

    print(f"\nArchivos generados en {RESULTS_DIR.relative_to(BASE_DIR)}/:")
    print(f"  {EVALUATION_METRICS_PATH.name}")
    print(f"  {TEST_PREDICTIONS_CSV_PATH.name}")
    print(f"  {CONFUSION_MATRIX_PATH.name}")
    print(f"  {ROC_CURVE_PATH.name}")


if __name__ == "__main__":
    main()