import csv
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import CVSNet
from test_monai_dataset import (
    BASE_DIR,
    load_dataset,
    split_dataset,
    to_monai_samples,
    get_transforms,
)

# Hiperparámetros y configuración

SEED = 42

BATCH_SIZE = 16
MAX_EPOCHS = 100

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# nº de épocas SIN mejora de la validation
EARLY_STOPPING_PATIENCE = 10

# Paciencia del scheduler: nº de épocas sin mejora tras las cuales se
# reduce el learning rate. 
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

BEST_MODEL_PATH = BASE_DIR / "best_model.pth"
HISTORY_CSV_PATH = BASE_DIR / "history.csv"

# Análisis del aprendizaje por lesión durante el entrenamiento.
TRAIN_PREDICTIONS_DIR = BASE_DIR / "train_predictions"
TRAIN_LEARNING_SUMMARY_PATH = BASE_DIR / "train_learning_summary.csv"
ALWAYS_WRONG_PATH = BASE_DIR / "always_wrong.csv"
ALWAYS_CORRECT_PATH = BASE_DIR / "always_correct.csv"
LEARNING_PROGRESS_PATH = BASE_DIR / "learning_progress.csv"

# Reproducibilidad
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# DataLoaders (reutilizando el pipeline de test_monai_dataset.py)

def create_dataloaders(
    batch_size: int = BATCH_SIZE,
) -> tuple[MonaiDataLoader, MonaiDataLoader, MonaiDataLoader]:
    dataset = load_dataset()
    train_entries, val_entries, test_entries = split_dataset(dataset)

    transforms = get_transforms()

    train_ds = MonaiDataset(data=to_monai_samples(train_entries), transform=transforms)
    val_ds = MonaiDataset(data=to_monai_samples(val_entries), transform=transforms)
    test_ds = MonaiDataset(data=to_monai_samples(test_entries), transform=transforms)

    train_loader = MonaiDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = MonaiDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = MonaiDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# DataLoader adicional: solo lesiones ORIGINALES de train, para análisis
def create_train_eval_loader(batch_size: int = BATCH_SIZE) -> MonaiDataLoader:
    
    dataset = load_dataset()
    train_entries, _, _ = split_dataset(dataset)
    original_entries = [e for e in train_entries if e["augmentation"] == "original"]

    transforms = get_transforms()
    eval_ds = MonaiDataset(data=to_monai_samples(original_entries), transform=transforms)
    return MonaiDataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)


# Una época de entrenamiento

def train_one_epoch(
    model: nn.Module,
    loader: MonaiDataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    n_correct = 0
    n_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_correct += (outputs.argmax(dim=1) == labels).sum().item()
        n_samples += batch_size

    train_loss = total_loss / n_samples
    train_accuracy = n_correct / n_samples

    return train_loss, train_accuracy



# Validación
def validate(
    model: nn.Module,
    loader: MonaiDataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_correct += (preds == labels).sum().item()
            n_samples += batch_size

    val_loss = total_loss / n_samples
    val_accuracy = n_correct / n_samples

    return val_loss, val_accuracy



# Guardar checkpoint

def save_checkpoint(model: nn.Module, path: Path = BEST_MODEL_PATH) -> None:
    torch.save(model.state_dict(), path)


def save_history(history: list[dict], path: Path = HISTORY_CSV_PATH) -> None:
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                "lr", "epoch_time", "best_model",
            ],
        )
        writer.writeheader()
        writer.writerows(history)



# Análisis del aprendizaje por lesión (NO forma parte del entrenamiento)


def evaluate_train_epoch(
    model: nn.Module,
    loader: MonaiDataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> list[dict]:
    
    model.eval()

    per_sample_criterion = nn.CrossEntropyLoss(weight=criterion.weight, reduction="none")

    records = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            lesion_ids = batch["lesion_id"]
            patient_ids = batch["patient_id"]

            logits = model(images)
            losses = per_sample_criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            for i in range(len(lesion_ids)):
                true_label = labels[i].item()
                predicted_label = predictions[i].item()
                records.append({
                    "epoch": epoch,
                    "patient_id": patient_ids[i],
                    "lesion_id": lesion_ids[i],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "prob_cvs_neg": probabilities[i, 0].item(),
                    "prob_cvs_pos": probabilities[i, 1].item(),
                    "loss": losses[i].item(),
                    "correct": predicted_label == true_label,
                })

    return records


def save_train_epoch_predictions(
    records: list[dict],
    epoch: int,
    output_dir: Path = TRAIN_PREDICTIONS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"epoch_{epoch:03d}.csv"
    pd.DataFrame(records).to_csv(path, index=False)


def build_train_learning_summary(all_records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_records)

    summary_rows = []
    for (patient_id, lesion_id), group in df.groupby(["patient_id", "lesion_id"], sort=False):
        group = group.sort_values("epoch")
        label = int(group["true_label"].iloc[0])

        correct_epochs = group.loc[group["correct"], "epoch"]
        incorrect_epochs = group.loc[~group["correct"], "epoch"]

        summary_rows.append({
            "patient_id": patient_id,
            "lesion_id": lesion_id,
            "label": label,
            "epoch_primer_acierto": int(correct_epochs.min()) if not correct_epochs.empty else None,
            "epoch_ultimo_error": int(incorrect_epochs.max()) if not incorrect_epochs.empty else None,
            "num_epocas_correcta": int(group["correct"].sum()),
            "num_epocas_incorrecta": int((~group["correct"]).sum()),
            "siempre_correcta": bool(group["correct"].all()),
            "siempre_incorrecta": bool((~group["correct"]).all()),
        })

    return pd.DataFrame(summary_rows)


def save_learning_analysis(summary_df: pd.DataFrame) -> None:
    summary_df.to_csv(TRAIN_LEARNING_SUMMARY_PATH, index=False)

    summary_df[summary_df["siempre_incorrecta"]].to_csv(ALWAYS_WRONG_PATH, index=False)
    summary_df[summary_df["siempre_correcta"]].to_csv(ALWAYS_CORRECT_PATH, index=False)

    progress_columns = ["patient_id", "lesion_id", "label", "epoch_primer_acierto"]
    summary_df[progress_columns].to_csv(LEARNING_PROGRESS_PATH, index=False)



# Main: bucle de entrenamiento completo

def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(BATCH_SIZE)
    print(f"Train: {len(train_loader.dataset)} muestras | "
          f"Val: {len(val_loader.dataset)} muestras | "
          f"Test: {len(test_loader.dataset)} muestras (no se usa en este script)")

    # Loader adicional, de solo lectura, para el análisis de aprendizaje
    train_eval_loader = create_train_eval_loader(BATCH_SIZE)
    print(f"Train (análisis por época, solo lesiones originales): {len(train_eval_loader.dataset)} muestras")

    model = CVSNet().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Histórico de métricas por época.
    history = []

    # Predicciones de train de TODAS las épocas
    all_train_eval_records: list[dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        # Análisis por lesión (no forma parte del entrenamiento)
        train_eval_records = evaluate_train_epoch(model, train_eval_loader, criterion, device, epoch)
        save_train_epoch_predictions(train_eval_records, epoch)
        all_train_eval_records.extend(train_eval_records)

        epoch_time = time.perf_counter() - epoch_start

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "lr": current_lr,
            "epoch_time": epoch_time,
            "best_model": is_best,
        })
        save_history(history, HISTORY_CSV_PATH)

        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        print(f"Best Model: {'YES' if is_best else 'NO'}")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping: {EARLY_STOPPING_PATIENCE} épocas "
                f"seguidas sin mejorar la validation loss (mejor valor: "
                f"{best_val_loss:.4f})."
            )
            break

    print(f"\nEntrenamiento terminado. Mejor validation loss: {best_val_loss:.4f}")
    print(f"Pesos guardados en: {BEST_MODEL_PATH}")
    print(f"Histórico guardado en: {HISTORY_CSV_PATH}")

    print("\nGenerando análisis de aprendizaje por lesión...")
    summary_df = build_train_learning_summary(all_train_eval_records)
    save_learning_analysis(summary_df)
    print(f"  {TRAIN_PREDICTIONS_DIR.name}/epoch_XXX.csv ({len(history)} épocas)")
    print(f"  {TRAIN_LEARNING_SUMMARY_PATH.name} ({len(summary_df)} lesiones)")
    print(f"  {ALWAYS_WRONG_PATH.name} ({int(summary_df['siempre_incorrecta'].sum())} lesiones)")
    print(f"  {ALWAYS_CORRECT_PATH.name} ({int(summary_df['siempre_correcta'].sum())} lesiones)")
    print(f"  {LEARNING_PROGRESS_PATH.name}")


if __name__ == "__main__":
    main()