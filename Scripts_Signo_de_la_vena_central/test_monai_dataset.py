import json
from collections import defaultdict
from pathlib import Path

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ConcatItemsd,
    DeleteItemsd,
    ToTensord,
)
from sklearn.model_selection import train_test_split


# Constantes

BASE_DIR = Path(__file__).parent.parent  # segmentaciones
DATASET_JSON_PATH = BASE_DIR / "dataset.json"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Semilla fija para que el split sea el mismo en cada ejecución.
RANDOM_SEED = 42

BATCH_SIZE = 8



# 1. Carga de dataset.json

def load_dataset(json_path: Path = DATASET_JSON_PATH) -> list[dict]:
 
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset



# 2. Split train/val/test por lesión

def get_base_lesion_id(entry: dict) -> str:
    suffix = f"_{entry['augmentation']}"
    assert entry["lesion_id"].endswith(suffix), (
        f"lesion_id {entry['lesion_id']!r} no termina en {suffix!r}"
    )
    return entry["lesion_id"][: -len(suffix)]


def split_dataset(
    dataset: list[dict],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
   
    # Agrupa las filas de dataset.json por lesión base.
    lesion_rows = defaultdict(list)
    lesion_label = {}
    for entry in dataset:
        base_id = get_base_lesion_id(entry)
        lesion_rows[base_id].append(entry)
        lesion_label[base_id] = entry["label"]

    base_ids = sorted(lesion_rows.keys())
    labels = [lesion_label[bid] for bid in base_ids]

    # Split 1: train vs (val + test).
    train_ids, rest_ids, _, rest_labels = train_test_split(
        base_ids,
        labels,
        train_size=train_frac,
        random_state=seed,
        stratify=labels,
    )

    # Split 2: dentro de "rest", separar val y test en la proporción
    
    rest_frac = 1.0 - train_frac
    val_ids, test_ids = train_test_split(
        rest_ids,
        train_size=val_frac / rest_frac,
        random_state=seed,
        stratify=rest_labels,
    )

    def rows_for(ids):
        return [row for bid in ids for row in lesion_rows[bid]]

    return rows_for(train_ids), rows_for(val_ids), rows_for(test_ids)



# 3. Muestras en formato MONAI (rutas absolutas)

def to_monai_samples(entries: list[dict]) -> list[dict]:
    
    return [
        {
            "flair": str(BASE_DIR / entry["flair"]),
            "swi": str(BASE_DIR / entry["swi"]),
            "label": entry["label"],
            "lesion_id": entry["lesion_id"],
            "patient_id": entry["patient_id"],
        }
        for entry in entries
    ]


# 4. Transformaciones MONAI


def get_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["flair", "swi"]),
            EnsureChannelFirstd(keys=["flair", "swi"]),
            ScaleIntensityd(keys=["flair", "swi"]),
            ConcatItemsd(keys=["flair", "swi"], name="image", dim=0),
            DeleteItemsd(keys=["flair", "swi"]),
            ToTensord(keys=["image", "label"]),
        ]
    )



# 5. Main: construir datasets/dataloaders y comprobar un batch


def main() -> None:
    print("=" * 60)
    print("test_monai_dataset.py — split + carga MONAI")
    print("=" * 60)

    dataset = load_dataset()
    train_entries, val_entries, test_entries = split_dataset(dataset)

    print(f"\nTotal de muestras en dataset.json: {len(dataset)}")
    print(f"  Train: {len(train_entries)} muestras")
    print(f"  Val:   {len(val_entries)} muestras")
    print(f"  Test:  {len(test_entries)} muestras")

    transforms = get_transforms()

    train_ds = Dataset(data=to_monai_samples(train_entries), transform=transforms)
    val_ds = Dataset(data=to_monai_samples(val_entries), transform=transforms)
    test_ds = Dataset(data=to_monai_samples(test_entries), transform=transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Smoke test: cargar un único batch de entrenamiento y comprobar
    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]

    print("\nPrimer batch de train_loader:")
    print(f"  images.shape  : {tuple(images.shape)}")
    assert images.shape[1:] == (2, 28, 28, 28), (
        f"Shape de parche inesperada: {tuple(images.shape[1:])} "
        f"(algún .nii.gz del batch no mide 28x28x28 o tiene un canal de más/menos)"
    )
    print(f"  labels.shape  : {tuple(labels.shape)}")
    print(f"  images.dtype  : {images.dtype}")
    print(f"  images.device : {images.device}")
    print(f"\n  labels          : {labels}")
    print(f"  batch['lesion_id']: {batch['lesion_id']}")


if __name__ == "__main__":
    main()
