import os
import shutil
import random
import numpy as np
import nibabel as nib
from skimage.measure import label
from tqdm import tqdm

# =========================
# CONFIGURACIÓN
# =========================
BASE_DIR = "/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_raw"
DIR_201 = os.path.join(BASE_DIR, "Dataset201_SyntheticCMB")
DIR_203 = os.path.join(BASE_DIR, "Dataset203_LesionGAN")
OUTPUT_DIR = os.path.join(BASE_DIR, "Dataset204_GaussGANMix")

# Semilla para reproducibilidad del submuestreo
random.seed(42)

# Muestreo objetivo por dataset
TARGET_LOW = 80
TARGET_HIGH = 20

# =========================
# FUNCIONES
# =========================
def count_lesions(label_path):
    """Carga una máscara NIfTI y cuenta el número de blobs 3D independientes."""
    data = nib.load(label_path).get_fdata()
    mask = (data > 0).astype(np.uint8)
    _, num_lesions = label(mask, return_num=True)
    return num_lesions

def analyze_and_sample(dataset_dir, dataset_name):
    """Analiza las etiquetas, estratifica por carga lesional y devuelve una muestra aleatoria."""
    labels_dir = os.path.join(dataset_dir, "labelsTr")
    
    low_burden = []
    high_burden = []
    
    files = [f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")]
    
    print(f"\nAnalizando {dataset_name}...")
    for f in tqdm(files):
        path = os.path.join(labels_dir, f)
        n_lesions = count_lesions(path)
        
        if 1 <= n_lesions <= 10:
            low_burden.append(f)
        elif 11 <= n_lesions <= 30:
            high_burden.append(f)

    print(f"[{dataset_name}] Encontrados: {len(low_burden)} Carga Baja | {len(high_burden)} Carga Alta")
    
    # Validar que existan suficientes imágenes para el muestreo
    if len(low_burden) < TARGET_LOW:
        print(f"Advertencia: No hay suficientes casos de carga baja en {dataset_name}. Se usarán todos ({len(low_burden)}).")
        sampled_low = low_burden
    else:
        sampled_low = random.sample(low_burden, TARGET_LOW)
        
    if len(high_burden) < TARGET_HIGH:
        print(f"Advertencia: No hay suficientes casos de carga alta en {dataset_name}. Se usarán todos ({len(high_burden)}).")
        sampled_high = high_burden
    else:
        sampled_high = random.sample(high_burden, TARGET_HIGH)

    # Para evitar colisiones de nombres al mezclar, devolveremos también un prefijo
    prefix = "D201_" if "201" in dataset_name else "D203_"
    
    return sampled_low + sampled_high, prefix

def copy_sampled_cases(sampled_files, prefix, source_dir, dest_dir):
    """Copia imágenes y etiquetas correspondientes a los casos seleccionados, añadiendo un prefijo."""
    src_labels = os.path.join(source_dir, "labelsTr")
    src_images = os.path.join(source_dir, "imagesTr")
    
    dst_labels = os.path.join(dest_dir, "labelsTr")
    dst_images = os.path.join(dest_dir, "imagesTr")
    
    for f in sampled_files:
        # Rutas de origen
        src_lbl_path = os.path.join(src_labels, f)
        
        # En nnUNetv2 las imágenes tienen el identificador de canal _0000 al final
        img_filename = f.replace(".nii.gz", "_0000.nii.gz")
        src_img_path = os.path.join(src_images, img_filename)
        
        # Rutas de destino con prefijo para evitar que casos con el mismo nombre se sobreescriban
        dst_lbl_path = os.path.join(dst_labels, prefix + f)
        dst_img_path = os.path.join(dst_images, prefix + img_filename)
        
        # Copiar
        shutil.copy2(src_lbl_path, dst_lbl_path)
        shutil.copy2(src_img_path, dst_img_path)

# =========================
# MAIN
# =========================
# Crear estructura de salida
os.makedirs(os.path.join(OUTPUT_DIR, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labelsTr"), exist_ok=True)

# 1. Analizar y muestrear D201
sampled_201, prefix_201 = analyze_and_sample(DIR_201, "Dataset201_SyntheticCMB")

# 2. Analizar y muestrear D203
sampled_203, prefix_203 = analyze_and_sample(DIR_203, "Dataset203_LesionGAN")

# 3. Copiar archivos al nuevo dataset
print("\nCopiando archivos al Dataset204_GaussGANMix...")
copy_sampled_cases(sampled_201, prefix_201, DIR_201, OUTPUT_DIR)
copy_sampled_cases(sampled_203, prefix_203, DIR_203, OUTPUT_DIR)

print("\nProceso finalizado con éxito.")
print(f"Dataset generado en: {OUTPUT_DIR}")
print("Nota: Recuerda generar el archivo dataset.json para el nuevo Dataset204.")
