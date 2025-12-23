import os
import numpy as np
import nibabel as nib
from multiprocessing import Pool

# --- CONFIGURACIÓN ---
folder_2d = r'Ensemble_Input/2D_AllFolds'
folder_3d = r'Ensemble_Input/3D_Fold0'

# Nombre de carpeta descriptivo
output_folder = r'Ensemble_Result_2D3D_Weighted7030_Threshold03'

# 1. PESOS
WEIGHT_2D = 0.7
WEIGHT_3D = 0.3

# 2. UMBRAL (THRESHOLD)
# En lugar de usar argmax (que equivale a 0.5), bajamos el listón.
# Cualquier vóxel con probabilidad combinada > 0.3 será considerado lesión.
THRESHOLD = 0.3

os.makedirs(output_folder, exist_ok=True)

def load_npz(path):
    # Carga las probabilidades [Shape: (Num_Clases, Z, X, Y) o (Num_Clases, X, Y, Z)]
    return np.load(path)['probabilities']

def process_case(case_id):
    try:
        # Rutas
        npz_2d = os.path.join(folder_2d, f"{case_id}.npz")
        npz_3d = os.path.join(folder_3d, f"{case_id}.npz")
        ref_nifti = os.path.join(folder_2d, f"{case_id}.nii.gz")
        
        if not os.path.exists(ref_nifti):
            print(f"Saltando {case_id}: Falta NIfTI de referencia")
            return

        # Cargar probabilidades
        # OJO: nnU-Net suele guardar la clase 1 (lesión) en el índice 1 del primer eje
        prob_2d = load_npz(npz_2d)
        prob_3d = load_npz(npz_3d)

        # Extraer solo el mapa de probabilidad de la lesión (Clase 1)
        p_lesion_2d = prob_2d[1]
        p_lesion_3d = prob_3d[1]

        # --- FUSIÓN PONDERADA ---
        ensemble_prob = (p_lesion_2d * WEIGHT_2D) + (p_lesion_3d * WEIGHT_3D)

        # --- APLICAR UMBRAL (Tu cambio nº 2) ---
        # Generamos la máscara binaria directamente con el umbral 0.3
        seg_raw = (ensemble_prob > THRESHOLD).astype(np.uint8)

        # --- FIX GEOMÉTRICO AUTOMÁTICO (Tu cambio nº 1) ---
        # Tus datos originales venían como (Z, X, Y) -> Índices (0, 1, 2)
        # Necesitamos pasarlos a (Y, X, Z) -> Índices (2, 1, 0) para que coincidan con el GT
        if seg_raw.shape[0] < seg_raw.shape[1]: # Detectar si Z está al principio
            seg_final = np.transpose(seg_raw, (2, 1, 0))
        else:
            seg_final = seg_raw

        # Guardar
        img_ref = nib.load(ref_nifti)
        img_out = nib.Nifti1Image(seg_final, img_ref.affine, img_ref.header)
        
        out_path = os.path.join(output_folder, f"{case_id}.nii.gz")
        nib.save(img_out, out_path)
        print(f"Procesado: {case_id} (Umbral > {THRESHOLD})")

    except Exception as e:
        print(f"Error en {case_id}: {e}")

if __name__ == "__main__":
    # Buscar archivos comunes
    files_2d = {f[:-4] for f in os.listdir(folder_2d) if f.endswith('.npz')}
    files_3d = {f[:-4] for f in os.listdir(folder_3d) if f.endswith('.npz')}
    common = list(files_2d.intersection(files_3d))
    
    print(f"Encontrados {len(common)} casos.")
    print(f"Configuración -> Pesos: 2D({WEIGHT_2D})/3D({WEIGHT_3D}) | Umbral: {THRESHOLD}")
    print("Corrección geométrica (Z,X,Y -> Y,X,Z) activada.")

    with Pool(processes=4) as p:
        p.map(process_case, common)

    print("¡Listo! Carpeta creada correctamente.")
