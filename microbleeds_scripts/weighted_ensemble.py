import os
import numpy as np
import nibabel as nib
from multiprocessing import Pool

# --- CONFIGURACIÓN ---
# Rutas a las carpetas que contienen los archivos .npz y .pkl
folder_2d = 'Ensemble_Input/2D_AllFolds'
folder_3d = 'Ensemble_Input/3D_Fold0'
output_folder = 'Ensemble_Result_Weighted_70_30'

# PESOS: Ajustar
# Si el 3D es muy conservador, dale poco peso para que no "mate" las detecciones del 2D.
WEIGHT_2D = 0.7
WEIGHT_3D = 0.3

# Crear carpeta de salida
os.makedirs(output_folder, exist_ok=True)

def load_npz(path):
    # Carga el array de probabilidades comprimido
    return np.load(path)['probabilities']

def save_segmentation(segmentation, reference_nifti_path, out_path):
    # Carga la geometría del archivo original para guardar la predicción correctamente
    img_ref = nib.load(reference_nifti_path)
    img_out = nib.Nifti1Image(segmentation, img_ref.affine, img_ref.header)
    nib.save(img_out, out_path)

def process_case(case_id):
    try:
        # Archivos de entrada
        npz_2d = os.path.join(folder_2d, f"{case_id}.npz")
        npz_3d = os.path.join(folder_3d, f"{case_id}.npz")
        
        # Archivo de referencia para cabeceras (usamos el nifti original predicho por el 2D)
        ref_nifti = os.path.join(folder_2d, f"{case_id}.nii.gz")
        
        if not os.path.exists(ref_nifti):
            print(f"Skipping {case_id}: No reference nifti found inside 2D folder")
            return

        # 1. Cargar probabilidades (Shape: [Num_Clases, X, Y, Z])
        prob_2d = load_npz(npz_2d)
        prob_3d = load_npz(npz_3d)

        # 2. Fusión Ponderada (Weighted Average)
        ensemble_prob = (prob_2d * WEIGHT_2D) + (prob_3d * WEIGHT_3D)

        # 3. Argmax (Obtener la clase final 0, 1, etc.)
        seg_final = np.argmax(ensemble_prob, axis=0).astype(np.uint8)

        # 4. Guardar
        out_path = os.path.join(output_folder, f"{case_id}.nii.gz")
        save_segmentation(seg_final, ref_nifti, out_path)
        print(f"Done: {case_id}")

    except Exception as e:
        print(f"Error processing {case_id}: {e}")

if __name__ == "__main__":
    # Buscar casos comunes (basado en nombres de archivos .npz)
    files_2d = {f[:-4] for f in os.listdir(folder_2d) if f.endswith('.npz')}
    files_3d = {f[:-4] for f in os.listdir(folder_3d) if f.endswith('.npz')}
    
    cases = list(files_2d.intersection(files_3d))
    print(f"Encontrados {len(cases)} casos para fusionar.")
    print(f"Usando pesos -> 2D: {WEIGHT_2D} | 3D: {WEIGHT_3D}")

    # Ejecución en paralelo (ajustar 'processes' según CPUs, ej: 4 u 8)
    with Pool(processes=4) as p:
        p.map(process_case, cases)

    print("Ensemble finalizado.")
