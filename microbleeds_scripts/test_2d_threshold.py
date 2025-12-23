import os
import numpy as np
import nibabel as nib

# --- CONFIGURACIÓN ---
# Carpeta donde están los .npz del Ensemble 2D (5 folds)
input_folder = r'Ensemble_Input/2D_AllFolds'
output_folder = r'Resultados_2D_Ensemble_Thresh08'

# TU NUEVO UMBRAL
THRESHOLD = 0.8

os.makedirs(output_folder, exist_ok=True)

print(f"Generando 2D Ensemble (Umbral {THRESHOLD}) con corrección de ejes...")

files = [f for f in os.listdir(input_folder) if f.endswith('.npz')]

for f in files:
    case_id = f.replace('.npz', '')
    
    try:
        # Rutas
        npz_path = os.path.join(input_folder, f)
        # Usamos el nifti 2D original como molde geométrico
        ref_path = os.path.join(input_folder, case_id + '.nii.gz') 
        
        if not os.path.exists(ref_path):
            print(f"Saltando {case_id}: No encuentro el NIfTI de referencia.")
            continue

        # 1. Cargar NIfTI de referencia para saber la forma CORRECTA
        img_ref = nib.load(ref_path)
        target_shape = img_ref.shape  # Ej: (512, 512, 35)

        # 2. Cargar Probabilidades (.npz)
        prob = np.load(npz_path)['probabilities']
        p_lesion = prob[1] # Clase 1
        
        # 3. Aplicar Umbral
        seg = (p_lesion > THRESHOLD).astype(np.uint8)
        
        # 4. --- CORRECCIÓN DE GEOMETRÍA ---
        # Si la forma no coincide con la referencia, intentamos transponer
        if seg.shape != target_shape:
            # Caso típico: Z está al principio en el npz (35, 512, 512) 
            # pero al final en el nifti (512, 512, 35)
            if seg.shape == (target_shape[2], target_shape[1], target_shape[0]):
                seg = np.transpose(seg, (2, 1, 0))
                
            # Verificación final
            if seg.shape != target_shape:
                 print(f"⚠️ {case_id}: ¡Error grave! Formas incompatibles: Pred{seg.shape} vs Ref{target_shape}")
                 continue

        # 5. Guardar
        img_out = nib.Nifti1Image(seg, img_ref.affine, img_ref.header)
        nib.save(img_out, os.path.join(output_folder, case_id + '.nii.gz'))
        print(f"Procesado: {case_id}")

    except Exception as e:
        print(f"Error en {case_id}: {e}")

print("¡Hecho! Ahora los ejes deberían coincidir.")
