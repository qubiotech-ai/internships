import os
import nibabel as nib
import numpy as np
from scipy.ndimage import label, generate_binary_structure

# --- CONFIGURACIÓN PARA TU MODELO 2D ---
# Ruta exacta donde están tus predicciones 2D (según tus mensajes anteriores)
input_folder = r"Test_Subset_Ensemble/predictions"

# Carpeta donde guardaremos las versiones mejoradas
output_folder = r"Test_Subset_Ensemble/predictions_2D_Cleaned"

# TAMAÑO MÍNIMO (Vóxeles)
# Como el 2D es muy bueno y tiene poco ruido, podemos ser conservadores.
# Borrar cosas menores a 3 vóxeles quitará las manchas sin tocar lesiones reales.
MIN_SIZE = 3

os.makedirs(output_folder, exist_ok=True)

print(f"--- LIMPIANDO PREDICCIONES 2D (Eliminando objetos < {MIN_SIZE} vox) ---")
print(f"Entrada: {input_folder}")
print(f"Salida:  {output_folder}\n")

if not os.path.exists(input_folder):
    print(f"ERROR: No encuentro la carpeta {input_folder}. Revisa la ruta.")
    exit()

files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]

for f in files:
    try:
        path_in = os.path.join(input_folder, f)
        img = nib.load(path_in)
        data = img.get_fdata().astype(np.uint8)
        
        # 1. Detectar manchas independientes
        structure = generate_binary_structure(3, 1) 
        labeled_array, num_features_orig = label(data, structure=structure)
        
        if num_features_orig == 0:
            # Si estaba vacío, lo copiamos vacío y seguimos
            nib.save(img, os.path.join(output_folder, f))
            continue

        # 2. Medir tamaño de cada mancha
        sizes = np.bincount(labeled_array.ravel())
        
        # 3. Crear filtro (True si es mayor que MIN_SIZE)
        mask_sizes = sizes > MIN_SIZE
        mask_sizes[0] = 0 # El fondo siempre es 0
        
        # 4. Aplicar filtro
        data_clean = mask_sizes[labeled_array].astype(np.uint8)
        
        # Guardar resultado
        path_out = os.path.join(output_folder, f)
        new_img = nib.Nifti1Image(data_clean, img.affine, img.header)
        nib.save(new_img, path_out)
        
        # Reporte
        labeled_array_new, num_features_new = label(data_clean, structure=structure)
        diff = num_features_orig - num_features_new
        
        if diff > 0:
            print(f"{f}: Eliminados {diff} objetos de ruido. (Quedan {num_features_new})")
        else:
            print(f"{f}: Limpio (Sin cambios).")

    except Exception as e:
        print(f"Error en {f}: {e}")

print("\n¡HECHO! Ahora evalúa la carpeta 'Test_Subset_Ensemble/predictions_2D_Cleaned'")
