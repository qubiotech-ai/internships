#!/bin/bash

# --- CONFIGURACIÓN DE RUTAS ---
# Ruta base en tu sistema
BASE_DIR="/media/PORT-DISK/Practicas/MicroBleeds_Generation/ADNI/workdir_ADNI_subset"
INPUT_DIR="$BASE_DIR/raw/positives"
OUTPUT_DIR="$BASE_DIR/synthseg_segmentations"
MASTER_QC="$BASE_DIR/synthseg_calidad_total.csv"

# Límite de imágenes por lote
LIMITE=50
CONTADOR=0

mkdir -p "$OUTPUT_DIR"

# --- BUCLE DE PROCESAMIENTO ---
for img in "$INPUT_DIR"/*.nii.gz; do
    if [ $CONTADOR -ge $LIMITE ]; then
        echo "Límite de $LIMITE imágenes alcanzado."
        break
    fi

    filename=$(basename "$img")
    basename="${filename%.nii.gz}"
    output_name="${basename}_segmentation.nii.gz"
    
    # Nombre del QC temporal DENTRO de la carpeta que Docker ve
    temp_qc_name="qc_temp_${basename}.csv"

    if [ -f "$OUTPUT_DIR/$output_name" ]; then
        echo "Saltando $filename (ya existe)."
        continue
    fi

    echo "[$((CONTADOR+1))/$LIMITE] Procesando: $filename"

    # EJECUCIÓN DOCKER
    # Importante: el QC se guarda en /data/ para que persista al salir el contenedor
    docker run -it --rm \
      -v "$BASE_DIR":/data \
      qubiodkhub/freesurfer:7.4.1 \
      mri_synthseg --i "/data/raw/positives/$filename" \
                   --o "/data/synthseg_segmentations/$output_name" \
                   --parc --robust --fast --threads 1 \
                   --qc "/data/$temp_qc_name"

    # GESTIÓN DEL CSV MAESTRO
    if [ -f "$BASE_DIR/$temp_qc_name" ]; then
        if [ ! -f "$MASTER_QC" ]; then
            # Si no existe el maestro, lo creamos con la cabecera del primero
            cp "$BASE_DIR/$temp_qc_name" "$MASTER_QC"
        else
            # Si existe, añadimos solo la línea de datos (fila 2)
            # Usamos sed para asegurar que no haya problemas de formato
            sed -n '2p' "$BASE_DIR/$temp_qc_name" >> "$MASTER_QC"
        fi
        # Borramos el temporal después de copiarlo
        rm "$BASE_DIR/$temp_qc_name"
    else
        echo "ERROR: No se generó el archivo de calidad para $filename"
    fi

    ((CONTADOR++))
done

echo "Proceso finalizado. Revisa $MASTER_QC para ver las métricas."
