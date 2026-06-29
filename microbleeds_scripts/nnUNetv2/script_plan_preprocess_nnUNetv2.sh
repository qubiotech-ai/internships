#!/bin/bash

# 1. Definir rutas
export nnUNet_raw="/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_raw"
export nnUNet_preprocessed="/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_preprocessed"
export nnUNet_results="/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_results"

DATASET=208
FOLD=4
TRAINER=nnUNetTrainer_250epochs
CONFIG_MODIF=3d_fullres_smallpatch
PLANS_MODIF=nnUNetPlans_newspacing

echo "=========================================="
echo "Iniciando plan y preprocesamiento: $(date)"
echo "Dispositivo CUDA visible:"
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
echo "=========================================="

# 2. Plan & Preprocess
## Cuando queremos modificar partes de la config (ej: smaller patches), lo más adecuado
## es dividir el plan y el preprocess
## Tras el plan, modificar manualmente el json añadiendo la nueva configuración
## y heredando las demás cosas de la antigua
echo "Ejecutando plan para Dataset ${DATASET}..."

#nnUNetv2_plan_and_preprocess -d $DATASET --verify_dataset_integrity

# Tras esto, modificar manualmente el plans json
# Preprocesar config nueva:

#nnUNetv2_preprocess -d $DATASET -c $CONFIG_MODIF -p $PLANS_MODIF

# Entrenar con esta nueva config:

nnUNetv2_train $DATASET $CONFIG_MODIF $FOLD -tr $TRAINER -p $PLANS_MODIF

# Si queremos entrenar con más de una config:

#CONFIG_2d=2d
#PLANS_2d=nnUNetPlans
#nnUNetv2_train $DATASET $CONFIG_2d $FOLD -tr $TRAINER -p $PLANS_2d
