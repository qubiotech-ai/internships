#!/bin/bash

export nnUNet_raw="/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_raw"
export nnUNet_preprocessed="/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_preprocessed"
export nnUNet_results="/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_results"

INPUT_FOLDER="/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_raw/Dataset205_MixCMB/imagesTs"
OUTPUT_FOLDER="/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_results/Dataset205_MixCMB/predicts_from_208_sinPROB_ensemble"

nnUNetv2_predict -i ${INPUT_FOLDER} -o ${OUTPUT_FOLDER} -d 208 -c 3d_fullres_smallpatch -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth -p nnUNetPlans_newspacing
nnUNetv2_evaluate_folder "/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_raw/Dataset205_MixCMB/labelsTs" ${OUTPUT_FOLDER} -djfile "${nnUNet_preprocessed}/Dataset208_RealMixCMB/dataset.json" -pfile "${nnUNet_results}/Dataset208_RealMixCMB/nnUNetTrainer_250epochs__nnUNetPlans_newspacing__3d_fullres_smallpatch/plans.json"
