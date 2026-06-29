import numpy as np, matplotlib.pyplot as plt, os, random

refined_dir = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/LesionGAN_trial_junio/refined_patches'
infer_dir   = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/LesionGAN_Native/infer'

files  = [f for f in os.listdir(refined_dir) if f.endswith('_refined.npy')]
sample = random.sample(files, min(6, len(files)))

n_slices = 4
fig, axes = plt.subplots(len(sample) * 3, n_slices, figsize=(n_slices * 3, len(sample) * 9))

for row, fname in enumerate(sample):
    name    = fname.replace('_refined.npy', '')
    refined = np.load(os.path.join(refined_dir, fname))
    healthy = np.load(os.path.join(infer_dir, f'{name}_healthy.npy'))
    diff    = healthy - refined
    vmin, vmax = healthy.min(), healthy.max()
    diff_max = float(diff.max())

    for z in range(n_slices):
        r = row * 3
        axes[r,   z].imshow(healthy[z],      cmap='gray',  vmin=vmin,  vmax=vmax)
        axes[r,   z].set_title(f'Sano Z={z}', fontsize=7)
        axes[r,   z].axis('off')

        axes[r+1, z].imshow(refined[z],        cmap='gray',  vmin=vmin,  vmax=vmax)
        axes[r+1, z].set_title(f'sCMB Z={z}',  fontsize=7)
        axes[r+1, z].axis('off')

        axes[r+2, z].imshow(diff[z],           cmap='Blues', vmin=0, vmax=max(0.1, diff_max))
        axes[r+2, z].set_title(f'Diff Z={z} max={diff_max:.2f}', fontsize=7)
        axes[r+2, z].axis('off')

plt.tight_layout()
out_path = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/LesionGAN_trial_junio/lesiongan_trial_junio_slices.png'
plt.savefig(out_path, dpi=120)
print(f'Guardada en {out_path}')
