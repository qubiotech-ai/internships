"""
LesionGAN 3D — Opción B+perturbación, Espacio Nativo Anisotrópico (1×1×4mm)
=============================================================================
Parche: (PATCH_Z=4, PATCH_XY=16, PATCH_XY=16) en espacio nativo nibabel (X,Y,Z)
        guardado como (Z,Y,X) tras transpose(2,1,0) en extracción
Máscara: (MASK_Z=3, MASK_XY=9, MASK_XY=9)
"""

import os, json, shutil, argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nilearn.image import resample_img as nilearn_resample
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

class Config:
    UNET_TRAIN_DIR = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/LesionGAN_Native/train'
    UNET_INFER_DIR = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/LesionGAN_Native/infer'
    BASE_OUT       = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/LesionGAN'
    CHECKPOINT_DIR = os.path.join(BASE_OUT, 'checkpoints')
    LOG_DIR        = os.path.join(BASE_OUT, 'logs')
    SAMPLES_DIR    = os.path.join(BASE_OUT, 'samples')
    REFINED_DIR    = os.path.join(BASE_OUT, 'refined_patches')
    NNUNET_BASE    = '/media/qubiotech/storage/DEEPLEARN/datasets/Neuro/CMB/ADNI_dataset/nnUNet_raw'
    NNUNET_IMGS    = os.path.join(NNUNET_BASE, 'Dataset203_LesionGAN/imagesTr')
    NNUNET_LBLS    = os.path.join(NNUNET_BASE, 'Dataset203_LesionGAN/labelsTr')

    # Tamaños — parches en disco tienen shape (PATCH_Z, PATCH_XY, PATCH_XY)
    PATCH_Z    = 4
    PATCH_XY   = 16
    MASK_Z     = 3
    MASK_XY    = 9

    # Arquitectura
    Z_DIM        = 64
    G_FEATURES   = 16
    D_FEATURES   = (32, 64, 128)
    PERTURB_SCALE = 0.10

    # Entrenamiento
    BATCH_SIZE   = 32
    NUM_EPOCHS   = 500
    LR_G         = 1e-4
    LR_D         = 2e-4
    BETA1        = 0.0
    BETA2        = 0.9
    NUM_WORKERS  = 4
    PRECISION    = '16-mixed'
    SEED         = 42
    D_TRAIN_FREQ = 1

    # Pérdidas
    LAMBDA_BORDER   = 3.0
    LAMBDA_VOLUME   = 1.0
    LAMBDA_SPARSITY = 1.0
    LAMBDA_CIRCULAR = 5.0
    LAMBDA_GP       = 10.0

    # Volumen
    VOL_MIN_MM3 = 10.0
    VOL_MAX_MM3 = 40.0

cfg = Config()


# =============================================================================
# UTILIDADES
# =============================================================================

def fix_shape(patch):
    target = (cfg.PATCH_Z, cfg.PATCH_XY, cfg.PATCH_XY)
    if patch.shape == target:
        return patch
    for i in range(3):
        if patch.shape[i] == cfg.PATCH_Z:
            p = np.moveaxis(patch, i, 0)
            if p.shape == target:
                return p
    raise ValueError(f"No se puede corregir shape {patch.shape} a {target}")


def apply_mask_to_patch(mask_small, patch):
    target_size = (cfg.PATCH_Z, cfg.PATCH_XY, cfg.PATCH_XY)
    mask_full = F.interpolate(mask_small, size=target_size,
                              mode='trilinear', align_corners=False)
    lesion_depth = (1.0 - mask_full) * 1.5
    result = patch - lesion_depth
    return result


def make_sphere_mask(radius_mm=None):
    if radius_mm is None:
        radius_mm = np.random.uniform(1.5, 3.5)
    z = (np.arange(cfg.PATCH_Z)  - (cfg.PATCH_Z  - 1) / 2.0) * 4.0
    y = (np.arange(cfg.PATCH_XY) - (cfg.PATCH_XY - 1) / 2.0) * 1.0
    x = (np.arange(cfg.PATCH_XY) - (cfg.PATCH_XY - 1) / 2.0) * 1.0
    ZZ, YY, XX = np.meshgrid(z, y, x, indexing='ij')
    dist = np.sqrt(ZZ**2 + YY**2 + XX**2)
    sigma = 1.5
    mask = 1.0 - np.clip((dist - radius_mm) / sigma + 0.5, 0.0, 1.0)
    return mask.astype(np.float32)


# =============================================================================
# DATASET
# =============================================================================

class LesionGANDataset(Dataset):

    def __init__(self, train_dir: str, augment: bool = True):
        self.train_dir = train_dir
        self.augment   = augment

        self.healthy_files = sorted([f for f in os.listdir(train_dir)
                                     if f.endswith('_healthy.npy')])
        self.mask_files    = sorted([f for f in os.listdir(train_dir)
                                     if f.endswith('_mask.npy')
                                     and '_target' not in f])
        self.real_files    = sorted([f for f in os.listdir(train_dir)
                                     if f.endswith('_target.npy')])

        if not self.healthy_files or not self.real_files or not self.mask_files:
            raise RuntimeError(
                f"Faltan parches en {train_dir}.\n"
                f"healthy={len(self.healthy_files)}, "
                f"mask={len(self.mask_files)}, real={len(self.real_files)}"
            )

        n = min(len(self.healthy_files), len(self.mask_files), len(self.real_files))
        self.healthy_files = self.healthy_files[:n]
        self.mask_files    = self.mask_files[:n]
        self.real_files    = self.real_files[:n]
        print(f"  LesionGAN Dataset (perturbación sobre esfera sintética): {n} pares")

    def __len__(self):
        return len(self.healthy_files)

    def _augment(self, *patches):
        for axis in [1, 2]:
            if np.random.random() > 0.5:
                patches = tuple(np.flip(p, axis=axis).copy() for p in patches)
        k = np.random.randint(0, 4)
        if k > 0:
            patches = tuple(np.rot90(p, k=k, axes=(1, 2)).copy() for p in patches)
        return patches

    def __getitem__(self, idx):
        h = fix_shape(np.load(os.path.join(self.train_dir,
                              self.healthy_files[idx])).astype(np.float32))
        m = make_sphere_mask()

        r_idx = np.random.randint(0, len(self.real_files))
        r = fix_shape(np.load(os.path.join(self.train_dir,
                              self.real_files[r_idx])).astype(np.float32))

        if self.augment:
            h, m = self._augment(h, m)
            r,   = self._augment(r)

        vol_vox  = float((1.0 - m).sum())
        vol_norm = float(np.clip(
            (np.log(max(vol_vox, 1.0)) - np.log(cfg.VOL_MIN_MM3)) /
            (np.log(cfg.VOL_MAX_MM3) - np.log(cfg.VOL_MIN_MM3)),
            0.0, 1.0
        ))

        return {
            'healthy':   torch.from_numpy(h[np.newaxis]),
            'mask_init': torch.from_numpy(m[np.newaxis]),
            'real':      torch.from_numpy(r[np.newaxis]),
            'vol_norm':  torch.tensor(vol_norm, dtype=torch.float32),
        }


# =============================================================================
# ARQUITECTURAS
# =============================================================================

class Generator3D(nn.Module):

    def __init__(self):
        super().__init__()
        f = cfg.G_FEATURES

        self.healthy_encoder = nn.Sequential(
            nn.Conv3d(1, f//2, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(f//2, f, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(f * 4 * 4 * 4, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mask_encoder = nn.Sequential(
            nn.Conv3d(1, f//2, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(f//2, f, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(f * 4 * 4 * 4, 64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(cfg.Z_DIM + 1 + 64 + 64, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, f * 2 * 1 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(size=(cfg.MASK_Z, 4, 4), mode='trilinear', align_corners=False),
            nn.Conv3d(f*2, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(f),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(cfg.MASK_Z, cfg.MASK_XY, cfg.MASK_XY),
                        mode='trilinear', align_corners=False),
            nn.Conv3d(f, f//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(f//2),
            nn.ReLU(inplace=True),
            nn.Conv3d(f//2, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, vol_norm, healthy, mask_init):
        ctx_h = self.healthy_encoder(healthy)
        ctx_m = self.mask_encoder(mask_init)

        inp  = torch.cat([z, vol_norm, ctx_h, ctx_m], dim=1)
        feat = self.fc(inp)
        feat = feat.view(-1, cfg.G_FEATURES*2, 1, 4, 4)
        pert = self.decoder(feat)

        sz = (cfg.PATCH_Z  - cfg.MASK_Z)  // 2
        sy = (cfg.PATCH_XY - cfg.MASK_XY) // 2
        sx = (cfg.PATCH_XY - cfg.MASK_XY) // 2
        base = mask_init[:, :,
                         sz:sz+cfg.MASK_Z,
                         sy:sy+cfg.MASK_XY,
                         sx:sx+cfg.MASK_XY]

        cmb_region    = 1.0 - base
        perturbation  = pert * cfg.PERTURB_SCALE * cmb_region
        mask_deformed = torch.clamp(cmb_region - perturbation, 0.0, 1.0)

        return mask_deformed


class Discriminator3D(nn.Module):

    def __init__(self):
        super().__init__()
        layers = []
        in_ch  = 1
        for i, feat in enumerate(cfg.D_FEATURES):
            stride = (1,2,2) if i < len(cfg.D_FEATURES) - 1 else 1
            layers += [
                nn.Conv3d(in_ch, feat, kernel_size=(3,4,4), stride=stride,
                          padding=(1,1,1), bias=False),
                nn.InstanceNorm3d(feat) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout3d(0.1) if i < len(cfg.D_FEATURES)-1 else nn.Identity(),
            ]
            in_ch = feat
        layers.append(nn.Conv3d(in_ch, 1, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# =============================================================================
# MÓDULO LIGHTNING
# =============================================================================

class LesionGANModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.G = Generator3D()
        self.D = Discriminator3D()
        self.automatic_optimization = False

    def forward(self, z, vol_norm, healthy, mask_init):
        mask = self.G(z, vol_norm, healthy, mask_init)
        return apply_mask_to_patch(mask, healthy)

    def _border_loss(self, mask):
        b = mask.shape[0]
        border_vals = torch.cat([
            mask[:,:, 0,:,:].reshape(b,-1), mask[:,:,-1,:,:].reshape(b,-1),
            mask[:,:,:, 0,:].reshape(b,-1), mask[:,:,:,-1,:].reshape(b,-1),
            mask[:,:,:,:, 0].reshape(b,-1), mask[:,:,:,:,-1].reshape(b,-1),
        ], dim=1)
        return (1.0 - border_vals).abs().mean()

    def _volume_loss(self, mask, vol_norm):
        n_vox = cfg.MASK_Z * cfg.MASK_XY * cfg.MASK_XY
        vol_fake = (1.0 - mask).sum(dim=(1,2,3,4)) / n_vox
        return F.l1_loss(vol_fake, vol_norm.squeeze())

    def _sparsity_loss(self, mask):
        return F.relu((1.0 - mask).mean() - 0.05)

    def _circularity_loss(self, mask):
        B = mask.shape[0]
        loss = torch.zeros(1, device=mask.device)
        ny, nx = mask.shape[3], mask.shape[4]
        gy = torch.linspace(-1, 1, ny, device=mask.device)
        gx = torch.linspace(-1, 1, nx, device=mask.device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')

        for b in range(B):
            per_z  = (1.0 - mask[b,0]).sum(dim=(-1,-2))
            best_z = per_z.argmax()
            sl     = 1.0 - mask[b, 0, best_z]
            total  = sl.sum() + 1e-6
            if total < 2:
                continue
            cy = (sl * grid_y).sum() / total
            cx = (sl * grid_x).sum() / total
            dy = grid_y - cy
            dx = grid_x - cx
            cov_yy = (sl * dy * dy).sum() / total
            cov_xx = (sl * dx * dx).sum() / total
            cov_yx = (sl * dy * dx).sum() / total
            trace  = cov_yy + cov_xx
            det    = cov_yy * cov_xx - cov_yx * cov_yx
            disc   = torch.sqrt(torch.clamp((trace/2)**2 - det, min=1e-8))
            lam1   = trace/2 + disc
            lam2   = trace/2 - disc
            ratio  = torch.sqrt(torch.clamp(lam2 / (lam1 + 1e-6), min=0.0))
            loss   = loss + F.relu(0.5 - ratio)
        return loss / B

    def _gradient_penalty(self, real, fake):
        B     = real.shape[0]
        alpha = torch.rand(B, 1, 1, 1, 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
        d_interp = self.D(interp)
        grads = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True
        )[0]
        grads_norm = grads.reshape(B, -1).norm(2, dim=1)
        return ((grads_norm - 1) ** 2).mean()

    def training_step(self, batch, batch_idx):
        opt_G, opt_D = self.optimizers()
        healthy   = batch['healthy']
        mask_init = batch['mask_init']
        real      = batch['real']
        vol_norm  = batch['vol_norm'].unsqueeze(1)
        B = healthy.shape[0]
        z = torch.randn(B, cfg.Z_DIM, device=self.device)

        # ---- Discriminador WGAN-GP ----
        opt_D.zero_grad()
        with torch.no_grad():
            mask_d = self.G(z, vol_norm, healthy, mask_init)
            fake_d = apply_mask_to_patch(mask_d, healthy)

        l_wass = self.D(fake_d.detach()).mean() - self.D(real).mean()
        l_gp   = self._gradient_penalty(real, fake_d)
        l_D    = l_wass + cfg.LAMBDA_GP * l_gp
        self.manual_backward(l_D)
        nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
        opt_D.step()

        # ---- Generador ----
        if batch_idx % cfg.D_TRAIN_FREQ == 0:
            opt_G.zero_grad()
            mask = self.G(z, vol_norm, healthy, mask_init)
            fake = apply_mask_to_patch(mask, healthy)

            l_adv      = -self.D(fake).mean()
            l_border   = self._border_loss(mask)
            l_volume   = self._volume_loss(mask, vol_norm)
            l_sparsity = self._sparsity_loss(mask)
            l_circular = self._circularity_loss(mask)
            l_G = (l_adv
                   + cfg.LAMBDA_BORDER   * l_border
                   + cfg.LAMBDA_VOLUME   * l_volume
                   + cfg.LAMBDA_SPARSITY * l_sparsity
                   + cfg.LAMBDA_CIRCULAR * l_circular)

            self.manual_backward(l_G)
            nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
            opt_G.step()
        else:
            with torch.no_grad():
                mask = self.G(z, vol_norm, healthy, mask_init)
                fake = apply_mask_to_patch(mask, healthy)
                l_adv      = -self.D(fake).mean()
                l_border   = self._border_loss(mask)
                l_volume   = self._volume_loss(mask, vol_norm)
                l_sparsity = self._sparsity_loss(mask)
                l_circular = self._circularity_loss(mask)
                l_G = (l_adv
                       + cfg.LAMBDA_BORDER   * l_border
                       + cfg.LAMBDA_VOLUME   * l_volume
                       + cfg.LAMBDA_SPARSITY * l_sparsity
                       + cfg.LAMBDA_CIRCULAR * l_circular)

        if batch_idx % 50 == 0:
            os.makedirs(cfg.SAMPLES_DIR, exist_ok=True)
            ep = self.current_epoch
            with torch.no_grad():
                for bi in range(min(4, B)):
                    np.save(f"{cfg.SAMPLES_DIR}/ep{ep:03d}_{batch_idx:05d}_s{bi}_healthy.npy",
                            healthy[bi,0].cpu().numpy())
                    np.save(f"{cfg.SAMPLES_DIR}/ep{ep:03d}_{batch_idx:05d}_s{bi}_fake.npy",
                            fake[bi,0].cpu().numpy())
                    np.save(f"{cfg.SAMPLES_DIR}/ep{ep:03d}_{batch_idx:05d}_s{bi}_mask.npy",
                            mask[bi,0].cpu().numpy())
                    np.save(f"{cfg.SAMPLES_DIR}/ep{ep:03d}_{batch_idx:05d}_s{bi}_real.npy",
                            real[bi,0].cpu().numpy())

        self.log_dict({
            'train/G_total':    l_G.item(),
            'train/G_adv':      l_adv.item(),
            'train/G_border':   (cfg.LAMBDA_BORDER   * l_border).item(),
            'train/G_volume':   (cfg.LAMBDA_VOLUME   * l_volume).item(),
            'train/G_sparsity': (cfg.LAMBDA_SPARSITY * l_sparsity).item(),
            'train/G_circular': (cfg.LAMBDA_CIRCULAR * l_circular).item(),
            'train/W_dist':     (-l_wass).item(),
            'train/D_gp':       (cfg.LAMBDA_GP * l_gp).item(),
        }, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.G.parameters(),
                                 lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2))
        opt_D = torch.optim.Adam(self.D.parameters(),
                                 lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2))
        return [opt_G, opt_D], []


# =============================================================================
# INFERENCIA
# =============================================================================

def run_inference(checkpoint_path: str):
    from scipy.ndimage import label as scipy_label
    os.makedirs(cfg.REFINED_DIR, exist_ok=True)
    model = LesionGANModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    infer_dir     = cfg.UNET_INFER_DIR
    healthy_files = sorted([f for f in os.listdir(infer_dir)
                             if f.endswith('_healthy.npy')])
    print(f"Aplicando LesionGAN a {len(healthy_files)} parches...")

    with torch.no_grad():
        for count, hf in enumerate(healthy_files):
            name = hf.replace('_healthy.npy', '')
            h    = fix_shape(np.load(os.path.join(infer_dir, hf)).astype(np.float32))
            healthy_t = torch.from_numpy(h[np.newaxis, np.newaxis]).to(device)

            mask_path = os.path.join(infer_dir, f"{name}_mask.npy")
            if not os.path.exists(mask_path):
                continue
            m = fix_shape(np.load(mask_path).astype(np.float32))
            mask_init_t = torch.from_numpy(m[np.newaxis, np.newaxis]).to(device)

            vol_vox  = float((1.0 - m).sum())
            vol_norm = float(np.clip(
                (np.log(max(vol_vox, 1.0)) - np.log(cfg.VOL_MIN_MM3)) /
                (np.log(cfg.VOL_MAX_MM3) - np.log(cfg.VOL_MIN_MM3)),
                0.0, 1.0))
            vol_t = torch.tensor([[vol_norm]], dtype=torch.float32).to(device)

            z    = torch.randn(1, cfg.Z_DIM, device=device)
            mask = model.G(z, vol_t, healthy_t, mask_init_t)

            # Componente conectado central — preserva valores continuos
            mask_np = mask[0,0].cpu().numpy()

            # Invertir: convenio generador es 0=lesión → invertir a 1=lesión
            mask_np_inverted = 1.0 - mask_np

            binary  = (mask_np_inverted > 0.3).astype(np.int32)
            labeled, n_components = scipy_label(binary)
            cz, cy, cx = np.array(mask_np.shape) // 2
            central_label = labeled[cz, cy, cx]

            if central_label > 0:
                central_region  = (labeled == central_label).astype(np.float32)
                mask_continuous = mask_np_inverted * central_region
            else:
                mask_continuous = binary.astype(np.float32)

            # Para apply_mask_to_patch necesita 0=lesión, 1=fondo
            mask_for_apply = 1.0 - mask_continuous
            mask_final = torch.from_numpy(mask_for_apply).unsqueeze(0).unsqueeze(0).to(device)
            fake       = apply_mask_to_patch(mask_final, healthy_t)

            # Guardar máscara con convenio 1=lesión, 0=fondo (shape MASK_Z×MASK_XY×MASK_XY)
            mask_save  = (mask_continuous > 0.3).astype(np.float32)

            np.save(os.path.join(cfg.REFINED_DIR, f"{name}_refined.npy"),
                    fake[0,0].cpu().numpy())
            np.save(os.path.join(cfg.REFINED_DIR, f"{name}_mask.npy"),
                    mask_save)

            meta_src = os.path.join(infer_dir, f"{name}_meta.json")
            meta_dst = os.path.join(cfg.REFINED_DIR, f"{name}_meta.json")
            if os.path.exists(meta_src) and not os.path.exists(meta_dst):
                shutil.copy2(meta_src, meta_dst)

            if (count+1) % 100 == 0:
                print(f"  Procesados: {count+1}/{len(healthy_files)}")

    print(f"\nInferencia completada: {len(healthy_files)} parches → {cfg.REFINED_DIR}")


# =============================================================================
# REINSERCIÓN
# =============================================================================

def run_reinsert():
    """
    Reconstruye imágenes .nii.gz en espacio nativo reinsertando parches.
    La label se construye desde las máscaras LesionGAN (_mask.npy),
    no desde D201.
    """
    os.makedirs(cfg.NNUNET_IMGS, exist_ok=True)
    os.makedirs(cfg.NNUNET_LBLS, exist_ok=True)

    refined_files = sorted([f for f in os.listdir(cfg.REFINED_DIR)
                             if f.endswith('_refined.npy')])
    subjects = {}
    for rf in refined_files:
        name = rf.replace('_refined.npy', '')
        subj = name.rsplit('_cmb', 1)[0]
        subjects.setdefault(subj, []).append(name)

    processed = 0
    for subj, patch_names in subjects.items():
        meta0_path = os.path.join(cfg.REFINED_DIR, f"{patch_names[0]}_meta.json")
        if not os.path.exists(meta0_path):
            continue
        with open(meta0_path) as f:
            meta0 = json.load(f)

        img_path_orig = meta0['img_path_orig']
        if not os.path.exists(img_path_orig):
            continue

        img_nii  = nib.load(img_path_orig)
        img_data = img_nii.get_fdata().astype(np.float32)
        # Label vacía — se construye desde las máscaras LesionGAN
        lbl_data = np.zeros(img_data.shape, dtype=np.uint8)

        for pname in patch_names:
            refined_path = os.path.join(cfg.REFINED_DIR, f"{pname}_refined.npy")
            mask_path    = os.path.join(cfg.REFINED_DIR, f"{pname}_mask.npy")
            meta_path    = os.path.join(cfg.REFINED_DIR, f"{pname}_meta.json")
            if not os.path.exists(refined_path):
                continue

            refined = np.load(refined_path)
            mask    = np.load(mask_path)
            with open(meta_path) as f:
                meta = json.load(f)

            refined_denorm = refined * meta['healthy_std'] + meta['healthy_mean']
            refined_xyz    = refined_denorm.transpose(2, 1, 0)  # (X, Y, Z)

            # Upsampling de máscara: (MASK_Z, MASK_XY, MASK_XY) → (PATCH_Z, PATCH_XY, PATCH_XY)
            mask_t   = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            mask_up  = F.interpolate(mask_t,
                                     size=(cfg.PATCH_Z, cfg.PATCH_XY, cfg.PATCH_XY),
                                     mode='trilinear', align_corners=False)
            mask_bin = (mask_up[0,0].numpy() > 0.3).astype(np.uint8)
            mask_xyz = mask_bin.transpose(2, 1, 0)  # (X, Y, Z)

            bbox = meta['bbox']
            x_s, x_e = bbox['x'][0], bbox['x'][1]
            y_s, y_e = bbox['y'][0], bbox['y'][1]
            z_s, z_e = bbox['z'][0], bbox['z'][1]

            vx_s = max(0, x_s); vx_e = min(x_e, img_data.shape[0])
            vy_s = max(0, y_s); vy_e = min(y_e, img_data.shape[1])
            vz_s = max(0, z_s); vz_e = min(z_e, img_data.shape[2])
            px_s = vx_s-x_s; px_e = px_s+(vx_e-vx_s)
            py_s = vy_s-y_s; py_e = py_s+(vy_e-vy_s)
            pz_s = vz_s-z_s; pz_e = pz_s+(vz_e-vz_s)

            img_data[vx_s:vx_e, vy_s:vy_e, vz_s:vz_e] = \
                refined_xyz[px_s:px_e, py_s:py_e, pz_s:pz_e]
            lbl_data[vx_s:vx_e, vy_s:vy_e, vz_s:vz_e] = np.maximum(
                lbl_data[vx_s:vx_e, vy_s:vy_e, vz_s:vz_e],
                mask_xyz[px_s:px_e, py_s:py_e, pz_s:pz_e]
            )

        nib.save(nib.Nifti1Image(img_data, img_nii.affine, img_nii.header),
                 os.path.join(cfg.NNUNET_IMGS, f"{subj}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(lbl_data, img_nii.affine, img_nii.header),
                 os.path.join(cfg.NNUNET_LBLS, f"{subj}.nii.gz"))

        processed += 1
        if processed % 20 == 0:
            print(f"  Reconstruidos: {processed}...")

    print(f"\nReinserción completada: {processed} volúmenes → {cfg.NNUNET_IMGS}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(cfg.SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',       default='train',
                        choices=['train', 'infer', 'reinsert'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--resume',     default=None)
    args = parser.parse_args()

    if args.mode == 'infer':
        if not args.checkpoint:
            raise ValueError("--mode infer requiere --checkpoint")
        run_inference(args.checkpoint)
        return
    if args.mode == 'reinsert':
        run_reinsert()
        return

    for d in [cfg.CHECKPOINT_DIR, cfg.LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    ds     = LesionGANDataset(cfg.UNET_TRAIN_DIR, augment=True)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True,
                        drop_last=True)
    model  = LesionGANModule()

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.CHECKPOINT_DIR, filename='lesiongan-{epoch:03d}',
        save_top_k=3, save_last=True,
        monitor='train/G_border_epoch', mode='min', every_n_epochs=10,
    )
    logger  = TensorBoardLogger(save_dir=cfg.LOG_DIR, name='lesiongan')
    trainer = pl.Trainer(
        max_epochs=cfg.NUM_EPOCHS, accelerator='gpu', devices=1,
        precision=cfg.PRECISION,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval='epoch')],
        logger=logger, log_every_n_steps=5, enable_progress_bar=True,
    )

    print("=" * 60)
    print("LESIONGAN 3D — Perturbación sobre esfera sintética")
    print("=" * 60)
    print(f"  Train dir     : {cfg.UNET_TRAIN_DIR}")
    print(f"  Pares         : {len(ds)}")
    print(f"  Parche nativo : ({cfg.PATCH_Z},{cfg.PATCH_XY},{cfg.PATCH_XY}) | "
          f"Máscara: ({cfg.MASK_Z},{cfg.MASK_XY},{cfg.MASK_XY})")
    print(f"  PERTURB_SCALE : {cfg.PERTURB_SCALE}")
    print(f"  Épocas        : {cfg.NUM_EPOCHS} | Batch: {cfg.BATCH_SIZE}")
    print(f"  λ_border={cfg.LAMBDA_BORDER} | λ_volume={cfg.LAMBDA_VOLUME} | "
          f"λ_sparsity={cfg.LAMBDA_SPARSITY} | λ_circular={cfg.LAMBDA_CIRCULAR}")
    print(f"  LR_G={cfg.LR_G} | LR_D={cfg.LR_D} | D_FREQ=1/{cfg.D_TRAIN_FREQ}")
    print(f"  BLEND_ALPHA   : {cfg.BLEND_ALPHA}  (blend p05_global en reinserción)")
    print(f"\n  tensorboard --logdir {cfg.LOG_DIR}")
    print("=" * 60)

    trainer.fit(model, train_dataloaders=loader, ckpt_path=args.resume)
    print(f"\nEntrenamiento completado.")
    print(f"  python train_lesiongan_500epochs.py --mode infer \\")
    print(f"    --checkpoint {cfg.CHECKPOINT_DIR}/last.ckpt")


if __name__ == '__main__':
    main()
