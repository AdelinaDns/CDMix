import torch
import torch.utils.data as data
import torchnet as tnt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import json
import pickle as pkl
import argparse
import pprint
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from models.stclassifier import PseTae
from dataset import PixelSetData, PixelSetData_preloaded
from learning.focal_loss import FocalLoss
from learning.weight_init import weight_init
from learning.metrics import mIou, confusion_matrix_analysis


# ---------------------------------------------------------------------------
# Mixup helpers
# ---------------------------------------------------------------------------

def mixup_2_dataset(x_pastis, y_pastis, x_slovakia, y_slovakia, lam, device):
    """
    Mixup între PASTIS și Slovakia la nivel de input tensori.
    lam=1   → pur PASTIS
    lam=0   → pur Slovakia
    Schedule: lam scade de la 1→0 pe parcursul epocilor (începem cu PASTIS, adăugăm Slovakia).

    x are structura: [(pixel_set, mask)] sau [pixel_set, mask]
    — recursive_todevice a transformat deja totul în tensori pe device.
    """
    def mix_tensors(a, b):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            # aliniază dimensiunea batch dacă e nevoie (ultimul batch poate diferi)
            min_b = min(a.shape[0], b.shape[0])
            return lam * a[:min_b] + (1 - lam) * b[:min_b]
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return [mix_tensors(ai, bi) for ai, bi in zip(a, b)]
        else:
            return a  # fallback

    x_mixed = mix_tensors(x_pastis, x_slovakia)

    # taie și labelurile la min_batch
    min_b = min(y_pastis.shape[0], y_slovakia.shape[0])
    y_a = y_pastis[:min_b]
    y_b = y_slovakia[:min_b]

    return x_mixed, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def compute_lam(epoch, total_epochs, warmup_epochs=20):
    """
    lam=1 (pur PASTIS) la epoch=1
    lam=0 (pur Slovakia) la epoch=warmup_epochs
    lam=0 constant după warmup_epochs
    """
    if epoch >= warmup_epochs:
        return 0.0
    return 1.0 - (epoch - 1) / (warmup_epochs - 1)
def cutmix_temporal(x_pastis, x_slovakia, lam):
    """
    Primele T1 = round(lam * T) timesteps din PASTIS, restul din Slovakia.
    x shape: (B, T, C, npixel) pentru pixel-set, (B, T, npixel) pentru mask.
    """
    def cut_temporal(a, b):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            min_b = min(a.shape[0], b.shape[0])
            a, b = a[:min_b], b[:min_b]
            T = a.shape[1]
            T1 = round(lam * T)
            out = b.clone()
            out[:, :T1] = a[:, :T1]
            return out
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return [cut_temporal(ai, bi) for ai, bi in zip(a, b)]
        return a

    x_mixed = cut_temporal(x_pastis, x_slovakia)
    min_b = min(x_pastis[0][0].shape[0] if isinstance(x_pastis[0], (list,tuple)) else x_pastis[0].shape[0],
                x_slovakia[0][0].shape[0] if isinstance(x_slovakia[0], (list,tuple)) else x_slovakia[0].shape[0])
    return x_mixed, min_b


def cutmix_bands(x_pastis, x_slovakia, lam):
    """
    C1 = round(lam * C) benzi alese random din PASTIS, restul din Slovakia.
    x shape pixel-set: (B, T, C, npixel)
    """
    def cut_bands(a, b):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            min_b = min(a.shape[0], b.shape[0])
            a, b = a[:min_b], b[:min_b]
            if a.dim() == 4:  # (B, T, C, npixel) — pixel-set
                C = a.shape[2]
                C1 = round(lam * C)
                band_idx = torch.randperm(C)[:C1]
                out = b.clone()
                out[:, :, band_idx, :] = a[:, :, band_idx, :]
            else:
                # mask (B, T, npixel) — nu are benzi, returnează Slovakia
                out = b.clone()
            return out
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return [cut_bands(ai, bi) for ai, bi in zip(a, b)]
        return a

    x_mixed = cut_bands(x_pastis, x_slovakia)
    min_b = min(x_pastis[0][0].shape[0] if isinstance(x_pastis[0], (list,tuple)) else x_pastis[0].shape[0],
                x_slovakia[0][0].shape[0] if isinstance(x_slovakia[0], (list,tuple)) else x_slovakia[0].shape[0])
    return x_mixed, min_b


def cutmix_pixels(x_pastis, x_slovakia, lam):
    """
    P1 = round(lam * npixel) pixeli aleși random din PASTIS, restul din Slovakia.
    x shape pixel-set: (B, T, C, npixel)
    mask shape: (B, T, npixel)
    """
    def cut_pixels(a, b):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            min_b = min(a.shape[0], b.shape[0])
            a, b = a[:min_b], b[:min_b]
            npixel = a.shape[-1]
            P1 = round(lam * npixel)
            pix_idx = torch.randperm(npixel)[:P1]
            out = b.clone()
            out[..., pix_idx] = a[..., pix_idx]
            return out
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return [cut_pixels(ai, bi) for ai, bi in zip(a, b)]
        return a

    x_mixed = cut_pixels(x_pastis, x_slovakia)
    min_b = min(x_pastis[0][0].shape[0] if isinstance(x_pastis[0], (list,tuple)) else x_pastis[0].shape[0],
                x_slovakia[0][0].shape[0] if isinstance(x_slovakia[0], (list,tuple)) else x_slovakia[0].shape[0])
    return x_mixed, min_b

def train_epoch_mixup(model, optimizer, criterion, pastis_loader, slovakia_loader,
                      epoch, device, config):
    """
    Antrenare cu mixup între PASTIS și Slovakia.
    Iterează după loader-ul mai mic (Slovakia), ciclează PASTIS.
    """
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true, y_pred = [], []

    lam = compute_lam(epoch, config['epochs'], config['mixup_warmup_epochs'])
    print(f'  [Mixup] epoch={epoch}, lam={lam:.3f}  '
          f'({"pur PASTIS" if lam==1 else "pur Slovakia" if lam==0 else f"mix {lam:.2f}/{1-lam:.2f}"})')

    pastis_iter = iter(pastis_loader)

    for i, (x_slovakia, y_slovakia) in enumerate(slovakia_loader):
        # ciclează PASTIS dacă e mai scurt
        try:
            x_pastis, y_pastis = next(pastis_iter)
        except StopIteration:
            pastis_iter = iter(pastis_loader)
            x_pastis, y_pastis = next(pastis_iter)

        x_slovakia = recursive_todevice(x_slovakia, device)
        x_pastis   = recursive_todevice(x_pastis,   device)
        y_slovakia  = y_slovakia.to(device)
        y_pastis    = y_pastis.to(device)

        if lam == 1.0:
            # pur PASTIS
            out  = model(x_pastis)
            loss = criterion(out, y_pastis.long())
            y_true.extend(list(map(int, y_pastis)))
            acc_meter.add(out.detach(), y_pastis)        # <-- adaugă
        elif lam == 0.0:
            # pur Slovakia
            out  = model(x_slovakia)
            loss = criterion(out, y_slovakia.long())
            y_true.extend(list(map(int, y_slovakia)))
            acc_meter.add(out.detach(), y_slovakia)      # <-- adaugă
        else:
            mixup_type = config['mixup_type']

            if mixup_type == 'linear':
                x_mixed, y_a, y_b = mixup_2_dataset(
                    x_pastis, y_pastis, x_slovakia, y_slovakia, lam, device)
            elif mixup_type == 'temporal':
                x_mixed, min_b = cutmix_temporal(x_pastis, x_slovakia, lam)
                y_a = y_pastis[:min_b]
                y_b = y_slovakia[:min_b]
            elif mixup_type == 'bands':
                x_mixed, min_b = cutmix_bands(x_pastis, x_slovakia, lam)
                y_a = y_pastis[:min_b]
                y_b = y_slovakia[:min_b]
            elif mixup_type == 'pixels':
                x_mixed, min_b = cutmix_pixels(x_pastis, x_slovakia, lam)
                y_a = y_pastis[:min_b]
                y_b = y_slovakia[:min_b]

            out  = model(x_mixed)
            loss = mixup_criterion(criterion, out, y_a.long(), y_b.long(), lam)
            y_true.extend(list(map(int, y_b)))
            acc_meter.add(out.detach(), y_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = out.detach()
        y_p  = pred.argmax(dim=1).cpu().numpy()
        # taie y_true la dimensiunea batch-ului curent dacă e nevoie
        y_pred.extend(list(y_p[:len(y_p)]))
        loss_meter.add(loss.item())

        if (i + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f},  lam={:.2f}'.format(
                i + 1, len(slovakia_loader),
                loss_meter.value()[0], acc_meter.value()[0], lam))

    return {
        'train_loss':     loss_meter.value()[0],
        'train_accuracy': acc_meter.value()[0],
        'train_IoU':      mIou(y_true, y_pred, n_classes=config['num_classes']),
    }
# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_single_dataset(folder, meanstd_filename, sub_classes, labels_column, config):
    """
    Instantiate a PixelSetData (or preloaded variant) for one dataset root.

    Parameters
    ----------
    folder           : str   - root directory of the dataset
    meanstd_filename : str   - filename of the mean/std pickle inside `folder`
    sub_classes      : list  - integer class indices to keep
    labels_column    : str   - name of the label column (e.g. 'CODE_GROUP')
    config           : dict  - global config (for npixel, geomfeat, preload)

    Remapping note
    --------------
    PASTIS   – PixelSetData remaps sub_classes [1,3,4,...,39] → 0..19 internally.
    Slovakia – sub_classes [0,1,...,19] overlap with the actual CODE_GROUP values
               (1,2,...,19) so no remapping occurs; labels stay as-is.
               Classes absent from data (0,10,11,17) simply never appear in batches
               and are excluded from the confusion matrix automatically.
    """
    meanstd_path = os.path.join(folder, meanstd_filename)
    mean_std = pkl.load(open(meanstd_path, 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None
    cls = PixelSetData_preloaded if config['preload'] else PixelSetData

    return cls(
        folder,
        labels=labels_column,
        npixel=config['npixel'],
        sub_classes=sub_classes,
        norm=mean_std,
        extra_feature=extra,
    )


def build_datasets(config):
    """
    Build every requested dataset and return a list of
        (dataset, name, is_slovakia)
    tuples.  At least one of --dataset_pastis / --dataset_slovakia must be set.

    Remapping policy
    ----------------
    PASTIS   → remap=True  : sub_classes [1,3,4,...,39] → 0..19  (done by PixelSetData)
    Slovakia → remap=False : labels stay as they are in CODE_GROUP (1,2,...,19)
                             num_classes=20 covers the highest label index (19).
    """
    datasets = []

    if config['dataset_pastis']:
        ds = build_single_dataset(
            folder=config['dataset_pastis'],
            meanstd_filename=config['pastis_meanstd'],
            sub_classes=config['pastis_subclasses'],
            labels_column=config['pastis_labels'],
            config=config,
        )
        datasets.append((ds, 'PASTIS', False))
        print(f'[Dataset] PASTIS   loaded from "{config["dataset_pastis"]}"  '
              f'({len(ds)} samples)  [remap=True]')

    if config['dataset_slovakia']:
        ds = build_single_dataset(
            folder=config['dataset_slovakia'],
            meanstd_filename=config['slovakia_meanstd'],
            sub_classes=config['slovakia_subclasses'],
            labels_column=config['slovakia_labels'],
            config=config,
        )
        datasets.append((ds, 'Slovakia', True))
        print(f'[Dataset] Slovakia loaded from "{config["dataset_slovakia"]}"  '
              f'({len(ds)} samples)  [labels as-is, no remap]')

    if not datasets:
        raise ValueError(
            'At least one of --dataset_pastis or --dataset_slovakia must be provided.'
        )

    return datasets


# ---------------------------------------------------------------------------
# Stratified splitting helpers
# ---------------------------------------------------------------------------

def stratified_split_train_val(dataset, seed):
    """
    Return (train_indices, val_indices, all_labels) for one dataset using a
    deterministic stratified 85 / 15 train-val split.

    The split is fully determined by `seed`, so passing the same seed for
    Slovakia always yields the same val partition regardless of whether
    PASTIS is also active.
    """
    all_labels = [int(dataset[i][1]) for i in range(len(dataset))]
    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.15,
        stratify=[all_labels[i] for i in indices],
        random_state=seed,
    )
    return train_idx, val_idx, all_labels


def _print_class_dist(split_name, indices, all_labels):
    labels_in_split = [all_labels[i] for i in indices]
    unique, counts = np.unique(labels_in_split, return_counts=True)
    print(f'    {split_name:5s} class dist: {dict(zip(unique.tolist(), counts.tolist()))}')


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def get_loaders(datasets_with_names, config):
    """
    Build train / val DataLoaders.

    Each dataset is split independently with a stratified 85/15 scheme.
    Slovakia always uses config['rdm_seed'] so its val partition is identical
    regardless of which other datasets are active.
    Subsets are concatenated in deterministic order (PASTIS first, Slovakia second).
    """
    train_subsets, val_subsets = [], []

    for ds, name, is_slovakia in datasets_with_names:
        split_seed = config['rdm_seed']

        train_idx, val_idx, all_labels = stratified_split_train_val(ds, split_seed)

        suffix = '  <- fixed partition' if is_slovakia else ''
        print(f'\n[{name}] Stratified 85/15 train/val split (seed={split_seed}{suffix})')
        print(f'  Train: {len(train_idx)},  Val: {len(val_idx)}')
        for split_name, idx in [('Train', train_idx), ('Val', val_idx)]:
            _print_class_dist(split_name, idx, all_labels)

        train_subsets.append(data.Subset(ds, train_idx))
        val_subsets.append(data.Subset(ds, val_idx))

    combined_train = data.ConcatDataset(train_subsets)
    combined_val   = data.ConcatDataset(val_subsets)

    print(f'\n[Loaders] Combined sizes -> Train: {len(combined_train)},  Val: {len(combined_val)}')

    train_loader = data.DataLoader(
        combined_train, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'],
    )
    val_loader = data.DataLoader(
        combined_val, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'],
    )

    return train_loader, val_loader

def get_loaders_mixup(datasets_with_names, config):
    train_subsets, val_subsets = [], []
    pastis_train_subset   = None
    slovakia_train_subset = None
    slovakia_val_subset   = None

    for ds, name, is_slovakia in datasets_with_names:
        train_idx, val_idx, all_labels = stratified_split_train_val(ds, config['rdm_seed'])

        suffix = '  <- fixed partition' if is_slovakia else ''
        print(f'\n[{name}] Stratified 85/15 train/val split (seed={config["rdm_seed"]}{suffix})')
        print(f'  Train: {len(train_idx)},  Val: {len(val_idx)}')
        for split_name, idx in [('Train', train_idx), ('Val', val_idx)]:
            _print_class_dist(split_name, idx, all_labels)

        train_subset = data.Subset(ds, train_idx)
        val_subset   = data.Subset(ds, val_idx)
        val_subsets.append(val_subset)

        if is_slovakia:
            slovakia_train_subset = train_subset
            slovakia_val_subset   = val_subset
        else:
            pastis_train_subset = train_subset

        train_subsets.append(train_subset)

    combined_train = data.ConcatDataset(train_subsets)
    combined_val   = data.ConcatDataset(val_subsets)
    print(f'\n[Loaders] Combined sizes -> Train: {len(combined_train)},  Val: {len(combined_val)}')

    train_loader = data.DataLoader(
        combined_train, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'],
    )
    val_loader = data.DataLoader(
        combined_val, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'],
    )

    pastis_loader   = None
    slovakia_loader = None
    if config.get('mixup_type') is not None and pastis_train_subset and slovakia_train_subset:
        pastis_loader = data.DataLoader(
            pastis_train_subset, batch_size=config['batch_size'],
            shuffle=True, num_workers=config['num_workers'],
            drop_last=True,
        )
        slovakia_loader = data.DataLoader(
            slovakia_train_subset, batch_size=config['batch_size'],
            shuffle=True, num_workers=config['num_workers'],
            drop_last=True,
        )

    slovakia_val_loader = data.DataLoader(
        slovakia_val_subset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'],
    ) if slovakia_val_subset else None

    eval_loader = slovakia_val_loader if slovakia_val_loader is not None else val_loader

    return train_loader, eval_loader, pastis_loader, slovakia_loader
# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(model, optimizer, config):
    """
    Load weights from --resume_path into `model`.

    Two modes controlled by --resume_mode:

    'resume'   – Full restore: all weights + optimizer state + starting epoch.
                 Use to continue an interrupted run on the exact same task.

    'finetune' – Backbone-only restore: all weights EXCEPT decoder.6 (the
                 final Linear projection to num_classes), which is re-initialised
                 randomly.  Use when adapting to a new dataset / class count.
                 Optimizer state is NOT restored (fresh Adam).

    Model structure (verified from state_dict):
        spatial_encoder.*            – PSE (mlp1, mlp2)
        temporal_encoder.*           – TAE (position_enc, attention, mlp)
        decoder.0 / decoder.1        – Linear(128→64) + BN
        decoder.3 / decoder.4        – Linear(64→32)  + BN
        decoder.6                    – Linear(32→num_classes)  <– dropped in finetune

    Returns
    -------
    start_epoch : int
    best_mIoU   : float
    """
    path = config['resume_path']
    mode = config['resume_mode']

    if not path:
        return 1, 0.0

    if not os.path.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')

    print(f'\n[Checkpoint] Loading "{path}"  (mode={mode})')
    ckpt = torch.load(path, map_location='cpu')
    saved_state = ckpt['state_dict']

    if mode == 'resume':
        model.load_state_dict(saved_state, strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_mIoU   = ckpt.get('best_mIoU', 0.0)
        print(f'[Checkpoint] Resumed from epoch {start_epoch - 1}  '
              f'(best_mIoU so far: {best_mIoU:.4f})')

    elif mode == 'finetune':
        # Drop only the final classification layer so num_classes can differ
        DROPPED_PREFIXES = ('decoder.6.',)

        backbone_state = {
            k: v for k, v in saved_state.items()
            if not any(k.startswith(p) for p in DROPPED_PREFIXES)
        }
        dropped_keys = [k for k in saved_state
                        if any(k.startswith(p) for p in DROPPED_PREFIXES)]

        missing, unexpected = model.load_state_dict(backbone_state, strict=False)

        if hasattr(model, 'decoder') and len(model.decoder) > 6:
            weight_init(model.decoder[6])
            print('[Checkpoint] decoder.6 (final Linear) re-initialised randomly.')
        else:
            print('[Checkpoint] WARNING: could not locate decoder[6] – '
                  'verify the model architecture.')

        print(f'[Checkpoint] Keys loaded        : {len(backbone_state)}')
        print(f'[Checkpoint] Keys dropped       : {dropped_keys}')
        if missing:
            print(f'[Checkpoint] Missing keys       : {missing}')
        if unexpected:
            print(f'[Checkpoint] Unexpected keys    : {unexpected}')

        start_epoch = 1
        best_mIoU   = 0.0
        print('[Checkpoint] Optimizer re-initialised (fine-tune mode).')

    else:
        raise ValueError(f'Unknown --resume_mode "{mode}". Choose "resume" or "finetune".')

    return start_epoch, best_mIoU


# ---------------------------------------------------------------------------
# Encoder freezing
# ---------------------------------------------------------------------------

def freeze_encoder(model, config):
    """
    Freeze spatial_encoder and/or temporal_encoder so only the decoder trains.
    Call AFTER load_checkpoint so loaded weights are preserved.

    Controlled by --freeze_encoder:
        'all'      – freeze spatial_encoder + temporal_encoder
        'spatial'  – freeze spatial_encoder only
        'temporal' – freeze temporal_encoder only
        False      – freeze nothing
    """
    freeze_mode = config.get('freeze_encoder', False)
    if not freeze_mode:
        return

    mode_to_modules = {
        'all':      ['spatial_encoder', 'temporal_encoder'],
        'spatial':  ['spatial_encoder'],
        'temporal': ['temporal_encoder'],
    }

    if freeze_mode not in mode_to_modules:
        raise ValueError(f'--freeze_encoder must be one of {list(mode_to_modules.keys())} or False, got "{freeze_mode}"')

    frozen_modules = mode_to_modules[freeze_mode]
    frozen_params, trainable_params = 0, 0

    for name, param in model.named_parameters():
        if any(name.startswith(m) for m in frozen_modules):
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()

    print(f'\n[Freeze] mode={freeze_mode}:')
    print(f'  Frozen modules       : {frozen_modules}')
    print(f'  Frozen parameters    : {frozen_params:,}')
    print(f'  Trainable parameters : {trainable_params:,}')
# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(conf_mat, class_names, title='Confusion Matrix', normalize=True):
    """
    Plot a seaborn heatmap confusion matrix and return a matplotlib Figure.
    """
    if normalize:
        row_sums = conf_mat.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        matrix = (conf_mat.astype(float) / row_sums * 100).round(1)
        fmt = '.1f'
        cbar_label = 'Recall (%)'
    else:
        matrix = conf_mat
        fmt = 'd'
        cbar_label = 'Count'

    n = len(class_names)
    fig_size = max(10, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.4,
        linecolor='#e0e0e0',
        cbar_kws={'label': cbar_label},
    )
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title(title, fontsize=14, pad=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    return fig


def log_confusion_matrix(conf_mat, present_labels, config, epoch, mode='val'):
    """
    Save normalised + raw confusion matrix figures to disk and log to W&B.

    Parameters
    ----------
    present_labels : list of int – class indices that actually appear in y_true.
                     Used as axis tick labels so there are no empty rows/columns.
                     For Slovakia these are the original CODE_GROUP values (1,2,...,19).
                     For PASTIS these are remapped indices (0..19).
    """
    class_names = [str(i) for i in present_labels]

    fig_norm = plot_confusion_matrix(
        conf_mat, class_names,
        title=f'{mode.capitalize()} Confusion Matrix (epoch {epoch}, normalised)',
        normalize=True,
    )
    fig_raw = plot_confusion_matrix(
        conf_mat, class_names,
        title=f'{mode.capitalize()} Confusion Matrix (epoch {epoch}, counts)',
        normalize=False,
    )

    cm_dir = os.path.join(config['res_dir'], 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    norm_path = os.path.join(cm_dir, f'cm_{mode}_epoch{epoch:04d}_norm.png')
    raw_path  = os.path.join(cm_dir, f'cm_{mode}_epoch{epoch:04d}_raw.png')
    fig_norm.savefig(norm_path, dpi=150)
    fig_raw.savefig(raw_path,  dpi=150)
    plt.close(fig_norm)
    plt.close(fig_raw)

    wandb.log({
        f'{mode}_conf_mat_normalised': wandb.Image(norm_path,
                                                    caption=f'Epoch {epoch} – normalised'),
        f'{mode}_conf_mat_counts':     wandb.Image(raw_path,
                                                    caption=f'Epoch {epoch} – counts'),
        'epoch': epoch,
    })


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, optimizer, criterion, data_loader, device, config):
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true, y_pred = [], []

    for i, (x, y) in enumerate(data_loader):
        y_true.extend(list(map(int, y)))
        x = recursive_todevice(x, device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y.long())
        loss.backward()
        optimizer.step()

        pred = out.detach()
        y_p = pred.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        acc_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % config['display_step'] == 0:
            print('Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}'.format(
                i + 1, len(data_loader),
                loss_meter.value()[0],
                acc_meter.value()[0]))

    return {
        'train_loss':     loss_meter.value()[0],
        'train_accuracy': acc_meter.value()[0],
        'train_IoU':      mIou(y_true, y_pred, n_classes=config['num_classes']),
    }


def evaluation(model, criterion, loader, device, config, mode='val'):
    """
    Run inference on `loader`.

    Returns
    -------
    metrics        : dict
    conf_mat       : np.ndarray  – built only on classes present in y_true,
                                   so no empty rows/columns for missing classes.
    present_labels : list of int – the class indices used as confusion matrix axes.
                                   For Slovakia: original CODE_GROUP values (1,2,...,19).
                                   For PASTIS:   remapped indices (0..19).
    """
    y_true, y_pred = [], []
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()

    for (x, y) in loader:
        y_true.extend(list(map(int, y)))
        x = recursive_todevice(x, device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x)
            loss = criterion(prediction, y)

        acc_meter.add(prediction, y)
        loss_meter.add(loss.item())
        y_p = prediction.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))

    metrics = {
        f'{mode}_accuracy': acc_meter.value()[0],
        f'{mode}_loss':     loss_meter.value()[0],
        f'{mode}_IoU':      mIou(y_true, y_pred, config['num_classes']),
    }

    # Only include classes that actually appear in y_true so the confusion
    # matrix has no empty rows/columns (relevant for Slovakia where labels
    # are not remapped and some indices like 0,10,11,17 don't exist in data).
    present_labels = sorted(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=present_labels)
    return metrics, conf_mat, present_labels


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(config['res_dir'], exist_ok=True)


def checkpoint(log, config):
    with open(os.path.join(config['res_dir'], 'trainlog.json'), 'w') as f:
        json.dump(log, f, indent=4)


def save_results(metrics, conf_mat, config):
    with open(os.path.join(config['res_dir'], 'val_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    pkl.dump(conf_mat, open(os.path.join(config['res_dir'], 'conf_mat.pkl'), 'wb'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config):
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])

    # Every run lives in its own subfolder: <res_dir>/<wandb_name>/
    config['res_dir'] = os.path.join(config['res_dir'], config['wandb_name'])
    prepare_output(config)

    run_args_path = os.path.join(config['res_dir'], 'args_log.json')
    with open(run_args_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f'Run arguments saved to {run_args_path}')

    wandb.init(
        entity='my_team_projects',
        project='PASTIS',
        config=config,
        name=config['wandb_name'],
        resume='allow' if config['resume_mode'] == 'resume' else None,
    )
    for k, v in config.items():
        wandb.run.summary[k] = v

    # ------------------------------------------------------------------
    # Datasets + loaders
    # ------------------------------------------------------------------
    datasets_with_names = build_datasets(config)
    first_ds = datasets_with_names[0][0]
    # train_loader, val_loader = get_loaders(datasets_with_names, config)
    train_loader, val_loader, pastis_loader, slovakia_loader = get_loaders_mixup(datasets_with_names, config)
    # validare că mixup e posibil
    if config.get('mixup_type') is not None:
        if pastis_loader is None or slovakia_loader is None:
            raise ValueError(
                'Mixup necesită ambele dataset-uri active (--dataset_pastis și --dataset_slovakia).')
        print(f'\n[Mixup] Activ: warmup={config["mixup_warmup_epochs"]} epoci, '
              f'lam: 1.0→0.0')
    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    device = torch.device(config['device'])

    model_config = dict(
        input_dim=config['input_dim'],
        mlp1=config['mlp1'],
        pooling=config['pooling'],
        mlp2=config['mlp2'],
        n_head=config['n_head'],
        d_k=config['d_k'],
        mlp3=config['mlp3'],
        dropout=config['dropout'],
        T=config['T'],
        len_max_seq=config['lms'],
        positions=first_ds.date_positions if config['positions'] == 'bespoke' else None,
        mlp4=config['mlp4'],
    )
    if config['geomfeat']:
        model_config.update(with_extra=True, extra_size=4)
    else:
        model_config.update(with_extra=False, extra_size=None)

    model = PseTae(**model_config)
    model = model.to(device)
    model.apply(weight_init)
    print(model.param_ratio())

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = FocalLoss(config['gamma'])

    # ------------------------------------------------------------------
    # Checkpoint loading  (resume / finetune / scratch)
    # ------------------------------------------------------------------
    start_epoch, best_mIoU = load_checkpoint(model, optimizer, config)

    # ------------------------------------------------------------------
    # Encoder freezing  (AFTER checkpoint so loaded weights are preserved)
    # Rebuild optimizer with only trainable params when freezing.
    # ------------------------------------------------------------------
    freeze_encoder(model, config)
    if config.get('freeze_encoder', False):
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=config['lr'])

    wandb.watch(model, criterion, log='all', log_freq=100)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    trainlog     = {}
    model_path   = os.path.join(config['res_dir'], 'model.pth.tar')
    best_conf_mat = None
    best_present_labels = None

    for epoch in range(start_epoch, config['epochs'] + 1):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))

        model.train()
        if config.get('mixup_type') is not None:
            train_metrics = train_epoch_mixup(
                model, optimizer, criterion,
                pastis_loader, slovakia_loader,
                epoch=epoch, device=device, config=config,
            )
        else:
            train_metrics = train_epoch(
                model, optimizer, criterion, train_loader,
                device=device, config=config,
            )
        print('Validation . . .')
        model.eval()
        val_metrics, conf_mat, present_labels = evaluation(
            model, criterion, val_loader, device=device, config=config, mode='val')

        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(
            val_metrics['val_loss'], val_metrics['val_accuracy'], val_metrics['val_IoU']))

        trainlog[epoch] = {**train_metrics, **val_metrics}
        checkpoint(trainlog, config)
        wandb.log({'epoch': epoch, **train_metrics, **val_metrics})

        # Confusion matrix every 10 epochs + always on the final epoch
        if epoch % 10 == 0 or epoch == config['epochs']:
            log_confusion_matrix(conf_mat, present_labels, config,
                                 epoch=epoch, mode='val')

        if val_metrics['val_IoU'] >= best_mIoU:
            best_mIoU           = val_metrics['val_IoU']
            best_conf_mat       = conf_mat.copy()
            best_present_labels = present_labels[:]

            torch.save({
                'epoch':      epoch,
                'best_mIoU':  best_mIoU,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
            }, model_path)

            artifact = wandb.Artifact(
                name='model_best', type='model',
                description=f'Best model at epoch {epoch}, val_IoU={best_mIoU:.4f}',
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    # ------------------------------------------------------------------
    # Final evaluation on best checkpoint
    # ------------------------------------------------------------------
    print('Saving results for best checkpoint (val_IoU={:.4f}) . . .'.format(best_mIoU))
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    best_val_metrics, best_conf_mat, best_present_labels = evaluation(
        model, criterion, val_loader, device=device, config=config, mode='val')
    save_results(best_val_metrics, best_conf_mat, config)

    # Best confusion matrix → dedicated W&B image
    class_names = [str(i) for i in best_present_labels]
    fig_best = plot_confusion_matrix(
        best_conf_mat, class_names,
        title='Best Val Confusion Matrix (normalised)',
        normalize=True,
    )
    best_cm_path = os.path.join(config['res_dir'], 'confusion_matrices',
                                'cm_best_val_norm.png')
    os.makedirs(os.path.dirname(best_cm_path), exist_ok=True)
    fig_best.savefig(best_cm_path, dpi=150)
    plt.close(fig_best)
    wandb.log({'best_val_conf_mat': wandb.Image(best_cm_path,
                                                 caption='Best checkpoint – normalised')})

    _, perf = confusion_matrix_analysis(best_conf_mat)
    print('Overall performance:')
    print('Acc: {},  IoU: {}'.format(perf['Accuracy'], perf['MACRO_IoU']))
    with open(os.path.join(config['res_dir'], 'overall.json'), 'w') as f:
        json.dump(perf, f, indent=4)
    wandb.log({'overall_accuracy': perf['Accuracy'], 'overall_MACRO_IoU': perf['MACRO_IoU']})

    wandb.finish()


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def parse_int_list(s):
    """Parse a bracketed or plain comma-separated int list, e.g. '[1,3,4]' or '1,3,4'."""
    s = s.replace('[', '').replace(']', '').strip()
    return list(map(int, s.split(',')))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # -----------------------------------------------------------------------
    # W&B / experiment identity
    # -----------------------------------------------------------------------
    YEAR = 2020
    parser.add_argument('--wandb_name', default=f'{YEAR}_128_mixup_temporal_20epochs', type=str,
                        help='Name of the W&B run. Results saved under <res_dir>/<wandb_name>/.')

    # -----------------------------------------------------------------------
    # Resume / fine-tune
    # -----------------------------------------------------------------------
    parser.add_argument('--mixup_type', default='temporal', type=str,
                    choices=['linear', 'temporal', 'bands', 'pixels', 'none'],
                    help='Tipul de mixup. Dacă nu e setat, nu se face mixup.')
    parser.add_argument('--mixup_warmup_epochs', default=50, type=int,  
                        help='Numărul de epoci în care lam scade de la 1.0 (pur PASTIS) '
                             'la 0.0 (pur Slovakia). După warmup antrenează doar pe Slovakia.')
    
    parser.add_argument('--resume_path', default='', type=str,#/home/mnegru/Adelina/PASTIS/results/baseline_128/model.pth.tar
                        help='Path to a checkpoint (.pth.tar). Leave empty for scratch.')
    parser.add_argument('--resume_mode', default='finetune', type=str,
                        choices=['resume', 'finetune'],
                        help=(
                            'resume   – restore ALL weights + optimizer + epoch.\n'
                            'finetune – backbone only; decoder.6 re-initialised randomly; '
                            'optimizer reset.'
                        ))
    parser.add_argument('--freeze_encoder', default=False, type=str,
                    help=(
                        'Which encoder module(s) to freeze after loading checkpoint. '
                        'Choices: all | spatial | temporal | False (default: False).'
                    ))

    # -----------------------------------------------------------------------
    # Dataset paths
    # -----------------------------------------------------------------------
    parser.add_argument('--dataset_pastis',
                        default='/home/mnegru/Adelina/Final_data/S2-2017-T31TFM-PixelSet',
                        type=str,
                        help='Root folder of the PASTIS dataset. Set to "" to disable.')
    parser.add_argument('--dataset_slovakia',
                        default=f'/home/mnegru/Adelina/Final_data/PixelSet-Slovakia-{YEAR}',
                        type=str,
                        help='Root folder of the Slovakia dataset. Set to "" to disable.')

    # -----------------------------------------------------------------------
    # PASTIS dataset-specific parameters
    # -----------------------------------------------------------------------
    parser.add_argument('--pastis_meanstd', default='S2-2017-T31TFM-meanstd.pkl', type=str,
                        help='mean/std pickle filename inside --dataset_pastis.')
    parser.add_argument('--pastis_labels', default='label_44class', type=str,
                        help='Label column name for PASTIS.')
    parser.add_argument('--pastis_subclasses',
                        default='[1,3,4,5,6,8,9,12,13,14,16,18,19,23,28,31,33,34,36,39]',
                        type=str,
                        help='PASTIS class indices to keep (remapped to 0..N-1 internally).')
    # -----------------------------------------------------------------------
    # Slovakia dataset-specific parameters
    # -----------------------------------------------------------------------
    parser.add_argument('--slovakia_meanstd', default=f'{YEAR}-Slovakia-meanstd.pkl', type=str,
                        help='mean/std pickle filename inside --dataset_slovakia.')
    parser.add_argument('--slovakia_labels', default='CODE_GROUP', type=str,
                        help='Label column name for Slovakia.')
    parser.add_argument('--slovakia_subclasses',
                        
                        # default='[1,3,4,5,6,8,9,12,13,14,16,18,19,23,28,31,33,34,36,39]',
                        default='[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]',
                        type=str,
                        help=(
                            'Slovakia class indices to keep. Labels are NOT remapped – '
                            'original CODE_GROUP values (e.g. 1,2,...,19) are used directly. '
                            'Set --num_classes 20 and --mlp4 [128,64,32,20] to cover index 19. '
                            'Classes absent from data (e.g. 0,10,11,17) simply never appear '
                            'in batches and are excluded from the confusion matrix automatically.'
                        ))

    # -----------------------------------------------------------------------
    # General set-up
    # -----------------------------------------------------------------------
    parser.add_argument('--res_dir', default='./results',
                        help='Root folder where results are stored. '
                             'Each run is saved under <res_dir>/<wandb_name>/.')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data-loading workers.')
    parser.add_argument('--rdm_seed', default=1, type=int,
                        help='Random seed. Also pins the deterministic Slovakia val split.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Compute device: cuda or cpu.')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Batch interval between training metric printouts.')
    parser.add_argument('--preload', dest='preload', action='store_true',
                        help='Load the entire dataset into RAM at start.')
    parser.set_defaults(preload=True)

    # -----------------------------------------------------------------------
    # Training hyper-parameters
    # -----------------------------------------------------------------------
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--gamma', default=1, type=float,
                        help='Focal-loss gamma.')
    parser.add_argument('--npixel', default=128, type=int,
                        help='Pixels sampled per parcel.')

    # -----------------------------------------------------------------------
    # Architecture – PSE
    # -----------------------------------------------------------------------
    parser.add_argument('--input_dim', default=10, type=int,
                        help='Number of spectral channels.')
    parser.add_argument('--mlp1', default='[10,32,64]', type=str,
                        help='MLP1 layer widths.')
    parser.add_argument('--pooling', default='mean_std', type=str,
                        help='Pixel-embedding pooling strategy.')
    parser.add_argument('--mlp2', default='[128,128]', type=str,
                        help='MLP2 layer widths.')
    parser.add_argument('--geomfeat', default=0, type=int,
                        help='1 to use precomputed geometric features in PSE.')

    # -----------------------------------------------------------------------
    # Architecture – TAE
    # -----------------------------------------------------------------------
    parser.add_argument('--n_head', default=4, type=int,
                        help='Number of attention heads.')
    parser.add_argument('--d_k', default=32, type=int,
                        help='Key/query dimension.')
    parser.add_argument('--mlp3', default='[512,128,128]', type=str,
                        help='MLP3 layer widths.')
    parser.add_argument('--T', default=1000, type=int,
                        help='Positional-encoding period.')
    parser.add_argument('--positions', default='bespoke', type=str,
                        help='Positional encoding: bespoke (date-based) or order.')
    parser.add_argument('--lms', default=None, type=int,
                        help='Max sequence length (required when positions==order).')
    parser.add_argument('--dropout', default=0.2, type=float)

    # -----------------------------------------------------------------------
    # Classifier head
    # -----------------------------------------------------------------------
    parser.add_argument('--num_classes', default=20, type=int,
                        help='Output classes. Must equal len(subclasses) for each dataset.')
    parser.add_argument('--mlp4', default='[128,64,32,20]', type=str,
                        help='MLP4 layer widths. Last value must equal --num_classes.')

    # -----------------------------------------------------------------------
    # Parse + post-process
    # -----------------------------------------------------------------------
    config = parser.parse_args()
    config = vars(config)

    # PATCHED_BY_run_slovakia_transfer_experiments__mixup_none_support
    if config.get('mixup_type') == 'none':
        config['mixup_type'] = None

    for k in list(config.keys()):
        if k.startswith('mlp'):
            config[k] = parse_int_list(config[k])

    config['pastis_subclasses']   = parse_int_list(config['pastis_subclasses'])
    config['slovakia_subclasses'] = parse_int_list(config['slovakia_subclasses'])

    os.makedirs('./logs', exist_ok=True)
    log_path = './logs/args_log.json'
    with open(log_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f'Arguments saved to {log_path} (also copied inside the run folder by main())')

    pprint.pprint(config)
    main(config)