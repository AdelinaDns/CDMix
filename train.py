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
# Dataset builders
# ---------------------------------------------------------------------------

def build_single_dataset(folder, meanstd_filename, sub_classes, labels_column, config):
    """
    Instantiate a PixelSetData (or preloaded variant) for one dataset root.

    Parameters
    ----------
    folder           : str  - root directory of the dataset
    meanstd_filename : str  - filename of the mean/std pickle inside `folder`
    sub_classes      : list - integer class indices to keep
    labels_column    : str  - name of the label column (e.g. 'CODE_GROUP')
    config           : dict - global config (for npixel, geomfeat, preload)
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

    The `is_slovakia` flag is stored so the loader factory can annotate logs
    and future mixup strategies can distinguish dataset origin.
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
        print(f'[Dataset] PASTIS   loaded from "{config["dataset_pastis"]}"  ({len(ds)} samples)')

    if config['dataset_slovakia']:
        ds = build_single_dataset(
            folder=config['dataset_slovakia'],
            meanstd_filename=config['slovakia_meanstd'],
            sub_classes=config['slovakia_subclasses'],
            labels_column=config['slovakia_labels'],
            config=config,
        )
        datasets.append((ds, 'Slovakia', True))
        print(f'[Dataset] Slovakia loaded from "{config["dataset_slovakia"]}"  ({len(ds)} samples)')

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

    Splitting rules
    ---------------
    * Every dataset is split independently with a stratified 85/15 train/val scheme.
    * Slovakia always uses `config['rdm_seed']` as the split seed, so its
      val partition is identical no matter which other datasets are active.
    * PASTIS also uses `config['rdm_seed']`.
    * After splitting, train / val subsets are concatenated across datasets
      (ConcatDataset) in a deterministic order (PASTIS first, then Slovakia),
      ready for future mixup strategies on the training loader.

    Parameters
    ----------
    datasets_with_names : list of (PixelSetData, str, bool)
        The bool flags whether the entry is the Slovakia dataset.

    Returns
    -------
    train_loader, val_loader
    """
    train_subsets, val_subsets = [], []

    for ds, name, is_slovakia in datasets_with_names:
        # Both datasets use config['rdm_seed'].
        # Slovakia determinism is guaranteed because the dataset itself and the
        # seed are always the same, regardless of what else is active.
        split_seed = config['rdm_seed']

        train_idx, val_idx, all_labels = stratified_split_train_val(ds, split_seed)

        suffix = '  <- fixed partition' if is_slovakia else ''
        print(f'\n[{name}] Stratified 85/15 train/val split (seed={split_seed}{suffix})')
        print(f'  Train: {len(train_idx)},  Val: {len(val_idx)}')
        for split_name, idx in [('Train', train_idx), ('Val', val_idx)]:
            _print_class_dist(split_name, idx, all_labels)

        train_subsets.append(data.Subset(ds, train_idx))
        val_subsets.append(data.Subset(ds, val_idx))

    # Concatenate across datasets in deterministic order: PASTIS -> Slovakia
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


# ---------------------------------------------------------------------------
# Confusion matrix plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(conf_mat, class_names, title='Confusion Matrix', normalize=True):
    """
    Plot a seaborn heatmap confusion matrix and return a matplotlib Figure.

    Parameters
    ----------
    conf_mat    : np.ndarray – raw (unnormalised) confusion matrix
    class_names : list of str
    title       : str
    normalize   : bool – if True, normalise rows to percentages (0-100)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if normalize:
        row_sums = conf_mat.sum(axis=1, keepdims=True)
        # Avoid division by zero for classes with zero samples
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


def log_confusion_matrix(conf_mat, config, epoch, mode='val'):
    """
    Save a confusion matrix figure to disk and log it to W&B.

    Parameters
    ----------
    conf_mat : np.ndarray
    config   : dict
    epoch    : int
    mode     : str – used in the figure title and W&B key
    """
    n = config['num_classes']
    # Use numeric class labels if no names are provided
    class_names = [str(i) for i in range(n)]

    # Normalised (recall %) version
    fig_norm = plot_confusion_matrix(conf_mat, class_names,
                                     title=f'{mode.capitalize()} Confusion Matrix '
                                           f'(epoch {epoch}, normalised)',
                                     normalize=True)
    # Raw counts version
    fig_raw = plot_confusion_matrix(conf_mat, class_names,
                                    title=f'{mode.capitalize()} Confusion Matrix '
                                          f'(epoch {epoch}, counts)',
                                    normalize=False)

    # Save locally
    cm_dir = os.path.join(config['res_dir'], 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    norm_path = os.path.join(cm_dir, f'cm_{mode}_epoch{epoch:04d}_norm.png')
    raw_path  = os.path.join(cm_dir, f'cm_{mode}_epoch{epoch:04d}_raw.png')
    fig_norm.savefig(norm_path, dpi=150)
    fig_raw.savefig(raw_path,  dpi=150)
    plt.close(fig_norm)
    plt.close(fig_raw)

    # Log to W&B
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
    Run inference on `loader` and return metrics + confusion matrix.

    Returns
    -------
    metrics  : dict
    conf_mat : np.ndarray  (always returned for both val and test modes)
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
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(config['num_classes'])))
    return metrics, conf_mat


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

    # Nest every run inside its own named subfolder so results from different
    # experiments never collide:  <res_dir>/<wandb_name>/
    config['res_dir'] = os.path.join(config['res_dir'], config['wandb_name'])
    prepare_output(config)

    # Save args inside the run folder (also written to ./logs/ at the bottom)
    run_args_path = os.path.join(config['res_dir'], 'args_log.json')
    with open(run_args_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f'Run arguments saved to {run_args_path}')

    wandb.init(
        entity='my_team_projects',
        project='PASTIS',
        config=config,
        name=config['wandb_name'],
    )
    for k, v in config.items():
        wandb.run.summary[k] = v

    # ------------------------------------------------------------------
    # Build datasets + loaders
    # ------------------------------------------------------------------
    datasets_with_names = build_datasets(config)

    # date_positions: taken from the first dataset.
    # If PASTIS and Slovakia use different acquisition calendars you will need
    # to pass separate position tensors to the model; flag that case here.
    first_ds = datasets_with_names[0][0]

    train_loader, val_loader = get_loaders(datasets_with_names, config)

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
    print(model.param_ratio())
    model = model.to(device)
    model.apply(weight_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = FocalLoss(config['gamma'])
    wandb.watch(model, criterion, log='all', log_freq=100)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    trainlog  = {}
    best_mIoU = 0
    model_path = os.path.join(config['res_dir'], 'model.pth.tar')
    best_conf_mat = None

    for epoch in range(1, config['epochs'] + 1):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))

        model.train()
        train_metrics = train_epoch(model, optimizer, criterion, train_loader,
                                    device=device, config=config)

        print('Validation . . .')
        model.eval()
        val_metrics, conf_mat = evaluation(model, criterion, val_loader,
                                           device=device, config=config, mode='val')

        print('Loss {:.4f},  Acc {:.2f},  IoU {:.4f}'.format(
            val_metrics['val_loss'], val_metrics['val_accuracy'], val_metrics['val_IoU']))

        trainlog[epoch] = {**train_metrics, **val_metrics}
        checkpoint(trainlog, config)
        wandb.log({'epoch': epoch, **train_metrics, **val_metrics})

        # Log confusion matrix every 10 epochs and always on the final epoch
        if epoch % 10 == 0 or epoch == config['epochs']:
            log_confusion_matrix(conf_mat, config, epoch=epoch, mode='val')

        if val_metrics['val_IoU'] >= best_mIoU:
            best_mIoU    = val_metrics['val_IoU']
            best_conf_mat = conf_mat.copy()

            torch.save({
                'epoch':      epoch,
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
    # Save final results for the best checkpoint
    # ------------------------------------------------------------------
    print('Saving results for best checkpoint (val_IoU={:.4f}) . . .'.format(best_mIoU))

    # Reload best model and re-evaluate so saved metrics are consistent
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    best_val_metrics, best_conf_mat = evaluation(model, criterion, val_loader,
                                                  device=device, config=config, mode='val')

    save_results(best_val_metrics, best_conf_mat, config)

    # Log best-epoch confusion matrix with a distinct key so it's easy to find in W&B
    n = config['num_classes']
    class_names = [str(i) for i in range(n)]
    fig_best = plot_confusion_matrix(best_conf_mat, class_names,
                                     title='Best Val Confusion Matrix (normalised)',
                                     normalize=True)
    best_cm_path = os.path.join(config['res_dir'], 'confusion_matrices', 'cm_best_val_norm.png')
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
    parser.add_argument('--wandb_name', default='baseline_128', type=str,
                        help='Name of the W&B run (experiment identifier).')

    # -----------------------------------------------------------------------
    # Dataset paths  (at least one required at runtime)
    # -----------------------------------------------------------------------
    parser.add_argument('--dataset_pastis',
                        default='/home/mnegru/Adelina/Final_data/S2-2017-T31TFM-PixelSet',
                        # default='',
                        type=str,
                        help='Root folder of the PASTIS dataset. '
                             'Set to empty string "" to disable.')
    parser.add_argument('--dataset_slovakia',
                        # default='/home/mnegru/Adelina/Final_data/PixelSet-Slovakia-2020',
                        default='',
                        type=str,
                        help='Root folder of the Slovakia dataset. '
                             'Set to empty string "" to disable.')

    # -----------------------------------------------------------------------
    # PASTIS dataset-specific parameters
    # -----------------------------------------------------------------------
    parser.add_argument('--pastis_meanstd', default='S2-2017-T31TFM-meanstd.pkl', type=str,
                        help='Filename of the mean/std pickle inside --dataset_pastis.')
    parser.add_argument('--pastis_labels', default='label_44class', type=str,
                        help='Label column name used for PASTIS.')
    parser.add_argument('--pastis_subclasses',
                        default='[1,3,4,5,6,8,9,12,13,14,16,18,19,23,28,31,33,34,36,39]',
                        type=str,
                        help='Bracketed comma-separated list of PASTIS class indices to keep.')

    # -----------------------------------------------------------------------
    # Slovakia dataset-specific parameters
    # -----------------------------------------------------------------------
    parser.add_argument('--slovakia_meanstd', default='2020-Slovakia-meanstd.pkl', type=str,
                        help='Filename of the mean/std pickle inside --dataset_slovakia.')
    parser.add_argument('--slovakia_labels', default='CODE_GROUP', type=str,
                        help='Label column name used for Slovakia.')
    parser.add_argument('--slovakia_subclasses',
                        default='[1,3,4,5,6,8,9,12,13,14,16,18,19,23,28,31,33,34,36,39]',
                        type=str,
                        help='Bracketed comma-separated list of Slovakia class indices to keep.')

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
    # Architecture - PSE
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
    # Architecture - TAE
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
    # Classifier head  (shared across datasets - must match subclass counts)
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

    # Parse all mlp* fields from string to int list
    for k in list(config.keys()):
        if k.startswith('mlp'):
            config[k] = parse_int_list(config[k])

    # Parse per-dataset subclass lists
    config['pastis_subclasses']   = parse_int_list(config['pastis_subclasses'])
    config['slovakia_subclasses'] = parse_int_list(config['slovakia_subclasses'])

    os.makedirs('./logs', exist_ok=True)
    log_path = './logs/args_log.json'
    with open(log_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f'Arguments saved to {log_path} (also copied inside the run folder by main())')

    pprint.pprint(config)
    main(config)