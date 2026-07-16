#!/usr/bin/env python3
"""
Diagnoza riguroasa a gap-ului de domeniu in spatiul de embedding (z, 128-d).

Raspunde la intrebarea: "MMD e la podea pentru ca domeniile chiar se suprapun,
sau pentru ca metrica/scala e gresita?"

Masuratori (pe embeddings standardizate per-dimensiune, ca scala unei dimensiuni
sa nu domine distantele):
  1. MMD^2 + PERMUTATION TEST  -> semnificatie statistica, nu doar magnitudine.
     p mic => gap real (oricat de mic); p mare => nedistins de zgomot de esantion.
  2. DOMAIN CLASSIFIER (logistic regression, 5-fold CV) -> proxy A-distance
     (Ben-David et al.), masura standard din literatura DA:
       acc ~ 0.5  => domeniile se suprapun (nedistinse liniar)
       acc -> 1.0 => domenii clar separabile => MMD-ul de antrenare era orb
  3. ||mu_s - mu_t|| vs imprastierea intra-domeniu (ordinea de marime a shiftului)
  4. CORAL distance (gap de ordin 2, covariante)
  5. PCA 2-D colorat pe domenii (verificare vizuala)

Foloseste args_log.json + model.pth.tar dintr-un folder de run, deci reconstruieste
dataseturile si modelul EXACT ca train.py. Ruleaza din radacina repo-ului PASTIS
(unde exista models/ si dataset.py):

  python diagnose_domain_gap.py --run_dir ./results_MDPI/target_slovakia_from_PASTIS_128_mmd_pretrained \
      --n_samples 1024 --device cuda

  # optional, si pentru modelul neantrenat (verifica "suprapunere de la initializare"):
  python diagnose_domain_gap.py --run_dir ... --random_init
"""

from __future__ import annotations

import argparse
import json
import os
import pickle as pkl

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.stclassifier import PseTae
from dataset import PixelSetData, PixelSetData_preloaded


# ---------------------------------------------------------------------------
# Statistici
# ---------------------------------------------------------------------------

def standardize_joint(a: torch.Tensor, b: torch.Tensor):
    """Z-score per dimensiune pe norul reunit [a;b] (anti scale-domination)."""
    tot = torch.cat([a, b], 0)
    mu, sd = tot.mean(0, keepdim=True), tot.std(0, keepdim=True) + 1e-8
    return (a - mu) / sd, (b - mu) / sd


def mmd2(a: torch.Tensor, b: torch.Tensor, bw=None):
    """MMD^2 (V-statistic) cu kernel RBF; bw implicit = media distantelor^2."""
    n = min(a.size(0), b.size(0))
    a, b = a[:n], b[:n]
    tot = torch.cat([a, b], 0)
    l2 = torch.cdist(tot, tot) ** 2
    if bw is None:
        bw = l2.sum() / (tot.size(0) ** 2 - tot.size(0) + 1e-8)
    k = torch.exp(-l2 / bw)
    val = (k[:n, :n] + k[n:, n:] - k[:n, n:] - k[n:, :n]).mean().item()
    return val, bw


def permutation_test(a: torch.Tensor, b: torch.Tensor, n_perm=500, seed=0):
    """
    H0: a si b provin din aceeasi distributie.
    Bandwidth-ul e fixat pe cel observat, apoi etichetele de domeniu se amesteca.
    Returneaza (mmd_obs, media_null, std_null, p_value).
    """
    g = torch.Generator().manual_seed(seed)
    obs, bw = mmd2(a, b)
    tot = torch.cat([a, b], 0)
    n = min(a.size(0), b.size(0))
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = torch.randperm(tot.size(0), generator=g)
        null[i] = mmd2(tot[idx[:n]], tot[idx[n:2 * n]], bw=bw)[0]
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, float(null.mean()), float(null.std()), float(p)


def domain_classifier(a: torch.Tensor, b: torch.Tensor, seed=0):
    """Logistic regression source-vs-target, 5-fold CV. Proxy A-dist = 2(2acc-1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X = torch.cat([a, b], 0).numpy()
    y = np.r_[np.zeros(a.size(0)), np.ones(b.size(0))]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = float(cross_val_score(LogisticRegression(max_iter=5000), X, y, cv=cv).mean())
    return acc, 2 * (2 * acc - 1)


def coral_distance(a: torch.Tensor, b: torch.Tensor):
    d = a.size(1)
    def cov(z):
        zc = z - z.mean(0, keepdim=True)
        return (zc.t() @ zc) / (z.size(0) - 1)
    return float(((cov(a) - cov(b)) ** 2).sum().item() / (4 * d * d))


def mean_gap_vs_spread(a: torch.Tensor, b: torch.Tensor):
    gap = float((a.mean(0) - b.mean(0)).norm().item())
    within = float(0.5 * (torch.cdist(a, a).mean() + torch.cdist(b, b).mean()).item())
    return gap, within


# ---------------------------------------------------------------------------
# Extragere embeddings (identic cu forward_with_embedding din train.py)
# ---------------------------------------------------------------------------

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return [recursive_todevice(c, device) for c in x]


@torch.no_grad()
def extract_embeddings(model, loader, device, n_samples):
    model.eval()
    chunks, total = [], 0
    for x, _ in loader:
        x = recursive_todevice(x, device)
        feat = model.spatial_encoder(x)
        z = model.temporal_encoder(feat)
        chunks.append(z.cpu())
        total += z.size(0)
        if total >= n_samples:
            break
    return torch.cat(chunks, 0)[:n_samples]


def build_dataset(folder, meanstd_filename, sub_classes, labels_column, config):
    mean_std = pkl.load(open(os.path.join(folder, meanstd_filename), 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None
    cls = PixelSetData_preloaded if config.get('preload') else PixelSetData
    return cls(folder, labels=labels_column, npixel=config['npixel'],
               sub_classes=sub_classes, norm=mean_std, extra_feature=extra)


def build_model(config, first_ds, device):
    model_config = dict(
        input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
        mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'],
        mlp3=config['mlp3'], dropout=config['dropout'], T=config['T'],
        len_max_seq=config['lms'],
        positions=first_ds.date_positions if config['positions'] == 'bespoke' else None,
        mlp4=config['mlp4'],
    )
    if config['geomfeat']:
        model_config.update(with_extra=True, extra_size=4)
    else:
        model_config.update(with_extra=False, extra_size=None)
    return PseTae(**model_config).to(device)


# ---------------------------------------------------------------------------
# Raportare
# ---------------------------------------------------------------------------

def run_battery(z_s: torch.Tensor, z_t: torch.Tensor, label: str, n_perm: int, out: dict):
    a, b = standardize_joint(z_s, z_t)

    obs, null_mu, null_sd, p = permutation_test(a, b, n_perm=n_perm)
    acc, a_dist = domain_classifier(z_s, z_t)
    gap, spread = mean_gap_vs_spread(z_s, z_t)
    coral = coral_distance(z_s, z_t)

    res = {
        'mmd2_std': obs, 'mmd2_null_mean': null_mu, 'mmd2_null_std': null_sd,
        'mmd2_p_value': p,
        'domain_clf_acc': acc, 'proxy_A_distance': a_dist,
        'mean_gap': gap, 'within_spread': spread, 'gap_over_spread': gap / (spread + 1e-8),
        'coral_distance': coral,
        'n_source': int(z_s.size(0)), 'n_target': int(z_t.size(0)), 'dim': int(z_s.size(1)),
    }
    out[label] = res

    sig = 'GAP REAL (p<0.05)' if p < 0.05 else 'nedistins de zgomot'
    sep = ('SEPARABILE' if acc > 0.75 else
           'partial separabile' if acc > 0.6 else 'suprapuse')
    print(f'\n===== {label} =====')
    print(f'  MMD^2 (standardizat) : {obs:.5f}   null: {null_mu:.5f} ± {null_sd:.5f}   p = {p:.4f}  -> {sig}')
    print(f'  Domain classifier    : acc = {acc:.3f}  (proxy A-distance = {a_dist:.2f})  -> domenii {sep}')
    print(f'  ||mu_s - mu_t||      : {gap:.3f}   vs imprastiere intra-domeniu {spread:.2f}   (raport {gap/spread:.4f})')
    print(f'  CORAL distance       : {coral:.6f}')
    return res


def pca_plot(z_s, z_t, path, title):
    tot = torch.cat([z_s, z_t], 0)
    tot = tot - tot.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(tot, q=2)
    proj = (tot @ V[:, :2]).numpy()
    n = z_s.size(0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(proj[:n, 0], proj[:n, 1], s=6, alpha=0.4, label='source (PASTIS)')
    ax.scatter(proj[n:, 0], proj[n:, 1], s=6, alpha=0.4, label='target (Slovakia)')
    ax.set_title(title); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'  PCA plot -> {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--run_dir', required=True,
                    help='Folder de run care contine args_log.json si model.pth.tar')
    ap.add_argument('--n_samples', type=int, default=1024, help='Embeddings per domeniu')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--n_perm', type=int, default=500, help='Permutari pentru testul de semnificatie')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--random_init', action='store_true',
                    help='Masoara si pe modelul NEANTRENAT (test "suprapunere de la initializare")')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    with open(os.path.join(args.run_dir, 'args_log.json')) as f:
        config = json.load(f)

    if not config.get('dataset_pastis') or not config.get('dataset_slovakia'):
        raise ValueError('Runul trebuie sa aiba ambele dataseturi active (source + target).')

    print('[Diag] Construiesc dataseturile...')
    ds_s = build_dataset(config['dataset_pastis'], config['pastis_meanstd'],
                         config['pastis_subclasses'], config['pastis_labels'], config)
    ds_t = build_dataset(config['dataset_slovakia'], config['slovakia_meanstd'],
                         config['slovakia_subclasses'], config['slovakia_labels'], config)

    import torch.utils.data as data
    ld = lambda ds: data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                    num_workers=config.get('num_workers', 4))
    loader_s, loader_t = ld(ds_s), ld(ds_t)

    model = build_model(config, ds_s, device)
    results = {}

    if args.random_init:
        from learning.weight_init import weight_init
        model.apply(weight_init)
        print('\n[Diag] === Model NEANTRENAT (random init) ===')
        z_s = extract_embeddings(model, loader_s, device, args.n_samples)
        z_t = extract_embeddings(model, loader_t, device, args.n_samples)
        run_battery(z_s, z_t, 'random_init', args.n_perm, results)
        pca_plot(z_s, z_t, os.path.join(args.run_dir, 'diag_pca_random_init.png'),
                 'Embeddings z (random init)')

    ckpt_path = os.path.join(args.run_dir, 'model.pth.tar')
    print(f'\n[Diag] === Checkpoint antrenat: {ckpt_path} ===')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    z_s = extract_embeddings(model, loader_s, device, args.n_samples)
    z_t = extract_embeddings(model, loader_t, device, args.n_samples)
    run_battery(z_s, z_t, 'trained', args.n_perm, results)
    pca_plot(z_s, z_t, os.path.join(args.run_dir, 'diag_pca_trained.png'),
             f'Embeddings z (epoch {ckpt.get("epoch", "?")})')

    out_path = os.path.join(args.run_dir, 'diag_domain_gap.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f'\n[Diag] Rezultate salvate in {out_path}')

    r = results['trained']
    print('\n================ VERDICT ================')
    if r['mmd2_p_value'] >= 0.05 and r['domain_clf_acc'] < 0.6:
        print('Domeniile se SUPRAPUN in embedding: MMD nesemnificativ + clasificator la chance.')
        print('=> MMD-ul de antrenare raporta corect; nu exista gap marginal de aliniat.')
    elif r['domain_clf_acc'] >= 0.6:
        print('Domeniile SUNT separabile (clasificatorul le distinge), desi MMD parea la podea.')
        print('=> MMD-ul de antrenare era ORB (scala/bandwidth); trebuie recalibrat:')
        print('   standardizeaza z inainte de MMD si/sau foloseste bandwidth mai mic.')
    else:
        print('Gap statistic detectabil (p<0.05) dar mic (clasificator aproape de chance).')
        print('=> Exista un shift marginal slab; efectul practic al alinierii va fi limitat.')


if __name__ == '__main__':
    main()