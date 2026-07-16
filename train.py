import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
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

    min_b = min(y_pastis.shape[0], y_slovakia.shape[0])
    y_a = y_pastis[:min_b]
    y_b = y_slovakia[:min_b]

    return x_mixed, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def compute_lam(epoch, total_epochs, warmup_epochs=20):

    if epoch >= warmup_epochs:
        return 0.0
    return 1.0 - (epoch - 1) / (warmup_epochs - 1)
def cutmix_temporal(x_pastis, x_slovakia, lam):

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
# Feature-space alignment helpers (MMD & CORAL) — a 4-a familie
# ---------------------------------------------------------------------------

def forward_with_embedding(model, x):
    feat   = model.spatial_encoder(x)
    z      = model.temporal_encoder(feat)
    logits = model.decoder(z)
    return z, logits


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_total = source.size(0) + target.size(0)
    total   = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(n_total, n_total, total.size(1))
    total1 = total.unsqueeze(1).expand(n_total, n_total, total.size(1))
    l2 = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = l2.detach().sum() / (n_total ** 2 - n_total + 1e-8)

    bandwidth  = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidths = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernels    = [torch.exp(-l2 / (bw + 1e-8)) for bw in bandwidths]
    return sum(kernels)


def mmd_loss(z_s, z_t, kernel_mul=2.0, kernel_num=5, standardize=True):
    n = min(z_s.size(0), z_t.size(0))
    z_s, z_t = z_s[:n], z_t[:n]
    if standardize:
        tot = torch.cat([z_s, z_t], dim=0)
        mu = tot.mean(dim=0, keepdim=True).detach()
        sd = (tot.std(dim=0, keepdim=True) + 1e-8).detach()
        z_s = (z_s - mu) / sd
        z_t = (z_t - mu) / sd
    kernels = gaussian_kernel(z_s, z_t, kernel_mul, kernel_num)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]
    return torch.mean(XX + YY - XY - YX)


def mmd_diagnostic(z_s, z_t, fractions=(0.1, 1.0, 10.0)):
    with torch.no_grad():
        n = min(z_s.size(0), z_t.size(0))
        a, b = z_s[:n].detach(), z_t[:n].detach()
        # aceeasi standardizare ca in mmd_loss, ca diagnoza sa masoare gapul real
        tot = torch.cat([a, b], dim=0)
        mu = tot.mean(dim=0, keepdim=True)
        sd = tot.std(dim=0, keepdim=True) + 1e-8
        a, b = (a - mu) / sd, (b - mu) / sd
        total = torch.cat([a, b], dim=0)
        m = total.size(0)
        t0 = total.unsqueeze(0).expand(m, m, total.size(1))
        t1 = total.unsqueeze(1).expand(m, m, total.size(1))
        l2 = ((t0 - t1) ** 2).sum(2)
        bw_median = (l2.sum() / (m ** 2 - m + 1e-8)).item()

        out = {'mmd_bw_median': bw_median}
        vals = []
        for frac in fractions:
            bw = bw_median * frac + 1e-8
            k = torch.exp(-l2 / bw)
            XX = k[:n, :n]; YY = k[n:, n:]; XY = k[:n, n:]; YX = k[n:, :n]
            v = torch.mean(XX + YY - XY - YX).item()
            tag = ('%g' % frac).replace('.', 'p')
            out[f'mmd_at_{tag}x'] = v
            vals.append(v)
        out['mmd_max'] = max(vals)
        return out


def coral_loss(z_s, z_t):
    d = z_s.size(1)

    def cov(z):
        n  = z.size(0)
        zc = z - z.mean(dim=0, keepdim=True)
        return (zc.t() @ zc) / (n - 1 + 1e-8)

    c_s = cov(z_s)
    c_t = cov(z_t)
    return ((c_s - c_t) ** 2).sum() / (4.0 * d * d)

class CentroidMemory:
    """Prototipuri EMA per clasa/domeniu + asignare stabila pe epoca."""

    def __init__(self, ema=0.9, std_ema=0.99):
        self.ema = ema
        self.std_ema = std_ema
        self.proto_s, self.proto_t = {}, {}    
        self.count_s, self.count_t = {}, {}     
        self.zstd = None                        
        self.assignment_map = {}                

    @staticmethod
    def _batch_means_counts(z, y):
        means, counts = {}, {}
        for k in torch.unique(y).tolist():
            zk = z[y == k]
            means[int(k)] = zk.mean(dim=0)      # cu gradient
            counts[int(k)] = zk.size(0)
        return means, counts

    def observe(self, z_s, y_s, z_t, y_t):
        s = torch.cat([z_s, z_t], dim=0).detach().std(dim=0)
        self.zstd = s.clone() if self.zstd is None else self.std_ema * self.zstd + (1 - self.std_ema) * s

        ms, cs = self._batch_means_counts(z_s, y_s)
        mt, ct = self._batch_means_counts(z_t, y_t)
        for k, v in ms.items():
            vd = v.detach()
            self.proto_s[k] = vd.clone() if k not in self.proto_s else self.ema * self.proto_s[k] + (1 - self.ema) * vd
            self.count_s[k] = self.count_s.get(k, 0) + cs[k]
        for k, v in mt.items():
            vd = v.detach()
            self.proto_t[k] = vd.clone() if k not in self.proto_t else self.ema * self.proto_t[k] + (1 - self.ema) * vd
            self.count_t[k] = self.count_t.get(k, 0) + ct[k]
        return ms, mt

    def compute_assignment(self, min_count=50, mutual_nn=True):
        std = self.zstd if self.zstd is not None else 1.0
        s_keys = [k for k in sorted(self.proto_s) if self.count_s.get(k, 0) >= min_count]
        t_keys = [k for k in sorted(self.proto_t) if self.count_t.get(k, 0) >= min_count]
        if not s_keys or not t_keys:
            self.assignment_map = {}
            return {}

        S = torch.stack([self.proto_s[k] for k in s_keys]) / (std if isinstance(std, float) else (std + 1e-8))
        T = torch.stack([self.proto_t[k] for k in t_keys]) / (std if isinstance(std, float) else (std + 1e-8))
        cost = torch.cdist(S, T)                                  # (ns, nt)
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost.cpu().numpy())
        pairs = {s_keys[i]: t_keys[j] for i, j in zip(r, c)}

        if mutual_nn:
            nn_t = {s_keys[i]: t_keys[int(cost[i].argmin())] for i in range(len(s_keys))}
            nn_s = {t_keys[j]: s_keys[int(cost[:, j].argmin())] for j in range(len(t_keys))}
            pairs = {k: v for k, v in pairs.items() if nn_t.get(k) == v and nn_s.get(v) == k}

        self.assignment_map = pairs
        return pairs

    def align_loss(self, means_s, means_t, direction='s2t'):
        std = (self.zstd + 1e-8) if self.zstd is not None else 1.0
        terms = []
        for sk, tk in self.assignment_map.items():
            if direction in ('s2t', 'both') and sk in means_s and tk in self.proto_t:
                terms.append((((means_s[sk] - self.proto_t[tk].detach()) / std) ** 2).sum())
            if direction in ('t2s', 'both') and tk in means_t and sk in self.proto_s:
                terms.append((((means_t[tk] - self.proto_s[sk].detach()) / std) ** 2).sum())
        if not terms:
            return None
        return torch.stack(terms).mean()



class DomainSpecificBN(nn.Module):
    """Invelis peste un strat BatchNorm cu doua ramuri (source / target)."""

    def __init__(self, bn):
        super().__init__()
        import copy
        self.bn_s = copy.deepcopy(bn)   # mosteneste weight/bias/running stats incarcate
        self.bn_t = copy.deepcopy(bn)
        self.domain = 'target'          # implicit: target (folosit la eval)

    def forward(self, x):
        return self.bn_s(x) if self.domain == 'source' else self.bn_t(x)


def convert_to_dsbn(module):
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            setattr(module, name, DomainSpecificBN(child))
            count += 1
        else:
            count += convert_to_dsbn(child)
    return count


def set_dsbn_domain(model, domain):
    for m in model.modules():
        if isinstance(m, DomainSpecificBN):
            m.domain = domain


def train_epoch_align(model, optimizer, criterion, pastis_loader, slovakia_loader,
                      epoch, device, config):

    acc_meter   = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter  = tnt.meter.AverageValueMeter()
    cls_meter   = tnt.meter.AverageValueMeter()
    align_meter = tnt.meter.AverageValueMeter()
    y_true, y_pred = [], []
    mmd_diag = None                      # diagnoza cu bandwidth fix, o data / epoca

    align_type  = config['align_type']
    lambda_a    = config['align_lambda']
    cls_domains = config.get('align_cls_domains', 'both')
    use_dsbn    = config.get('dsbn', False)

    warmup = config.get('align_warmup_epochs', 0) or 0
    lam_eff = lambda_a if warmup <= 0 else lambda_a * min(1.0, max(0, epoch - 1) / warmup)

    if align_type == 'centroid' and not hasattr(model, '_centroid_mem'):
        model._centroid_mem = CentroidMemory(ema=config.get('centroid_ema', 0.9))
    mem = getattr(model, '_centroid_mem', None)

    if align_type == 'centroid':
        mem.compute_assignment(min_count=config.get('centroid_min_count', 50),
                               mutual_nn=bool(config.get('centroid_mutual_nn', 1)))

    print(f'  [Align] epoch={epoch}, type={align_type}, lambda_a={lambda_a:.3f} '
          f'(eff={lam_eff:.3f}), cls_domains={cls_domains}, dsbn={use_dsbn}')

    pastis_iter = iter(pastis_loader)

    for i, (x_slovakia, y_slovakia) in enumerate(slovakia_loader):
        try:
            x_pastis, y_pastis = next(pastis_iter)
        except StopIteration:
            pastis_iter = iter(pastis_loader)
            x_pastis, y_pastis = next(pastis_iter)

        x_slovakia = recursive_todevice(x_slovakia, device)
        x_pastis   = recursive_todevice(x_pastis,   device)
        y_slovakia = y_slovakia.to(device)
        y_pastis   = y_pastis.to(device)

        if use_dsbn:
            set_dsbn_domain(model, 'source')
        if align_type == 'project' and config.get('project_freeze_source', 1):
            # sursa din backbone INGHETAT (referinta preantrenata, fara gradient)
            with torch.no_grad():
                z_s = model._frozen_temporal(model._frozen_spatial(x_pastis))
            out_s = None
        else:
            z_s, out_s = forward_with_embedding(model, x_pastis)
        if use_dsbn:
            set_dsbn_domain(model, 'target')
        z_t, out_t = forward_with_embedding(model, x_slovakia)

        if align_type == 'project':
            loss_cls = criterion(out_t, y_slovakia.long())   # target = ancora
        elif cls_domains == 'both':
            loss_cls = criterion(out_s, y_pastis.long()) + criterion(out_t, y_slovakia.long())
        elif cls_domains == 'target':
            loss_cls = criterion(out_t, y_slovakia.long())
        elif cls_domains == 'source':
            loss_cls = criterion(out_s, y_pastis.long())
        else:
            raise ValueError(f'align_cls_domains necunoscut: {cls_domains}')

        if align_type == 'mmd':
            loss_align = mmd_loss(z_s, z_t,
                                  kernel_mul=config['mmd_kernel_mul'],
                                  kernel_num=config['mmd_kernel_num'])
        elif align_type == 'coral':
            loss_align = coral_loss(z_s, z_t)
        elif align_type == 'centroid':
            means_s, means_t = mem.observe(z_s, y_pastis, z_t, y_slovakia)
            la = mem.align_loss(means_s, means_t,
                                direction=config.get('centroid_direction', 's2t'))
            loss_align = la if la is not None else z_t.sum() * 0.0
        elif align_type == 'project':
            z_s_proj = model._projector(z_s)
            if config.get('project_align', 'mmd') == 'coral':
                loss_align = coral_loss(z_s_proj, z_t)
            else:
                loss_align = mmd_loss(z_s_proj, z_t,
                                      kernel_mul=config['mmd_kernel_mul'],
                                      kernel_num=config['mmd_kernel_num'])
        elif align_type is None:
            loss_align = z_t.sum() * 0.0
        else:
            raise ValueError(f'align_type necunoscut: {align_type}')

        loss = loss_cls + lam_eff * loss_align

        if align_type == 'mmd' and i == 0:
            mmd_diag = mmd_diagnostic(z_s, z_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_meter.add(out_t.detach(), y_slovakia)
        y_true.extend(list(map(int, y_slovakia)))
        y_pred.extend(list(out_t.detach().argmax(dim=1).cpu().numpy()))
        loss_meter.add(loss.item())
        cls_meter.add(loss_cls.item())
        align_meter.add(loss_align.item())

        if (i + 1) % config['display_step'] == 0:
            align_tag = align_type.upper() if align_type is not None else 'NOALIGN'
            print('Step [{}/{}], Loss: {:.4f}, Cls: {:.4f}, {}: {:.4f}, Acc: {:.2f}'.format(
                i + 1, len(slovakia_loader),
                loss_meter.value()[0], cls_meter.value()[0],
                align_tag, align_meter.value()[0], acc_meter.value()[0]))

    align_contrib = lam_eff * align_meter.value()[0]

    metrics = {
        'train_loss':          loss_meter.value()[0],
        'train_accuracy':      acc_meter.value()[0],
        'train_IoU':           mIou(y_true, y_pred, n_classes=config['num_classes']),
        'train_cls':           cls_meter.value()[0],
        'train_align':         align_meter.value()[0],
        'train_align_contrib': align_contrib,
        'align_lambda_eff':    lam_eff,
    }

    if mmd_diag is not None:
        metrics.update(mmd_diag)
        print('  [MMD-diag] bw_median={:.4g}  mmd@0.1x={:.4g}  mmd@1x={:.4g}  mmd@10x={:.4g}  max={:.4g}'.format(
            mmd_diag['mmd_bw_median'], mmd_diag['mmd_at_0p1x'],
            mmd_diag['mmd_at_1x'], mmd_diag['mmd_at_10x'], mmd_diag['mmd_max']))

    if align_type == 'centroid' and mem is not None:
        n_pairs = len(mem.assignment_map)
        metrics['centroid_n_pairs'] = n_pairs
        amap = ', '.join(f'{s}->{t}' for s, t in sorted(mem.assignment_map.items())) or '(inca gol)'
        print(f'  [Centroid] {n_pairs} perechi (min_count+mutualNN): {amap}')

    return metrics
# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_single_dataset(folder, meanstd_filename, sub_classes, labels_column, config):
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
    needs_paired = (config.get('mixup_type') is not None
                    or config.get('align_type') is not None
                    or config.get('dsbn', False))
    if needs_paired and pastis_train_subset and slovakia_train_subset:
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
            f'{mode}_accuracy':          acc_meter.value()[0],
            f'{mode}_balanced_accuracy': 100.0 * balanced_accuracy_score(y_true, y_pred),
            f'{mode}_loss':              loss_meter.value()[0],
            f'{mode}_IoU':               mIou(y_true, y_pred, config['num_classes']),
        }

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
        project='MDPI_PASTIS',
        config=config,
        tags=[t.strip() for t in config['wandb_tags'].split(',') if t.strip()],
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
    train_loader, val_loader, pastis_loader, slovakia_loader = get_loaders_mixup(datasets_with_names, config)

    use_align_loop = (config.get('align_type') is not None or config.get('dsbn', False))
    if use_align_loop:
        if pastis_loader is None or slovakia_loader is None:
            raise ValueError(
                'Alinierea/DSBN necesită ambele dataset-uri active '
                '(--dataset_pastis și --dataset_slovakia).')
        print(f'\n[Align] Activ: type={config["align_type"]}, '
              f'lambda_a={config["align_lambda"]}, '
              f'cls_domains={config.get("align_cls_domains", "both")}, '
              f'dsbn={config.get("dsbn", False)}')
    elif config.get('mixup_type') is not None:
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
    start_epoch, best_mIoU = load_checkpoint(model, optimizer, config)


    if config.get('dsbn', False):
        n_bn = convert_to_dsbn(model)
        model = model.to(device)
        print(f'[DSBN] {n_bn} straturi BatchNorm convertite in Domain-Specific BN.')
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    if config.get('align_type') == 'project':
        import copy
        hidden = config.get('project_hidden', 0)
        if hidden and hidden > 0:
            model._projector = nn.Sequential(
                nn.Linear(128, hidden), nn.ReLU(), nn.Linear(hidden, 128)).to(device)
        else:
            model._projector = nn.Linear(128, 128).to(device)   # varianta A: liniar
        if config.get('project_freeze_source', 1):
            model._frozen_spatial  = copy.deepcopy(model.spatial_encoder).to(device).eval()
            model._frozen_temporal = copy.deepcopy(model.temporal_encoder).to(device).eval()
            for p in list(model._frozen_spatial.parameters()) + list(model._frozen_temporal.parameters()):
                p.requires_grad = False
        print(f'[Project] proiector {"MLP" if hidden else "Linear"}(128->128) creat; '
              f'frozen_source={bool(config.get("project_freeze_source", 1))}, '
              f'align={config.get("project_align", "mmd")}')
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config['lr'])


    freeze_encoder(model, config)
    if config.get('freeze_encoder', False):
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=config['lr'])

    wandb.watch(model, criterion, log='all', log_freq=100)


    trainlog     = {}
    model_path   = os.path.join(config['res_dir'], 'model.pth.tar')
    best_conf_mat = None
    best_present_labels = None
    best_acc     = 0.0          # <-- adaugă
    best_bal_acc = 0.0          # <-- adaugă
    for epoch in range(start_epoch, config['epochs'] + 1):
        print('EPOCH {}/{}'.format(epoch, config['epochs']))

        model.train()
        if config.get('align_type') is not None or config.get('dsbn', False):
            train_metrics = train_epoch_align(
                model, optimizer, criterion,
                pastis_loader, slovakia_loader,
                epoch=epoch, device=device, config=config,
            )
        elif config.get('mixup_type') is not None:
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
        if config.get('dsbn', False):
            set_dsbn_domain(model, 'target')   # evaluam pe target -> ramura target BN
        val_metrics, conf_mat, present_labels = evaluation(
            model, criterion, val_loader, device=device, config=config, mode='val')

        print('Loss {:.4f},  Acc {:.2f},  BalAcc {:.2f},  IoU {:.4f}'.format(
                    val_metrics['val_loss'], val_metrics['val_accuracy'],
                    val_metrics['val_balanced_accuracy'], val_metrics['val_IoU']))

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
            # Running bests pe validare
            best_acc     = max(best_acc,     val_metrics['val_accuracy'])
            best_bal_acc = max(best_bal_acc, val_metrics['val_balanced_accuracy'])
            # best_mIoU e actualizat în blocul de checkpoint de mai sus

            wandb.run.summary['best_val_accuracy']          = best_acc
            wandb.run.summary['best_val_balanced_accuracy'] = best_bal_acc
            wandb.run.summary['best_val_IoU']               = best_mIoU
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
    if config.get('dsbn', False):
        set_dsbn_domain(model, 'target')
    best_val_metrics, best_conf_mat, best_present_labels = evaluation(
        model, criterion, val_loader, device=device, config=config, mode='val')
    save_results(best_val_metrics, best_conf_mat, config)

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
    parser.add_argument('--wandb_tags', default='task_1', type=str,
                        help='Comma-separated W&B tags for the run (e.g. "task_1" or "task_2").')

    # -----------------------------------------------------------------------
    # Resume / fine-tune
    # -----------------------------------------------------------------------
    parser.add_argument('--mixup_type', default='temporal', type=str,
                    choices=['linear', 'temporal', 'bands', 'pixels'],
                    help='Tipul de mixup. Dacă nu e setat, nu se face mixup.')
    parser.add_argument('--mixup_warmup_epochs', default=50, type=int,  
                        help='Numărul de epoci în care lam scade de la 1.0 (pur PASTIS) '
                             'la 0.0 (pur Slovakia). După warmup antrenează doar pe Slovakia.')


    parser.add_argument('--align_type', default=None, type=str,
                        choices=['mmd', 'coral', 'centroid', 'project'],
                        help='Metodă de aliniere în spațiul de feature pe embedding-ul '
                             'global z (128-d). Dacă e setată (sau --dsbn), are prioritate '
                             'față de mixup: L = L_FL + lambda_a * L_align. '
                             'centroid = pseudo-corespondenta prin asignare Hungarian '
                             '(nu presupune corespondenta de indici de clasa). '
                             'Lăsat gol => fără termen de aliniere (folosit cu --dsbn).')
    parser.add_argument('--align_lambda', default=1.0, type=float,
                        help='Ponderea lambda_a a termenului de aliniere (mmd/coral/centroid).')
    parser.add_argument('--align_cls_domains', default='target', type=str,
                        choices=['both', 'target', 'source'],
                        help='Pe ce domenii se calculează pierderea de clasificare în '
                             'timpul alinierii (target = Slovacia, etichetat din LPIS).')
    parser.add_argument('--centroid_ema', default=0.9, type=float,
                        help='Coeficient EMA pentru prototipurile per clasa (align_type=centroid).')
    parser.add_argument('--align_warmup_epochs', default=0, type=int,
                        help='Warmup pentru lambda de aliniere: 0->lambda_a liniar pe primele '
                             'N epoci (0 = fara warmup). Recomandat ~10 pentru centroid, ca '
                             'embedding-ul sa se formeze inainte sa alinieze.')
    parser.add_argument('--centroid_min_count', default=50, type=int,
                        help='Nr. minim de exemple acumulate ca o clasa sa intre in asignare '
                             '(exclude clasele rare cu prototip zgomotos).')
    parser.add_argument('--centroid_mutual_nn', default=1, type=int,
                        help='1 = pastreaza doar perechile mutual-cel-mai-apropiat in asignare '
                             '(stabilizeaza matching-ul). 0 = toate perechile Hungarian.')
    parser.add_argument('--centroid_direction', default='s2t', type=str,
                        choices=['s2t', 't2s', 'both'],
                        help='Directia alinierii de centroizi. s2t (recomandat) misca doar '
                             'sursa spre target, lasand target-ul ancorat de clasificare; '
                             'both = simetric (poate corupe target-ul la matching gresit).')
    parser.add_argument('--project_freeze_source', default=1, type=int,
                        help='align_type=project: 1 = backbone sursa inghetat (copie a '
                             'ponderilor preantrenate, referinta fixa); 0 = sursa din '
                             'backbone-ul partajat antrenabil.')
    parser.add_argument('--project_align', default='mmd', type=str, choices=['mmd', 'coral'],
                        help='align_type=project: cum se aliniaza T(z_s) la z_t (mmd standardizat '
                             'sau coral).')
    parser.add_argument('--project_hidden', default=0, type=int,
                        help='align_type=project: 0 = proiector Linear(128,128) (varianta A); '
                             '>0 = MLP cu acest strat ascuns (varianta B).')
    parser.add_argument('--dsbn', action='store_true',
                        help='Domain-Specific BatchNorm: statistici + affine BN separate '
                             'per domeniu (sursa/target). Se poate combina cu orice --align_type '
                             'sau folosi singur (fara --align_type) pentru DSBN pur. Absoarbe '
                             'shiftul de medie per-domeniu fara hiperparametru.')
    parser.add_argument('--mmd_kernel_num', default=5, type=int,
                        help='Numărul de kernel-uri RBF în amestecul MMD.')
    parser.add_argument('--mmd_kernel_mul', default=2.0, type=float,
                        help='Multiplicatorul de bandwidth între kernel-urile RBF succesive.')

    parser.add_argument('--resume_path', default='', type=str,
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


    parser.add_argument('--dataset_pastis',
                        default='',
                        type=str,
                        help='Root folder of the PASTIS dataset. Set to "" to disable.')
    parser.add_argument('--dataset_slovakia',
                        default=f'',
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