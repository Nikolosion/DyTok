# train_cl.py — DyTok: class-incremental learning на CIFAR-10/100 (работает на CPU).
#
# Протокол (стандартный для continual learning, ср. De Lange et al., 2021):
#   * CIFAR-100: 5 шагов × 20 классов; CIFAR-10: 5 шагов × 2 класса;
#   * каждый шаг: expand_classifier -> [sleep: refresh слотов, обучение novel-головы,
#     двухконтурная дистилляция (латентные «сны» + реальные данные), калибровка]
#     -> дневное обучение на новых классах -> оценка на всех виденных классах;
#   * в конце шага: freeze_anchors + snapshot_anchor_habits (реплей-буфер для следующего sleep).
#
# Метрики: AA (average accuracy), BWT (backward transfer), Forgetting — task-level,
# плюс переклассовая матрица A[шаг][класс]. Диагностика: usage кодбука, замороженные
# слоты/коды, ID-стабильность на фиксированном probe-батче.
#
# Запуск (из корня проекта, внутри venv):
#   venv\Scripts\python train_cl.py --dataset cifar10 --quick --tag smoke
#   venv\Scripts\python train_cl.py --dataset cifar100 --epochs 2 --tag with-sleep
#   venv\Scripts\python train_cl.py --dataset cifar100 --epochs 2 --no-sleep --tag no-sleep
#
# Результаты пишутся в results/<dataset>_<tag>.json (строка-лог на каждый шаг).

import argparse
import json
import sys
import time
from pathlib import Path

# Консоль Windows может быть cp1251 — принудительно UTF-8, чтобы кириллица не падала
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import MultiTokenizer, RNNClassifier, SymbolicDyTokTokenizer

# ==========================================================
# Препроцессинг: (B,3,32,32) in [0,1] -> (byte_grid [B,G,G] long, patch_feat [B,G,G,8])
# Тот же рецепт, что в test.py: mean/std патча, яркость, модули градиента.
# ==========================================================
def image_to_grid(x, G, P):
    B = x.shape[0]
    patches = F.unfold(x, kernel_size=P, stride=P).view(B, 3 * P * P, -1).permute(0, 2, 1)
    patches = patches.view(B, G * G, 3, P * P)
    mean_c = patches.mean(dim=3)                      # (B, G*G, 3)
    std_c = patches.std(dim=3, unbiased=False)        # (B, G*G, 3)
    lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    gx = F.pad((lum[:, :, 1:] - lum[:, :, :-1]).abs(), (0, 1)).unsqueeze(1)   # (B,1,32,32)
    gy = F.pad((lum[:, 1:, :] - lum[:, :-1, :]).abs(), (0, 0, 0, 1)).unsqueeze(1)
    gx_p = F.unfold(gx, kernel_size=P, stride=P).view(B, G * G, P * P).mean(-1)
    gy_p = F.unfold(gy, kernel_size=P, stride=P).view(B, G * G, P * P).mean(-1)
    byte = (0.299 * mean_c[..., 0] + 0.587 * mean_c[..., 1] + 0.114 * mean_c[..., 2])
    byte = byte.mul(255).round().clamp(0, 255).long()
    feat = torch.cat([mean_c, std_c, gx_p[..., None], gy_p[..., None]], dim=-1)  # (B, G*G, 8)
    return byte.view(B, G, G), feat.view(B, G, G, -1)


def make_collate(G, P):
    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])  # (B,3,32,32) float [0,1] (ToTensor)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        bg, pf = image_to_grid(imgs, G, P)
        return {"image": (bg, pf), "labels": y}
    return collate


class CifarSubset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def collect(loader, n, device):
    """Первые n сэмплов загрузчика -> dict входа + метки."""
    ib, if_, y = [], [], []
    got = 0
    for batch in loader:
        ib.append(batch["image"][0].to(device))
        if_.append(batch["image"][1].to(device))
        y.append(batch["labels"].to(device))
        got += batch["labels"].size(0)
        if got >= n:
            break
    return {"image": (torch.cat(ib)[:n], torch.cat(if_)[:n]),
            "labels": torch.cat(y)[:n]}


# ==========================================================
# Модель
# ==========================================================
def build_model(n_classes, args, device):
    tok = SymbolicDyTokTokenizer(
        8, d_model=args.d_model, n_dims=2, vocab=256,
        num_slots=args.num_slots, max_emissions=args.max_emissions,
        num_dream_templates=args.num_dream_templates)
    mtok = MultiTokenizer({"image": tok}, d_model=args.d_model, n_heads=4,
                          num_fused=args.num_fused)
    return RNNClassifier(args.d_model, n_classes, num_old_classes=n_classes,
                         tokenizer=mtok).to(device)


# ==========================================================
# Обучение и оценка
# ==========================================================
def train_step(model, loader, opt, device, aux_w=0.25, sp_w=1e-3):
    model.train()
    tot, n = 0.0, 0
    for batch in tqdm(loader, desc="daytime", leave=False, file=sys.stdout):
        inp = {"image": (batch["image"][0].to(device), batch["image"][1].to(device))}
        y = batch["labels"].to(device)
        opt.zero_grad(set_to_none=True)
        fused, sids, coords, pres, auxs = model.tokenizer(inp)
        logits = model(fused)
        aux = (sum(a["vq_commit"] for a in auxs.values()) * aux_w +
               sum(a["sparsity"] for a in auxs.values()) * sp_w)
        loss = F.cross_entropy(logits, y) + aux
        loss.backward()
        model.tokenizer.mask_frozen_grads()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.tokenizer.sparsify_slots()
        tot += loss.item() * y.size(0)
        n += y.size(0)
    return tot / n


@torch.no_grad()
def evaluate_per_class(model, loader, device):
    model.eval()
    correct, total = {}, {}
    for batch in loader:
        inp = {"image": (batch["image"][0].to(device), batch["image"][1].to(device))}
        y = batch["labels"].to(device)
        fused, *_ = model.tokenizer(inp)
        logits = model(fused)
        pred = logits.argmax(-1)
        for c in torch.unique(y).tolist():
            m = (y == c)
            correct[c] = correct.get(c, 0) + int(pred[m].eq(y[m]).sum())
            total[c] = total.get(c, 0) + int(m.sum())
    return {c: correct[c] / total[c] for c in total}


# ==========================================================
# Метрики continual learning (task-level)
# ==========================================================
def cl_metrics(A, cps):
    """A: список dict {class: acc} по шагам; cps — классов в задаче.
    Возвращает (AA, BWT, Forgetting)."""
    T = len(A)
    task_acc = [[float(np.mean([A[t][c] for c in range(i * cps, (i + 1) * cps)]))
                 for i in range(t + 1)] for t in range(T)]
    AA = float(np.mean([np.mean(row) for row in task_acc]))
    BWT = float(np.mean([task_acc[T - 1][i] - task_acc[i][i] for i in range(T - 1)])) if T > 1 else 0.0
    Fgt = max(0.0, float(np.mean(
        [max(task_acc[t][i] for t in range(i, T)) - task_acc[T - 1][i] for i in range(T)])))
    return AA, BWT, Fgt


# ==========================================================
# Основной цикл
# ==========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--sleep-steps", type=int, default=160)
    ap.add_argument("--sleep-batch", type=int, default=256)
    ap.add_argument("--snap-batch", type=int, default=64)
    ap.add_argument("--no-sleep", action="store_true")
    ap.add_argument("--max-train", type=int, default=0, help="0 = весь обучающий набор")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="быстрый смоук: 1 эпоха, 4k сэмплов")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-slots", type=int, default=16)
    ap.add_argument("--max-emissions", type=int, default=8)
    ap.add_argument("--num-fused", type=int, default=8)
    ap.add_argument("--num-dream-templates", type=int, default=32)
    args = ap.parse_args()

    if args.quick:
        args.epochs = 1
        if args.max_train == 0:
            args.max_train = 4000
        args.sleep_steps = 60

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, P = 4, 8   # 32x32 -> сетка 4x4 из патчей 8x8
    collate = make_collate(G, P)

    ds_cls = datasets.CIFAR100 if args.dataset == "cifar100" else datasets.CIFAR10
    tf = transforms.ToTensor()
    base_train = ds_cls(root=args.data, train=True, download=True, transform=tf)
    base_test = ds_cls(root=args.data, train=False, download=True, transform=tf)
    num_classes = len(base_train.classes)
    cps = num_classes // args.steps
    assert num_classes % args.steps == 0, "число классов должно делиться на шаги"

    train_labels = np.array(base_train.targets)
    test_labels = np.array(base_test.targets)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(base_train))
    train_perm = perm[:args.max_train] if args.max_train > 0 else perm

    model = build_model(cps, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] {args.dataset}: {num_classes} классов, {args.steps} шагов × {cps}, "
          f"batch={args.batch}, epochs={args.epochs}, sleep_steps={args.sleep_steps}, "
          f"no_sleep={args.no_sleep}, device={device}, params={n_params/1e6:.2f}M")

    A = []           # переклассовая точность по шагам
    history = []     # логи шагов
    probe, ref_ids = None, None
    t_start = time.time()

    for s in range(args.steps):
        lo, hi = s * cps, (s + 1) * cps
        new_cls = list(range(lo, hi))
        log = {"step": s + 1, "classes": f"{lo}-{hi - 1}"}
        t0 = time.time()

        train_ds = CifarSubset(base_train, train_perm[np.isin(train_labels[train_perm], range(lo, hi))])
        test_ds = CifarSubset(base_test, np.where(np.isin(test_labels, range(hi)))[0])
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  collate_fn=collate, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate)
        log["n_train"] = len(train_ds)

        if s > 0:
            model.expand_classifier(hi)

        if not args.no_sleep and s > 0:
            sample = collect(train_loader, args.sleep_batch, device)
            model.tokenizer.daytime_allocate_orphans({"image": sample["image"]})
            model.sleep_phase_consolidation(
                {"image": sample["image"]}, sample["labels"],
                old_classes=list(range(lo)), new_classes=new_cls,
                steps=args.sleep_steps)
            log["sleep_s"] = round(time.time() - t0, 1)
            log["calib_beta"] = round(float(model.calib_beta.item()), 4)

        # Оптимизатор пересобираем после expand — у классификатора новые веса
        decay, no_decay = [], []
        for p in model.parameters():
            if p.requires_grad:
                (decay if p.ndim >= 2 else no_decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

        t0 = time.time()
        for ep in range(args.epochs):
            log[f"loss_ep{ep + 1}"] = round(train_step(model, train_loader, opt, device), 4)
        log["train_s"] = round(time.time() - t0, 1)

        t0 = time.time()
        per_class = evaluate_per_class(model, test_loader, device)
        A.append(per_class)
        log["eval_s"] = round(time.time() - t0, 1)
        log["acc_seen"] = round(float(np.mean([per_class[c] for c in range(hi)])), 4)

        # Диагностика
        tok = model.tokenizer.mods["image"]
        usage = tok.code_usage
        log["codes_used"] = int((usage > 1.0).sum())
        log["frozen_codes"] = int(tok.codebook_frozen.sum())
        log["frozen_slots"] = int(tok.slot_frozen.sum())
        if probe is None:
            probe = collect(test_loader, 64, device)
            with torch.no_grad():
                _, _, _, _, auxs = model.tokenizer({"image": probe["image"]})
            ref_ids = {n: auxs[n]["ids"].reshape(auxs[n]["ids"].shape[0], -1)
                       for n in model.tokenizer.names}   # VQ-id на патч (B, L)
        else:
            log["id_stability_vs_s1"] = round(
                float(model.tokenizer.id_stability({"image": probe["image"]}, ref_ids)["image"]), 4)

        # Конец шага: заморозка anchors + снапшот снов (реплей для следующего sleep)
        if not args.no_sleep:
            snap = collect(train_loader, max(args.snap_batch, args.num_dream_templates), device)
            model.tokenizer.freeze_anchors()
            model.tokenizer.snapshot_anchor_habits({"image": snap["image"]})

        AA, BWT, Fgt = cl_metrics(A, cps)
        log["AA"], log["BWT"], log["forgetting"] = round(AA, 4), round(BWT, 4), round(Fgt, 4)
        log["elapsed_s"] = round(time.time() - t_start, 1)
        history.append(log)
        print(json.dumps(log, ensure_ascii=False), flush=True)

    final_acc = float(np.mean([A[-1][c] for c in range(num_classes)]))
    AA, BWT, Fgt = cl_metrics(A, cps)
    result = {
        "dataset": args.dataset, "tag": args.tag or ("no-sleep" if args.no_sleep else "run"),
        "seed": args.seed, "no_sleep": args.no_sleep, "epochs": args.epochs,
        "steps": args.steps, "sleep_steps": args.sleep_steps, "batch": args.batch,
        "d_model": args.d_model, "num_slots": args.num_slots,
        "max_emissions": args.max_emissions, "num_fused": args.num_fused,
        "max_train": args.max_train, "n_params": n_params,
        "final_acc": round(final_acc, 4), "AA": round(AA, 4),
        "BWT": round(BWT, 4), "forgetting": round(Fgt, 4),
        "history": history,
        "A": {str(s + 1): A[s] for s in range(len(A))},
    }
    out = Path("results")
    out.mkdir(exist_ok=True)
    fname = out / f"{args.dataset}_{result['tag']}.json"
    fname.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== {args.dataset} [{result['tag']}] ===")
    print(f"final acc (all {num_classes}): {final_acc:.4f} | AA={AA:.4f} BWT={BWT:.4f} "
          f"Forgetting={Fgt:.4f} | time={result['history'][-1]['elapsed_s']}s")
    print(f"saved: {fname}")


if __name__ == "__main__":
    main()
