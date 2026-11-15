# train_food101.py
import csv
import html
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import MultiTokenizer, RNNClassifier, SymbolicDyTokTokenizer

# ==========================================
# 1. Конфигурация
# ==========================================
CFG = {
    # локальный датасет: картинки в dataset/images/{split}/{category}/,
    # заголовки в dataset/texts/{split}_titles.csv
    "dataset_root": "dataset",
    "split_train": "train",
    "split_val": "test",
    "num_classes": 101,
    # текст: сырые UTF-8 байты
    "max_bytes": 128,
    "text_tokens": 16,          # max_emissions 1D-токенизатора
    "text_feat_dim": 4,
    # изображение: сетка патчей
    "grid": 16,                 # 16x16 = 256 патчей
    "patch_size": 8,            # картинка ресайзится в 128x128
    "image_tokens": 16,
    "image_feat_dim": 8,
    # токенизаторы и cross-attention
    "d_model": 128,
    "num_slots": 32,
    "num_heads": 8,
    "num_fused": 16,
    # обучение
    "batch_size": 32,
    "num_workers": 0,
    "epochs": 10,
    "lr": 3e-4,                 # обучение с нуля, не fine-tune
    "weight_decay": 1e-2,
    "warmup_ratio": 0.1,
    "grad_clip": 1.0,
    "save_path": "best_local_food101_dytok.pt",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

IMG_SIZE = CFG["grid"] * CFG["patch_size"]


# ==========================================
# 2. Препроцессинг: байты текста + сетка патчей
# ==========================================
def text_to_bytes(text: str) -> torch.Tensor:
    b = text.encode("utf-8", errors="ignore")[: CFG["max_bytes"]]
    out = torch.zeros(CFG["max_bytes"], dtype=torch.uint8)   # паддинг нулями
    if len(b):
        out[: len(b)] = torch.tensor(list(b), dtype=torch.uint8)
    return out


def text_features(text_bytes: torch.Tensor) -> torch.Tensor:
    b = text_bytes.float()
    alpha = ((b >= 65) & (b <= 90)) | ((b >= 97) & (b <= 122))
    digit = (b >= 48) & (b <= 57)
    space = (b == 32) | (b == 10) | (b == 9)
    return torch.stack([b / 255.0, alpha.float(), digit.float(), space.float()], dim=-1)


def image_to_grid(pil_img):
    """PIL -> (byte_grid [G,G] uint8, patch_feat [G,G,8])."""
    G, P = CFG["grid"], CFG["patch_size"]
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)).div_(255.0).permute(2, 0, 1)  # (3,S,S)

    patches = F.unfold(x.unsqueeze(0), kernel_size=P, stride=P)[0].view(3, P * P, G * G)
    mean, std = patches.mean(dim=1), patches.std(dim=1)          # (3, L) каждый

    lum = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]             # (S,S)
    gx = F.pad((lum[:, 1:] - lum[:, :-1]).abs(), (0, 1))[None, None]
    gy = F.pad((lum[1:, :] - lum[:-1, :]).abs(), (0, 0, 0, 1))[None, None]
    gx = F.unfold(gx, kernel_size=P, stride=P)[0].mean(dim=0)    # (L,)
    gy = F.unfold(gy, kernel_size=P, stride=P)[0].mean(dim=0)

    byte = (0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2])
    byte = byte.mul(255).round().clamp(0, 255).to(torch.uint8)
    feat = torch.stack([mean[0], mean[1], mean[2], std[0], std[1], std[2], gx, gy], dim=-1)
    return byte.view(G, G), feat.view(G, G, -1)


def collate_fn(batch):
    text_key = "text" if "text" in batch[0] else "title"
    tb, ib, iff, labels = [], [], [], []
    for item in batch:
        tb.append(text_to_bytes(str(item[text_key])))
        g, f = image_to_grid(item["image"])
        ib.append(g)
        iff.append(f)
        labels.append(int(item["label"]))

    text_bytes = torch.stack(tb)
    return {
        "text_bytes": text_bytes,                       # [B, L] uint8
        "text_feat": text_features(text_bytes),         # [B, L, 4]
        "image_bytes": torch.stack(ib),                 # [B, G, G] uint8
        "image_feat": torch.stack(iff),                 # [B, G, G, 8]
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def model_inputs(batch):
    """Вход модели — dict по модальностям: name -> (byte_grid, patch_feat).

    nn.Embedding требует Long-индексов, поэтому uint8-байты приводим к long.
    """
    return {
        "text": (batch["text_bytes"].long(), batch["text_feat"]),
        "image": (batch["image_bytes"].long(), batch["image_feat"]),
    }


# ==========================================
# 2b. Локальный датасет: CSV + папки с картинками
# ==========================================
class LocalFoodDataset(Dataset):
    """
    dataset/texts/{split}_titles.csv — строки без заголовка вида:
        apple_pie_851.jpg,Crock-Pot Ladies Crock-Pot Apple Pie Moonshine,apple_pie
    Картинка лежит в dataset/images/{split}/{category}/{filename}.
    Метка = порядковый номер категории в алфавитном порядке (совпадает с Food101).
    """

    def __init__(self, split, root="dataset"):
        self.img_root = Path(root) / "images" / split
        csv_path = Path(root) / "texts" / f"{split}_titles.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.items = [(fn, html.unescape(title), cat)
                          for fn, title, cat in csv.reader(f)]
        self.classes = sorted({cat for _, _, cat in self.items})
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        print(f"[{split}] примеров: {len(self.items)}, классов: {len(self.classes)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, title, cat = self.items[idx]
        img = Image.open(self.img_root / cat / fn).convert("RGB")
        return {"image": img, "title": title, "label": self.class2idx[cat]}


# ==========================================
# 3. Модель
# ==========================================
def make_multitok_model(raw_patch_dims: dict, d_model=64, num_classes=4, num_old_classes=2,
                        n_heads=4, num_fused=12, tok_kwargs=None):
    """raw_patch_dims: name -> (raw_patch_dim, n_dims). Пример: {"seq": (16,1), "grid": (16,2)}."""
    toks = {}
    for name, (rpd, nd) in raw_patch_dims.items():
        kw = (tok_kwargs or {}).get(name, {})
        toks[name] = SymbolicDyTokTokenizer(rpd, d_model=d_model, n_dims=nd, **kw)
    tok = MultiTokenizer(toks, d_model=d_model, n_heads=n_heads, num_fused=num_fused)
    return RNNClassifier(d_model, num_classes, num_old_classes, tokenizer=tok)


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==========================================
# 4. Обучение и валидация
# ==========================================
def compute_topk_accuracy(logits, labels, topk=(1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0).item() for k in topk]


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, amp_dtype):
    model.train()
    total_loss, correct_top1, total_samples = 0.0, 0.0, 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        batch = move_batch(batch, device)
        labels = batch["labels"]
        inp = model_inputs(batch)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            logits = model(inp)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct_top1 += compute_topk_accuracy(logits.detach(), labels)[0]
        total_samples += bs
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}",
                         acc=f"{correct_top1 / total_samples:.4f}")

    return total_loss / total_samples, correct_top1 / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype):
    model.eval()
    total_loss, correct_top1, correct_top5, total_samples = 0.0, 0.0, 0.0, 0

    for batch in tqdm(loader, desc="Val", leave=False):
        batch = move_batch(batch, device)
        labels = batch["labels"]
        inp = model_inputs(batch)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            logits = model(inp)
            loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        top1, top5 = compute_topk_accuracy(logits, labels)
        correct_top1 += top1
        correct_top5 += top5
        total_samples += bs

    return total_loss / total_samples, correct_top1 / total_samples, correct_top5 / total_samples


# ==========================================
# 5. Запуск
# ==========================================
def main():
    device = CFG["device"]
    train_ds = LocalFoodDataset(CFG["split_train"], root=CFG["dataset_root"])
    val_ds = LocalFoodDataset(CFG["split_val"], root=CFG["dataset_root"])

    loader_kw = dict(num_workers=CFG["num_workers"],
                     pin_memory=(device == "cuda"), collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, **loader_kw)
    sample = train_ds[0]
    print("Ключи элемента:", sample.keys())
    print("Содержимое:", sample)

    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, **loader_kw)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))

    model = make_multitok_model(
        raw_patch_dims={"text": (CFG["text_feat_dim"], 1),
                        "image": (CFG["image_feat_dim"], 2)},
        d_model=CFG["d_model"],
        num_classes=CFG["num_classes"],
        num_old_classes=CFG["num_classes"],   # не continual — все классы известны
        n_heads=CFG["num_heads"],
        num_fused=CFG["num_fused"],
        tok_kwargs={"text": {"max_emissions": CFG["text_tokens"], "num_slots": CFG["num_slots"]},
                    "image": {"max_emissions": CFG["image_tokens"], "num_slots": CFG["num_slots"]}},
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # weight_decay не применяем к скалярам/LayerNorm/bias; замороженные параметры исключены
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": CFG["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=CFG["lr"],
    )

    total_steps = len(train_loader) * CFG["epochs"]
    scheduler = cosine_with_warmup(optimizer, int(total_steps * CFG["warmup_ratio"]), total_steps)

    best_val_acc = 0.0
    for epoch in range(1, CFG["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, amp_dtype)
        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, criterion, device, amp_dtype)

        print(f"Epoch {epoch:02d}/{CFG['epochs']:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Top-1: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Top-1: {val_acc1*100:.2f}% | Val Top-5: {val_acc5*100:.2f}%")

        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "cfg": CFG,
                        "val_acc1": val_acc1, "val_acc5": val_acc5}, CFG["save_path"])
            print(f"  --> Сохранен новый лучший чекпоинт (Val Top-1: {best_val_acc*100:.2f}%)")


if __name__ == "__main__":
    main()