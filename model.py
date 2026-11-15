import copy, itertools, math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NDSeparableConv(nn.Module):
    """N-D свёртка: 1D depthwise вдоль каждой оси + pointwise. (B,D,*spatial)->(B,D,*spatial)."""
    def __init__(self, d_model, n_dims, kernel_size=3, pointwise=True):
        super().__init__()
        self.n_dims = n_dims
        self.axis_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2,
                      groups=d_model, bias=False) for _ in range(n_dims)])
        self.pw = nn.Linear(d_model, d_model, bias=False) if pointwise else nn.Identity()

    def forward(self, h):
        B, D = h.shape[:2]
        for axis, conv in enumerate(self.axis_convs):
            h = h.movedim(2 + axis, -1)
            rest = h.shape[2:-1]
            h = conv(h.movedim(1, -2).reshape(-1, D, h.shape[-1]))
            h = h.reshape(B, *rest, D, h.shape[-1]).movedim(-2, 1).movedim(-1, 2 + axis)
        return self.pw(h.movedim(1, -1)).movedim(-1, 1)


class CayleyRotation(nn.Module):
    """Ортогональная ротация SO(D): R=(I+A)^{-1}(I-A), A=skew-skew^T. Init: skew=0 -> R=I."""
    def __init__(self, d_model):
        super().__init__()
        self.skew = nn.Parameter(torch.zeros(d_model, d_model))
        self.register_buffer("eye", torch.eye(d_model))

    def forward(self, x):
        A = self.skew - self.skew.T
        R = torch.linalg.solve(self.eye + A, self.eye - A)
        return x @ R.T


def nd_maxpool(x, k=3):
    n = x.ndim - 2
    if n <= 3:
        return getattr(F, f"max_pool{n}d")(x, k, stride=1, padding=k // 2)
    p = k // 2
    xp = F.pad(x, (p,) * (2 * n), value=float("-inf"))
    out = torch.full_like(x, float("-inf"))
    for sh in itertools.product(range(k), repeat=n):
        sl = (slice(None), slice(None)) + tuple(slice(t, t + s) for t, s in zip(sh, x.shape[2:]))
        out = torch.maximum(out, xp[sl])
    return out


def nd_avgpool(x, k=3):
    n = x.ndim - 2
    if n <= 3:
        return getattr(F, f"avg_pool{n}d")(x, k, stride=1, padding=k // 2, count_include_pad=False)
    p = k // 2
    xp = F.pad(x, (p,) * (2 * n), value=0.0)
    op = F.pad(torch.ones_like(x), (p,) * (2 * n), value=0.0)
    s = torch.zeros_like(x); c = torch.zeros_like(x)
    for sh in itertools.product(range(k), repeat=n):
        sl = (slice(None), slice(None)) + tuple(slice(t, t + sz) for t, sz in zip(sh, x.shape[2:]))
        s = s + xp[sl]; c = c + op[sl]
    return s / c.clamp(min=1)


def nd_positional_encoding(coords, d_model, mode="linear", normalize=True):
    n_dims = coords.shape[-1]
    per_dim = (d_model // n_dims) // 2 * 2
    if per_dim < 2:
        raise ValueError("d_model слишком мал для n_dims")
    nf = per_dim // 2
    dtype = coords.dtype if coords.is_floating_point() else torch.float32
    if mode == "linear":
        freqs = torch.linspace(1., 3., nf, device=coords.device, dtype=dtype) * math.pi
        x = coords - 0.5
    else:
        freqs = 12. * torch.exp(torch.arange(0, per_dim, 2, device=coords.device, dtype=dtype)
                                * (-math.log(100.) / per_dim))
        x = coords
    ang = x.unsqueeze(-1) * freqs
    pe = torch.stack([ang.sin(), ang.cos()], -1).flatten(-3)
    if pe.shape[-1] < d_model:
        pe = F.pad(pe, (0, d_model - pe.shape[-1]))
    return F.normalize(pe, dim=-1) if normalize else pe


import copy, itertools, math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NDSeparableConv(nn.Module):
    """N-D свёртка: 1D depthwise вдоль каждой оси + pointwise. (B,D,*spatial)->(B,D,*spatial)."""
    def __init__(self, d_model, n_dims, kernel_size=3, pointwise=True):
        super().__init__()
        self.n_dims = n_dims
        self.axis_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2,
                      groups=d_model, bias=False) for _ in range(n_dims)])
        self.pw = nn.Linear(d_model, d_model, bias=False) if pointwise else nn.Identity()

    def forward(self, h):
        B, D = h.shape[:2]
        for axis, conv in enumerate(self.axis_convs):
            h = h.movedim(2 + axis, -1)
            rest = h.shape[2:-1]
            h = conv(h.movedim(1, -2).reshape(-1, D, h.shape[-1]))
            h = h.reshape(B, *rest, D, h.shape[-1]).movedim(-2, 1).movedim(-1, 2 + axis)
        return self.pw(h.movedim(1, -1)).movedim(-1, 1)


class CayleyRotation(nn.Module):
    """Ортогональная ротация SO(D): R=(I+A)^{-1}(I-A), A=skew-skew^T. Init: skew=0 -> R=I."""
    def __init__(self, d_model):
        super().__init__()
        self.skew = nn.Parameter(torch.zeros(d_model, d_model))
        self.register_buffer("eye", torch.eye(d_model))

    def forward(self, x):
        A = self.skew - self.skew.T
        R = torch.linalg.solve(self.eye + A, self.eye - A)
        return x @ R.T


def nd_maxpool(x, k=3):
    n = x.ndim - 2
    if n <= 3:
        return getattr(F, f"max_pool{n}d")(x, k, stride=1, padding=k // 2)
    p = k // 2
    xp = F.pad(x, (p,) * (2 * n), value=float("-inf"))
    out = torch.full_like(x, float("-inf"))
    for sh in itertools.product(range(k), repeat=n):
        sl = (slice(None), slice(None)) + tuple(slice(t, t + s) for t, s in zip(sh, x.shape[2:]))
        out = torch.maximum(out, xp[sl])
    return out


def nd_avgpool(x, k=3):
    n = x.ndim - 2
    if n <= 3:
        return getattr(F, f"avg_pool{n}d")(x, k, stride=1, padding=k // 2, count_include_pad=False)
    p = k // 2
    xp = F.pad(x, (p,) * (2 * n), value=0.0)
    op = F.pad(torch.ones_like(x), (p,) * (2 * n), value=0.0)
    s = torch.zeros_like(x); c = torch.zeros_like(x)
    for sh in itertools.product(range(k), repeat=n):
        sl = (slice(None), slice(None)) + tuple(slice(t, t + sz) for t, sz in zip(sh, x.shape[2:]))
        s = s + xp[sl]; c = c + op[sl]
    return s / c.clamp(min=1)


def nd_positional_encoding(coords, d_model, mode="linear", normalize=True):
    n_dims = coords.shape[-1]
    per_dim = (d_model // n_dims) // 2 * 2
    if per_dim < 2:
        raise ValueError("d_model слишком мал для n_dims")
    nf = per_dim // 2
    dtype = coords.dtype if coords.is_floating_point() else torch.float32
    if mode == "linear":
        freqs = torch.linspace(1., 3., nf, device=coords.device, dtype=dtype) * math.pi
        x = coords - 0.5
    else:
        freqs = 12. * torch.exp(torch.arange(0, per_dim, 2, device=coords.device, dtype=dtype)
                                * (-math.log(100.) / per_dim))
        x = coords
    ang = x.unsqueeze(-1) * freqs
    pe = torch.stack([ang.sin(), ang.cos()], -1).flatten(-3)
    if pe.shape[-1] < d_model:
        pe = F.pad(pe, (0, d_model - pe.shape[-1]))
    return F.normalize(pe, dim=-1) if normalize else pe


class SymbolicDyTokTokenizer(nn.Module):
    """
    Слой 1 обучаемый; слой 2 гейтит по ID (сумки кодов), слот = разреженная комбинация ID.
    forward -> (tokens, slot_ids, coords, presence, aux)
    """
    def __init__(self, raw_patch_dim, d_model=64, n_dims=2, vocab=512, num_slots=24,
                 anchor_ratio=0.6, max_emissions=12, window=3, ids_per_slot=4,
                 num_dream_templates=128, use_offset_bags=False, min_peak=0.05,
                 temperature=0.10, ema_decay=0.99, dead_thresh=1.0):
        super().__init__()
        self.d_model, self.n_dims, self.vocab = d_model, n_dims, vocab
        self.num_slots = num_slots
        self.num_anchors = int(num_slots * anchor_ratio)
        self.num_reserve = num_slots - self.num_anchors
        self.max_emissions, self.window, self.ids_per_slot = max_emissions, window, ids_per_slot
        self.num_dream_templates = num_dream_templates
        self.use_offset_bags, self.min_peak, self.temperature = use_offset_bags, min_peak, temperature
        self.ema_decay, self.dead_thresh = ema_decay, dead_thresh
        self.bag_dim = vocab * (window ** n_dims) if use_offset_bags else vocab

        # --- Слой 1: ОБУЧАЕМЫЙ ---
        self.byte_embed = nn.Embedding(256, d_model)
        nn.init.normal_(self.byte_embed.weight)
        self.byte_embed.weight.data = F.normalize(self.byte_embed.weight.data, dim=-1)
        self.patch_proj = nn.Linear(raw_patch_dim, d_model, bias=False)
        self.patch_proj.weight.data = F.normalize(self.patch_proj.weight.data, dim=0)
        self.local_context = NDSeparableConv(d_model, n_dims, kernel_size=3)

        # --- VQ-кодбук (буфер, EMA; anchor-строки заморожены) ---
        self.register_buffer("codebook", F.normalize(torch.randn(vocab, d_model), dim=-1))
        self.register_buffer("code_ema_count", torch.ones(vocab))
        self.register_buffer("code_ema_weight", self.codebook.clone())
        self.register_buffer("code_usage", torch.zeros(vocab))
        self.register_buffer("codebook_frozen", torch.zeros(vocab, dtype=torch.bool))

        # --- Слоты = комбинации ID (обучаемые веса над алфавитом) ---
        w = torch.randn(num_slots, self.bag_dim) * 0.01
        self.slot_weights = nn.Parameter(w)
        self.register_buffer("slot_active", torch.zeros(num_slots, dtype=torch.bool))
        self.slot_active[: self.num_anchors] = True
        self.register_buffer("slot_frozen", torch.zeros(num_slots, dtype=torch.bool))
        self.register_buffer("lru_usage", torch.zeros(num_slots))
        self.sink_threshold = nn.Parameter(torch.tensor(0.25))

        # --- Память снов ---
        self.register_buffer("saved_dream_slots", torch.zeros(num_dream_templates, max_emissions, dtype=torch.long))
        self.register_buffer("saved_dream_coords", torch.zeros(num_dream_templates, max_emissions, n_dims))
        self.register_buffer("saved_dream_pres", torch.ones(num_dream_templates, max_emissions))

        self.concat_proj = nn.Linear(2 * d_model, d_model, bias=False)
        nn.init.orthogonal_(self.concat_proj.weight)
        self.sparsify_slots()

    # ---------- слой 1 ----------
    def layer1_encode(self, x, patch_feat=None, with_pe=False, pe_scale=None, pe_mode=None):
        B = x.shape[0]; spatial = x.shape[1:]; n = len(spatial)
        if n != self.n_dims:
            raise ValueError(f"ожидалось {self.n_dims}D, получено {n}D")
        L = math.prod(spatial)
        raw = self.byte_embed(x)
        if patch_feat is not None:
            raw = raw + self.patch_proj(patch_feat)
        raw = F.normalize(raw, dim=-1)
        h = self.local_context(raw.movedim(-1, 1)).movedim(1, -1)
        conv_flat = F.normalize(h.reshape(B, L, self.d_model), dim=-1)
        mean = F.normalize(conv_flat.mean(1, keepdim=True), dim=-1)
        salience = 1.0 - (conv_flat * mean).sum(-1).clamp(min=0.0)
        axes = [torch.linspace(0., 1., s, device=x.device) for s in spatial]
        coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(1, L, n).expand(B, L, n)
        h1 = None
        if with_pe:
            pe_mode = pe_mode or ("log" if n == 1 else "linear")
            pe_scale = pe_scale if pe_scale is not None else (0.15 if n == 1 else 0.20)
            pe = nd_positional_encoding(coords, self.d_model, mode=pe_mode, normalize=(pe_mode == "linear"))
            h1 = F.normalize(conv_flat + pe_scale * pe, dim=-1)
        return h1, conv_flat, salience, coords

    # ---------- VQ ----------
    def quantize(self, conv_flat, update_ema=True):
        c = F.normalize(self.codebook, dim=-1)
        ids = (F.normalize(conv_flat, dim=-1) @ c.T).argmax(-1)      # (B,L) — дискретно
        q = self.codebook[ids]
        vec_st = conv_flat + (q - conv_flat).detach()                # straight-through
        commit = F.mse_loss(conv_flat, q.detach())
        if self.training and update_ema:
            self._ema_codebook(conv_flat, ids)
        return ids, vec_st, commit

    @torch.no_grad()
    def _ema_codebook(self, vecs, ids):
        flat = vecs.reshape(-1, self.d_model); idf = ids.reshape(-1)
        oh = F.one_hot(idf, self.vocab).float()
        counts = oh.sum(0); sums = oh.T @ flat
        d = self.ema_decay
        self.code_ema_count.mul_(d).add_(counts, alpha=1 - d)
        self.code_ema_weight.mul_(d).add_(sums, alpha=1 - d)
        new = self.code_ema_weight / self.code_ema_count.clamp(min=1e-5).unsqueeze(-1)
        upd = (~self.codebook_frozen) & (self.code_ema_count > 1e-3)
        self.codebook[upd] = F.normalize(new[upd], dim=-1)
        self.code_usage.mul_(0.99).add_(counts, alpha=0.01)
        dead = (~self.codebook_frozen) & (self.code_ema_count < self.dead_thresh)
        if dead.any() and flat.size(0) > 0:
            di = torch.where(dead)[0]
            src = flat[torch.randint(0, flat.size(0), (di.numel(),), device=flat.device)]
            self.codebook[di] = F.normalize(src, dim=-1)
            self.code_ema_count[di] = 1.0
            self.code_ema_weight[di] = self.codebook[di]

    # ---------- сумки ID и гейтинг ----------
    def id_bag(self, ids, spatial):
        oh = F.one_hot(ids, self.vocab).movedim(-1, 1).float()       # (B,V,*spatial)
        if not self.use_offset_bags:
            h = nd_avgpool(oh, self.window)
        else:
            p = self.window // 2
            ohp = F.pad(oh, (p,) * (2 * self.n_dims), value=0.0)
            feats = []
            for sh in itertools.product(range(self.window), repeat=self.n_dims):
                sl = (slice(None), slice(None)) + tuple(slice(t, t + s) for t, s in zip(sh, spatial))
                feats.append(ohp[sl])
            h = torch.cat(feats, dim=1)                              # (B,V*k^n,*spatial)
        return F.normalize(h.flatten(2).transpose(1, 2), dim=-1)     # (B,L,bag_dim)

    def gate(self, bag, salience):
        w = F.normalize(self.slot_weights, dim=-1)
        sim = (bag @ w.T).masked_fill(~self.slot_active.view(1, 1, -1), -1e9)
        B, L, _ = sim.shape
        sink = self.sink_threshold.expand(B, L, 1)
        probs = F.softmax(torch.cat([sim, sink], -1) / self.temperature, -1)[..., :-1]
        return probs * (0.5 + salience.unsqueeze(-1)), probs

    def slot_vector(self, slot_ids):
        w = self.slot_weights[slot_ids]                              # (B,K,bag_dim)
        if self.use_offset_bags:
            w = w.view(*w.shape[:-1], self.window ** self.n_dims, self.vocab).sum(-2)
        return F.normalize(w @ self.codebook, dim=-1)                # комбинация ID -> вектор

    def compose_slot_output(self, z_q, v_agg, coords, presence):
        what = F.normalize(v_agg + 0.35 * self.concat_proj(torch.cat([z_q, v_agg], -1)), dim=-1)
        where = nd_positional_encoding(coords, self.d_model)
        return F.normalize(what + 0.25 * where, dim=-1) * presence.unsqueeze(-1)

    # ---------- forward ----------
    def forward(self, inputs):
        byte_grid, patch_feat = inputs
        B = byte_grid.shape[0]; spatial = byte_grid.shape[1:]; L = math.prod(spatial)
        _, conv_flat, salience, coords = self.layer1_encode(byte_grid, patch_feat)
        ids, vec_st, commit = self.quantize(conv_flat)
        bag = self.id_bag(ids.view(B, *spatial), spatial)
        weighted, probs = self.gate(bag, salience)
        max_prob, best_slot = weighted.max(-1)

        ramp = torch.arange(L, device=byte_grid.device, dtype=max_prob.dtype) * 1e-6
        nms = max_prob - ramp.unsqueeze(0)
        pooled = nd_maxpool(nms.view(B, 1, *spatial), self.window).view(B, L)
        is_peak = (nms == pooled) & (max_prob > self.min_peak)
        peak_scores = max_prob * is_peak.float()

        K = self.max_emissions
        top_scores, peak_pos = torch.topk(peak_scores, K, dim=-1)
        raster = torch.argsort(peak_pos, dim=-1)
        bidx = torch.arange(B, device=byte_grid.device).unsqueeze(-1)
        ordered_pos = peak_pos[bidx, raster]
        presence = top_scores[bidx, raster]
        slot_ids = best_slot[bidx, ordered_pos]

        z_q = self.slot_vector(slot_ids)
        sim_occ = torch.matmul(z_q, conv_flat.transpose(1, 2))
        radius = 3.0 / (L ** (1.0 / self.n_dims))
        dist_mask = (torch.cdist(coords[bidx, ordered_pos], coords) <= radius)
        dist_mask = dist_mask | F.one_hot(ordered_pos, L).bool()
        alpha = F.softmax(sim_occ.masked_fill(~dist_mask, -1e9) / self.temperature, -1)
        v_agg = F.normalize(alpha @ vec_st, dim=-1)
        occ_coords = alpha @ coords

        if self.training:
            with torch.no_grad():
                self.lru_usage.mul_(0.95).add_(probs.mean((0, 1)), alpha=0.05)

        tokens = self.compose_slot_output(z_q, v_agg, occ_coords, presence)
        aux = {"vq_commit": commit, "sparsity": self.slot_weights.abs().mean(),
               "ids": ids.view(B, *spatial)}
        return tokens, slot_ids, occ_coords, presence, aux

    # ---------- обслуживание слотов / кодбука ----------
    @torch.no_grad()
    def sparsify_slots(self):
        w = self.slot_weights.data
        keep = w.abs().topk(self.ids_per_slot, dim=-1).indices
        self.slot_weights.data = w * torch.zeros_like(w).scatter_(-1, keep, 1.0)

    def mask_frozen_grads(self):
        if self.slot_weights.grad is not None:
            self.slot_weights.grad[self.slot_frozen] = 0.0

    @torch.no_grad()
    def freeze_anchors(self):
        self.slot_frozen |= self.slot_active
        self.codebook_frozen |= (self.code_usage > self.dead_thresh)

    @torch.no_grad()
    def daytime_allocate_orphans(self, inputs, top_k_global=16):
        _, conv_flat, salience, _ = self.layer1_encode(*inputs)
        B, L, D = conv_flat.shape; spatial = inputs[0].shape[1:]
        ids, _, _ = self.quantize(conv_flat, update_ema=False)
        bag = self.id_bag(ids.view(B, *spatial), spatial)
        is_bg = salience < salience.mean(1, keepdim=True)
        self.lru_usage.mul_(0.5)
        reserve_ids = torch.arange(self.num_anchors, self.num_slots, device=conv_flat.device)

        for _ in range(self.num_reserve):
            cand = reserve_ids[~self.slot_frozen[reserve_ids]]
            if cand.numel() == 0:
                break
            active = torch.where(self.slot_active)[0]
            w = F.normalize(self.slot_weights[active], dim=-1)
            cov = (bag @ w.T).max(-1)[0]
            orphan = (1 - cov) * (~is_bg).float() * salience
            val, fidx = orphan.view(-1).max(0)
            if val.item() < 0.12:
                break
            b, t = (fidx // L).item(), (fidx % L).item()
            seed = bag[b, t]
            new_w = torch.zeros_like(seed)
            top = seed.abs().topk(self.ids_per_slot).indices
            new_w[top] = seed[top]
            new_w = F.normalize(new_w, dim=-1)
            for idx in active:
                u = F.normalize(self.slot_weights[idx], dim=-1)
                dot = torch.dot(new_w, u)
                if dot > 0.30:
                    new_w = F.normalize(new_w - 0.65 * dot * u, dim=-1)
            vs = self.lru_usage[cand].clone()
            vs[~self.slot_active[cand]] = -1.0
            slot_id = cand[vs.argmin()].item()
            self.slot_weights.data[slot_id] = new_w
            self.slot_active[slot_id] = True
            self.lru_usage[slot_id] = 1.0
            is_bg = is_bg | ((bag @ new_w) > 0.55)

    @torch.no_grad()
    def refresh_reserve_slots(self, inputs, top_k=48):
        _, conv_flat, _, _ = self.layer1_encode(*inputs)
        B, L, D = conv_flat.shape; spatial = inputs[0].shape[1:]
        ids, _, _ = self.quantize(conv_flat, update_ema=False)
        bag = self.id_bag(ids.view(B, *spatial), spatial).reshape(-1, self.bag_dim)
        for s in range(self.num_anchors, self.num_slots):
            if self.slot_active[s] and not self.slot_frozen[s]:
                sim = bag @ F.normalize(self.slot_weights[s], dim=-1)
                _, ti = torch.topk(sim, min(top_k, bag.size(0)))
                mean_bag = F.normalize(bag[ti].mean(0), dim=-1)
                new_w = torch.zeros_like(mean_bag)
                top = mean_bag.abs().topk(self.ids_per_slot).indices
                new_w[top] = mean_bag[top]
                self.slot_weights.data[s] = F.normalize(new_w, dim=-1)

    @torch.no_grad()
    def id_stability(self, inputs, ref_ids):
        _, conv_flat, _, _ = self.layer1_encode(*inputs)
        ids, _, _ = self.quantize(conv_flat, update_ema=False)
        return (ids == ref_ids).float().mean().item()

    @torch.no_grad()
    def snapshot_anchor_habits(self, inputs_p1):
        was = self.training; self.eval()
        n = self.num_dream_templates
        _, ids, coords, pres, _ = self((inputs_p1[0][:n], inputs_p1[1][:n]))
        self.saved_dream_slots.copy_(ids)
        self.saved_dream_coords.copy_(coords)
        self.saved_dream_pres.copy_(pres)
        self.train(was)

    def generate_latent_dreams(self, batch_size=64):
        device = self.saved_dream_slots.device
        idx = torch.randint(0, self.num_dream_templates, (batch_size,), device=device)
        slot_ids = self.saved_dream_slots[idx]
        coords = (self.saved_dream_coords[idx]
                  + 0.02 * torch.randn(batch_size, self.max_emissions, self.n_dims, device=device)).clamp(0, 1)
        pres = self.saved_dream_pres[idx]
        z_q = self.slot_vector(slot_ids)
        v_agg = F.normalize(z_q + 0.04 * torch.randn_like(z_q), dim=-1)
        return self.compose_slot_output(z_q, v_agg, coords, pres)


class CrossAttentionFusion(nn.Module):
    """
    tokens: dict name -> (B, K_i, D), уже умноженные на presence (norm == presence).
    Возвращает fused (B, num_fused, D). Маскирование padding по norm, защита от NaN.
    """
    def __init__(self, d_model, n_heads, num_fused, names, dropout=0.0):
        super().__init__()
        self.names = list(names)
        self.mod_embed = nn.ParameterDict({n: nn.Parameter(torch.zeros(d_model)) for n in self.names})
        self.cross = nn.ModuleDict({
            n: nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for n in self.names})
        self.cross_norm = nn.ModuleDict({n: nn.LayerNorm(d_model) for n in self.names})
        self.queries = nn.Parameter(F.normalize(torch.randn(num_fused, d_model), dim=-1))
        self.pool = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                 nn.Linear(2 * d_model, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _safe_mask(mask):
        # если у строки все ключи замаскированы — размаскировать, иначе softmax даст NaN
        return mask & ~mask.all(dim=-1, keepdim=True)

    def forward(self, tokens):
        names = list(tokens.keys())
        B = next(iter(tokens.values())).size(0)
        streams, masks = {}, {}
        for n in names:
            t = tokens[n]
            streams[n] = t + self.mod_embed[n]                 # маркер модальности
            masks[n] = t.norm(dim=-1) < 1e-4                   # presence≈0 -> padding

        # 1) каждый поток cross-attend к остальным (если поток один — к себе)
        updated = {}
        for n in names:
            others = [m for m in names if m != n] or [n]
            kv = torch.cat([streams[m] for m in others], dim=1)
            kpm = self._safe_mask(torch.cat([masks[m] for m in others], dim=1))
            ctx, _ = self.cross[n](streams[n], kv, kv, key_padding_mask=kpm, need_weights=False)
            updated[n] = self.cross_norm[n](streams[n] + ctx)

        # 2) сжатие обучаемыми query в фиксированную длину
        kv_all = torch.cat([updated[n] for n in names], dim=1)
        kpm_all = self._safe_mask(torch.cat([masks[n] for n in names], dim=1))
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool(q, kv_all, kv_all, key_padding_mask=kpm_all, need_weights=False)
        fused = self.pool_norm(q + pooled)
        fused = self.ffn_norm(fused + self.ffn(fused))
        return F.normalize(fused, dim=-1)

class MultiTokenizer(nn.Module):
    def __init__(self, tokenizers: dict, d_model=64, n_heads=4, num_fused=12):
        super().__init__()
        self.mods = nn.ModuleDict(tokenizers)          # name -> SymbolicDyTokTokenizer
        self.names = list(tokenizers.keys())
        self.fusion = CrossAttentionFusion(d_model, n_heads, num_fused, self.names)

    def forward(self, inputs: dict):
        tokens, slot_ids, coords, presence, auxs = {}, {}, {}, {}, {}
        for n in self.names:
            t, sid, c, p, aux = self.mods[n](inputs[n])
            tokens[n], slot_ids[n], coords[n], presence[n], auxs[n] = t, sid, c, p, aux
        fused = self.fusion(tokens)
        return fused, slot_ids, coords, presence, auxs

    # --- сны: каждый токенизатор генерит свои, фузим тем же модулем ---
    def generate_latent_dreams(self, batch_size=64):
        dreams = {n: self.mods[n].generate_latent_dreams(batch_size) for n in self.names}
        return self.fusion(dreams)

    # --- обслуживание: по каждому токенизатору независимо ---
    def refresh_reserve_slots(self, inputs: dict, top_k=48):
        for n in self.names:
            self.mods[n].refresh_reserve_slots(inputs[n], top_k)

    def daytime_allocate_orphans(self, inputs: dict, **kw):
        for n in self.names:
            self.mods[n].daytime_allocate_orphans(inputs[n], **kw)

    def snapshot_anchor_habits(self, inputs: dict):
        for n in self.names:
            self.mods[n].snapshot_anchor_habits(inputs[n])

    def freeze_anchors(self):
        for n in self.names:
            self.mods[n].freeze_anchors()

    def mask_frozen_grads(self):
        for n in self.names:
            self.mods[n].mask_frozen_grads()

    def sparsify_slots(self):
        for n in self.names:
            self.mods[n].sparsify_slots()

    def id_stability(self, inputs: dict, ref_ids: dict):
        return {n: self.mods[n].id_stability(inputs[n], ref_ids[n]) for n in self.names}

    @staticmethod
    def slice_inputs(inputs: dict, n):
        return {name: (bg[:n], pf[:n] if pf is not None else None)
                for name, (bg, pf) in inputs.items()}

class RNNClassifier(nn.Module):
    def __init__(self, d_model=64, num_classes=4, num_old_classes=2, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer                     # MultiTokenizer (или одиночный)
        self.num_old_classes = num_old_classes
        self.core_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.novel_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.feature_rotator = CayleyRotation(d_model)
        self.classifier = nn.Linear(d_model, num_classes, bias=False)
        self.register_buffer("calib_gamma", torch.ones(1))
        self.register_buffer("calib_beta", torch.zeros(1))
        self.register_buffer("calib_idx", torch.arange(num_old_classes, num_classes))

    def head_logits(self, h, use_rotator=True, calibrate=False):
        if h.dim() == 3:
            h = h.squeeze(0)
        if use_rotator:
            h = self.feature_rotator(h)
        logits = 12.0 * F.normalize(h, dim=-1) @ F.normalize(self.classifier.weight, dim=-1).T
        if calibrate and self.calib_idx.numel() > 0:
            logits[:, self.calib_idx] = logits[:, self.calib_idx] * self.calib_gamma + self.calib_beta
        return logits

    def forward(self, inputs, hidden=None):
        if not torch.is_tensor(inputs):                # dict (multi) или tuple (single)
            if self.tokenizer is None:
                raise ValueError("Подключите tokenizer или передайте тензор токенов")
            inputs = self.tokenizer(inputs)[0]         # fused (B, K_fused, D)
        _, h = self.core_gru(inputs, hidden)
        return self.head_logits(h, use_rotator=True, calibrate=True)

    @staticmethod
    def remap_labels(y_local, env_classes):
        return torch.as_tensor(env_classes, dtype=torch.long, device=y_local.device)[y_local]

    @torch.no_grad()
    def expand_classifier(self, new_num_classes):
        old = self.classifier
        if new_num_classes <= old.out_features:
            return
        new = nn.Linear(old.in_features, new_num_classes, bias=False).to(old.weight.device)
        new.weight.data = F.normalize(torch.randn_like(new.weight.data), dim=-1)
        new.weight.data[: old.out_features] = old.weight.data
        self.classifier = new

    @staticmethod
    def _suppress_margin(logits, suppress_idx, ref_max, gap=3.5):
        if suppress_idx.numel() == 0:
            return logits.sum() * 0.0
        return F.relu(logits.index_select(1, suppress_idx) - (ref_max - gap)).mean()

    def sleep_phase_consolidation(self, inputs_phase2, y_phase2, old_classes=None,
                                  new_classes=None, steps=160, shared_grad_scale=0.3):
        assert self.tokenizer is not None
        tok = self.tokenizer
        device = next(iter(inputs_phase2.values()))[0].device     # dict входов
        y_phase2 = torch.as_tensor(y_phase2, dtype=torch.long, device=device)
        if old_classes is None:
            old_classes = list(range(self.num_old_classes))
        if new_classes is None:
            new_classes = list(range(self.num_old_classes, self.classifier.out_features))
        old_idx = torch.as_tensor(old_classes, dtype=torch.long, device=device)
        new_idx = torch.as_tensor(new_classes, dtype=torch.long, device=device)
        assert torch.isin(y_phase2, torch.cat([old_idx, new_idx])).all()

        present = torch.unique(y_phase2)
        shared_idx = old_idx[torch.isin(old_idx, present)]
        absent_idx = old_idx[~torch.isin(old_idx, present)]
        vis_idx = torch.cat([shared_idx, new_idx]).sort().values
        g2v = torch.full((self.classifier.out_features,), -1, dtype=torch.long, device=device)
        g2v[vis_idx] = torch.arange(vis_idx.numel(), device=device)

        teacher_gru = copy.deepcopy(self.core_gru).eval()
        w_old_frozen = F.normalize(self.classifier.weight.data[old_idx].clone(), dim=-1)

        with torch.no_grad():
            n = tok.num_dream_templates if hasattr(tok, "num_dream_templates") else \
                next(iter(tok.mods.values())).num_dream_templates
            sub = MultiTokenizer.slice_inputs(inputs_phase2, n)
            tok.refresh_reserve_slots(sub)                          # по каждому токенизатору
            comp_phase2 = tok(inputs_phase2)[0]                     # fused

        # 1. novel_gru + головы видимых классов
        self.novel_gru.requires_grad_(True).train()
        opt_novel = torch.optim.Adam(list(self.novel_gru.parameters()) + [self.classifier.weight], lr=0.008)
        for _ in range(90):
            idx = torch.randint(0, comp_phase2.size(0), (64,), device=device)
            opt_novel.zero_grad()
            _, h_nov = self.novel_gru(comp_phase2[idx])
            F.cross_entropy(self.head_logits(h_nov, use_rotator=False).index_select(1, vis_idx),
                            g2v[y_phase2[idx]]).backward()
            self.classifier.weight.grad[absent_idx] = 0.0
            self.classifier.weight.grad[shared_idx] *= shared_grad_scale
            opt_novel.step()

        # 2. Сброс отсутствующих старых голов + ортогонализация новых
        with torch.no_grad():
            keep = ~torch.isin(old_idx, present)
            self.classifier.weight.data[absent_idx] = w_old_frozen[keep]
            for c in new_idx.tolist():
                w_c = self.classifier.weight.data[c]
                for o in old_idx.tolist():
                    u = F.normalize(self.classifier.weight.data[o], dim=-1)
                    w_c = w_c - torch.dot(w_c, u) * u
                self.classifier.weight.data[c] = F.normalize(w_c, dim=-1)
        self.novel_gru.eval().requires_grad_(False)

        # 3. Двухконтурная дистилляция (сны фузятся тем же модулем)
        opt_sleep = torch.optim.Adam(list(self.core_gru.parameters())
                                     + list(self.feature_rotator.parameters()), lr=0.005)
        for _ in range(steps):
            opt_sleep.zero_grad()
            x_dream = tok.generate_latent_dreams(64)                  # fused сны
            with torch.no_grad():
                _, h_t = teacher_gru(x_dream)
                target_old = 12.0 * F.normalize(h_t.squeeze(0), dim=-1) @ w_old_frozen.T
                pseudo = target_old.argmax(-1)
            _, h_d = self.core_gru(x_dream)
            ld = self.head_logits(h_d)
            dream_old = ld.index_select(1, old_idx)
            loss_kd_p1 = F.mse_loss(dream_old, target_old) + F.cross_entropy(dream_old, pseudo)
            m1 = self._suppress_margin(ld, new_idx, dream_old.max(-1, keepdim=True)[0].detach())

            idx = torch.randint(0, comp_phase2.size(0), (64,), device=device)
            with torch.no_grad():
                _, h_nov = self.novel_gru(comp_phase2[idx])
                target_vis = self.head_logits(h_nov, use_rotator=False).index_select(1, vis_idx)
            _, h_r = self.core_gru(comp_phase2[idx])
            lr_ = self.head_logits(h_r)
            real_vis = lr_.index_select(1, vis_idx)
            loss_ce = F.cross_entropy(real_vis, g2v[y_phase2[idx]])
            loss_kd_p2 = F.mse_loss(real_vis, target_vis)
            m2 = self._suppress_margin(lr_, absent_idx, real_vis.max(-1, keepdim=True)[0].detach())

            (1.5 * loss_kd_p1 + loss_ce + 1.0 * loss_kd_p2 + 2.0 * (m1 + m2)).backward()
            opt_sleep.step()

        # 4. Калибровка только новых колонок
        with torch.no_grad():
            self.calib_idx = new_idx.clone()
            self.calib_gamma.fill_(1.0); self.calib_beta.fill_(0.0)
            mask_new = torch.isin(y_phase2, new_idx)
            if new_idx.numel() > 0 and mask_new.any():
                _, h_d = self.core_gru(tok.generate_latent_dreams(200))
                l_d = self.head_logits(h_d)
                e1 = (l_d.index_select(1, old_idx).max(-1)[0] - l_d.index_select(1, new_idx).max(-1)[0]).mean()
                _, h_r = self.core_gru(comp_phase2[mask_new][:200])
                l_r = self.head_logits(h_r)
                e2 = (l_r.index_select(1, new_idx).max(-1)[0] - l_r.index_select(1, old_idx).max(-1)[0]).mean()
                self.calib_beta.fill_(0.5 * (e1 - e2).item())
            tok.freeze_anchors()

if __name__ == "__main__":
    def make_multitok_model(raw_patch_dims: dict, d_model=64, num_classes=4, num_old_classes=2,
                            n_heads=4, num_fused=12, tok_kwargs=None):
        """raw_patch_dims: name -> (raw_patch_dim, n_dims). Пример: {"seq": (16,1), "grid": (16,2)}."""
        toks = {}
        for name, (rpd, nd) in raw_patch_dims.items():
            kw = (tok_kwargs or {}).get(name, {})
            toks[name] = SymbolicDyTokTokenizer(rpd, d_model=d_model, n_dims=nd, **kw)
        tok = MultiTokenizer(toks, d_model=d_model, n_heads=n_heads, num_fused=num_fused)
        return RNNClassifier(d_model, num_classes, num_old_classes, tokenizer=tok)
    
    
    model = make_multitok_model(
        raw_patch_dims={"seq": (16, 1), "grid": (16, 2)},
        d_model=64, num_classes=2, num_old_classes=2, n_heads=4, num_fused=12,
        tok_kwargs={"seq":  {"vocab": 256, "max_emissions": 12},
                    "grid": {"vocab": 512, "max_emissions": 12}})
    
    inputs = {"seq":  (byte_seq,  patch_seq),       # byte_seq: (B, L)
              "grid": (byte_grid, patch_grid)}      # byte_grid: (B, H, W)
    
    # --- Этап 1: сквозное обучение (токенизаторы + фузия + RNN) ---
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for inputs, y in phase1_loader:
        opt.zero_grad()
        fused, slot_ids, coords, presence, auxs = model.tokenizer(inputs)
        logits = model(fused)
        loss = (F.cross_entropy(logits, y)
                + 0.25 * sum(a["vq_commit"] for a in auxs.values())
                + 1e-3 * sum(a["sparsity"] for a in auxs.values()))
        loss.backward()
        model.tokenizer.mask_frozen_grads()
        opt.step()
        model.tokenizer.sparsify_slots()
    
    # --- Конец фазы 1 ---
    model.tokenizer.freeze_anchors()
    model.tokenizer.snapshot_anchor_habits(inputs_phase1)      # dict по модальностям
    
    # --- Новая среда ---
    model.tokenizer.daytime_allocate_orphans(inputs_phase2)    # по каждому токенизатору
    model.expand_classifier(5)
    y_glob = RNNClassifier.remap_labels(y_local_phase2, [1, 2, 3, 4])
    model.sleep_phase_consolidation(inputs_phase2, y_glob,
                                    old_classes=[0, 1], new_classes=[2, 3, 4])