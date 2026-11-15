(byte_grid, patch_feat)
   │  Слой 1 (обучаемый): byte_embed + patch_proj + NDSeparableConv
   ▼
conv_flat (B,L,D) ──► VQ: ids = argmax cos(conv_flat, codebook)   [дискретный интерфейс]
   │                        │  straight-through vec_st + commitment + EMA кодбука
   ▼                        ▼
id_bag (B,L,V·k^n) ──► гейт: cos(bag, slot_weights)  [слот = разреженная комбинация ID]
   │  sink + softmax + salience
   ▼
N-D NMS (рампа + min_peak) → K пиков → slot_ids, presence, coords
   │  z_q = normalize(W_q @ codebook);  v_agg = агрегация vec_st в окрестности
   ▼
compose: [z_q‖v_agg] + PE → концепт-токены (B,K,D)
   ▼
GRU (core) → CayleyRotation SO(D) → cosine-головы → logits

---

inputs = {"seq": (byte_seq, patch_seq), "grid": (byte_grid, patch_grid)}
   │  каждый токенизатор независимо: слой1→VQ→ID-гейт→NMS→K_i токенов
   ▼
tokens_i (B,K_i,D) + presence_i
   │  CrossAttentionFusion:
   │   1) каждый поток cross-attend к ДРУГИМ потокам
   │   2) обучаемые fusion-query сжимают всё в K_fused токенов
   ▼
fused (B,K_fused,D) ──► GRU → CayleyRotation → cosine-головы