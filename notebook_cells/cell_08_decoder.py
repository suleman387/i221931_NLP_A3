# ============================================================
# CELL 8 — DECODER-ONLY TRANSFORMER (Part C) — from SCRATCH
#
# Pure Causal Language Model with masked self-attention.
# NO nn.Transformer or any forbidden abstraction used.
#
# RAG TEMPLATE (prepended to decoder target):
# -----------------------------------------------
# [REVIEW] <review_text>
# [SENTIMENT] <predicted_sentiment>
# [CATEGORY] <predicted_category>
# [CONTEXT] <top-k retrieved summaries>
# [EXPLANATION]
# -----------------------------------------------
# The decoder is trained to predict tokens of the explanation
# conditioned on the above context window.
# ============================================================

DEC_D_MODEL  = 128
DEC_N_HEADS  = 4
DEC_D_FF     = 256
DEC_N_LAYERS = 2
DEC_DROPOUT  = 0.1
DEC_MAX_LEN  = 128


# ---------- CAUSAL MASK ----------
def make_causal_mask(seq_len, device):
    """
    Lower-triangular boolean mask to prevent attending to future positions.
    mask[i,j] = 1 if j <= i  (position i can attend to position j)
    Shape: (1, 1, seq_len, seq_len) for broadcasting over (B, H, S, S)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)


# ---------- DECODER BLOCK (causal self-attention only) ----------
class TransformerDecoderBlock(nn.Module):
    """
    Single decoder block: causal MHA → LayerNorm → FFN → LayerNorm
    Using pre-LN (more stable training).
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn  = PositionwiseFFN(d_model, d_ff, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        residual = x
        x = self.ln1(x)
        x = residual + self.drop(self.attn(x, x, x, causal_mask))
        residual = x
        x = self.ln2(x)
        x = residual + self.drop(self.ffn(x))
        return x


# ---------- DECODER-ONLY LANGUAGE MODEL ----------
class DecoderOnlyTransformer(nn.Module):
    """
    Stacked causal decoder blocks for autoregressive generation.
    forward() returns token logits: (B, S, vocab_size)
    """
    def __init__(self, vocab_size, d_model, n_heads, d_ff,
                 n_layers, max_len=DEC_MAX_LEN, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final  = nn.LayerNorm(d_model)
        self.lm_head   = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: share embedding and output projection weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, x):
        S = x.size(1)
        causal_mask = make_causal_mask(S, x.device)
        h = self.pos_enc(self.embedding(x) * math.sqrt(DEC_D_MODEL))
        for layer in self.layers:
            h = layer(h, causal_mask)
        h = self.ln_final(h)
        return self.lm_head(h)   # (B, S, vocab_size)

decoder_model = DecoderOnlyTransformer(
    vocab_size = ACTUAL_VOCAB_SIZE,
    d_model    = DEC_D_MODEL,
    n_heads    = DEC_N_HEADS,
    d_ff       = DEC_D_FF,
    n_layers   = DEC_N_LAYERS,
    pad_idx    = PAD_IDX,
    dropout    = DEC_DROPOUT,
).to(DEVICE)

dec_params = sum(p.numel() for p in decoder_model.parameters() if p.requires_grad)
print(f"Decoder parameters: {dec_params:,}")
