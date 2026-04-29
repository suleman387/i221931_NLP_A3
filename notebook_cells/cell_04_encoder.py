# ============================================================
# CELL 4 — ENCODER-ONLY TRANSFORMER (Part A) — from SCRATCH
#
# Architecture:
#   Embedding → Positional Encoding → N × TransformerEncoderBlock
#   → [CLS] representation → two linear heads:
#       head_sentiment  → 3-class softmax
#       head_category   → 5-class softmax
#
# NO nn.Transformer / nn.MultiheadAttention / nn.TransformerEncoder used.
# ============================================================

# ---------- HYPERPARAMETERS ----------
ENC_D_MODEL   = 128   # embedding / hidden dimension
ENC_N_HEADS   = 4     # attention heads (d_model must be divisible)
ENC_D_FF      = 256   # feed-forward inner dimension
ENC_N_LAYERS  = 2     # stacked encoder blocks
ENC_DROPOUT   = 0.1
N_SENTIMENT   = 3
N_CATEGORY    = len(FILES)  # 3

# ---------- POSITIONAL ENCODING ----------
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------- SCALED DOT-PRODUCT ATTENTION ----------
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
    Q,K,V shape: (batch, n_heads, seq_len, d_k)
    """
    d_k   = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, S, S)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn  = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn   # (B, H, S, d_k)


# ---------- MULTI-HEAD ATTENTION ----------
class MultiHeadAttention(nn.Module):
    """
    Projects input into Q, K, V via learned linear layers,
    splits into n_heads, computes attention in parallel, concatenates.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        # Separate projection matrices for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        # x: (B, S, d_model) → (B, n_heads, S, d_k)
        B, S, _ = x.size()
        return x.view(B, S, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))  # (B, H, S, d_k)
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        out, _ = scaled_dot_product_attention(Q, K, V, mask)  # (B, H, S, d_k)

        # Concatenate heads: (B, H, S, d_k) → (B, S, d_model)
        B, H, S, d_k = out.size()
        out = out.transpose(1, 2).contiguous().view(B, S, H * d_k)
        return self.W_o(out)


# ---------- POSITION-WISE FEED-FORWARD ----------
class PositionwiseFFN(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------- ENCODER BLOCK ----------
class TransformerEncoderBlock(nn.Module):
    """
    Pre-LN variant:
      x → LayerNorm → MHA → residual → LayerNorm → FFN → residual
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn  = PositionwiseFFN(d_model, d_ff, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention sublayer with residual
        residual = x
        x = self.ln1(x)
        x = residual + self.drop(self.attn(x, x, x, src_mask))
        # Feed-forward sublayer with residual
        residual = x
        x = self.ln2(x)
        x = residual + self.drop(self.ffn(x))
        return x


# ---------- FULL ENCODER MODEL ----------
class EncoderTransformer(nn.Module):
    """
    Encoder-only Transformer for multi-task classification.
    forward() returns tuple of 3:
        (sentiment_logits, category_logits, cls_embedding)
    """
    def __init__(self, vocab_size, d_model, n_heads, d_ff,
                 n_layers, n_sentiment, n_category,
                 max_len=MAX_SEQ_LEN, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final  = nn.LayerNorm(d_model)

        # Two separate classification heads (shared encoder backbone)
        self.head_sentiment = nn.Linear(d_model, n_sentiment)
        self.head_category  = nn.Linear(d_model, n_category)
        self.pad_idx = pad_idx

    def make_pad_mask(self, src):
        # src: (B, S) — mask positions where PAD token exists
        # Returns: (B, 1, 1, S) boolean mask (1 = valid, 0 = pad)
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, src):
        mask = self.make_pad_mask(src).to(src.device)
        x = self.embedding(src) * math.sqrt(ENC_D_MODEL)  # scale embedding
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        # Use the first token ([CLS]-like) as the fixed-dim representation
        cls_emb = x[:, 0, :]                              # (B, d_model)
        return self.head_sentiment(cls_emb), self.head_category(cls_emb), cls_emb

encoder_model = EncoderTransformer(
    vocab_size  = ACTUAL_VOCAB_SIZE,
    d_model     = ENC_D_MODEL,
    n_heads     = ENC_N_HEADS,
    d_ff        = ENC_D_FF,
    n_layers    = ENC_N_LAYERS,
    n_sentiment = N_SENTIMENT,
    n_category  = N_CATEGORY,
    pad_idx     = PAD_IDX,
    dropout     = ENC_DROPOUT,
).to(DEVICE)

total_params = sum(p.numel() for p in encoder_model.parameters() if p.requires_grad)
print(f"Encoder parameters: {total_params:,}")
