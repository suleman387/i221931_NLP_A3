# ============================================================
# CELL 3 — VOCABULARY & DATASET
# Build vocab from TRAINING data only; convert tokens → indices
# ============================================================

# ---------- BUILD VOCABULARY ----------
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
EOS_TOKEN = '<EOS>'
BOS_TOKEN = '<BOS>'

counter = Counter()
for rec in train_data:
    counter.update(simple_tokenize(rec['text']))
    counter.update(simple_tokenize(rec['explanation']))

# Most common tokens up to cap (reserve 4 special tokens)
most_common = counter.most_common(VOCAB_SIZE_CAP - 4)
vocab = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] + [tok for tok, _ in most_common]

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

PAD_IDX = word2idx[PAD_TOKEN]
UNK_IDX = word2idx[UNK_TOKEN]
BOS_IDX = word2idx[BOS_TOKEN]
EOS_IDX = word2idx[EOS_TOKEN]

ACTUAL_VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {ACTUAL_VOCAB_SIZE}")

# ---------- ENCODE FUNCTION ----------
def encode(text, max_len=MAX_SEQ_LEN):
    tokens = simple_tokenize(text)[:max_len]
    ids    = [word2idx.get(t, UNK_IDX) for t in tokens]
    # Pad / truncate to exactly max_len
    ids   += [PAD_IDX] * (max_len - len(ids))
    return ids[:max_len]

def encode_with_eos(text, max_len=MAX_SEQ_LEN):
    """For decoder: prepend BOS, append EOS, then pad."""
    tokens = simple_tokenize(text)[: max_len - 2]
    ids    = [BOS_IDX] + [word2idx.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
    ids   += [PAD_IDX] * (max_len - len(ids))
    return ids[:max_len]

# ---------- PYTORCH DATASET ----------
class ReviewDataset(Dataset):
    """Returns (encoder_input, sentiment_label, category_label, decoder_input, decoder_target)."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        enc  = torch.tensor(encode(rec['text']),               dtype=torch.long)
        sent = torch.tensor(rec['sentiment'],                  dtype=torch.long)
        cat  = torch.tensor(rec['category'],                   dtype=torch.long)
        # Decoder target is the synthesized explanation
        dec_full   = encode_with_eos(rec['explanation'], MAX_SEQ_LEN)
        dec_in     = torch.tensor(dec_full[:-1], dtype=torch.long)  # teacher-forced input
        dec_target = torch.tensor(dec_full[1:],  dtype=torch.long)  # one-step shifted target
        return enc, sent, cat, dec_in, dec_target

BATCH_SIZE = 64

train_ds = ReviewDataset(train_data)
val_ds   = ReviewDataset(val_data)
test_ds  = ReviewDataset(test_data)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
