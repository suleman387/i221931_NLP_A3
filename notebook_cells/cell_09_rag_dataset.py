# ============================================================
# CELL 9 — RAG TEMPLATE CONSTRUCTION + DECODER DATASET
#
# Template (all components concatenated into one token sequence):
#   [REVIEW] <review tokens>
#   [SENTIMENT] <sentiment string>
#   [CATEGORY] <category string>
#   [CONTEXT] <top-k retrieved summary tokens>
#   [EXPLANATION] <target explanation tokens + EOS>
#
# The decoder is trained to predict the [EXPLANATION] portion
# autoregressively using teacher forcing.
# ============================================================

# Special marker tokens (added to vocab as pseudo-tokens)
MARKERS = ['[REVIEW]', '[SENTIMENT]', '[CATEGORY]', '[CONTEXT]', '[EXPLANATION]']
for m in MARKERS:
    if m not in word2idx:
        word2idx[m] = len(word2idx)
        idx2word[len(idx2word)] = m
        vocab.append(m)

ACTUAL_VOCAB_SIZE = len(vocab)  # update after adding marker tokens
# Re-tie lm_head weight after vocab expansion
decoder_model.embedding = nn.Embedding(ACTUAL_VOCAB_SIZE, DEC_D_MODEL, padding_idx=PAD_IDX).to(DEVICE)
decoder_model.lm_head   = nn.Linear(DEC_D_MODEL, ACTUAL_VOCAB_SIZE, bias=False).to(DEVICE)
decoder_model.lm_head.weight = decoder_model.embedding.weight

def build_rag_sequence(record, use_retrieval=True, max_total=DEC_MAX_LEN):
    """
    Constructs the full input token sequence for the decoder:
      [REVIEW] <enc_of_review> [SENTIMENT] <sent_str> [CATEGORY] <cat_str>
      [CONTEXT] <retrieved_summaries> [EXPLANATION] <target_explanation> <EOS>

    Parameters
    ----------
    record        : dict with keys text, sentiment, category, summary, explanation
    use_retrieval : bool — if False, [CONTEXT] section is empty (ablation)
    max_total     : maximum total token count (hard truncation)
    """
    tokens = []

    # 1. Review text (first 40 tokens)
    tokens += [word2idx['[REVIEW]']]
    tokens += [word2idx.get(t, UNK_IDX) for t in simple_tokenize(record['text'])[:40]]

    # 2. Predicted sentiment (string label)
    sent_str = SENTIMENT_LABELS[record['sentiment']]
    tokens += [word2idx['[SENTIMENT]']]
    tokens += [word2idx.get(t, UNK_IDX) for t in simple_tokenize(sent_str)]

    # 3. Predicted category (string label) — the derived text-only feature
    cat_str = CATEGORY_LABELS[record['category']]
    tokens += [word2idx['[CATEGORY]']]
    tokens += [word2idx.get(t, UNK_IDX) for t in simple_tokenize(cat_str)]

    # 4. Retrieved context (top-k summaries), empty if use_retrieval=False
    tokens += [word2idx['[CONTEXT]']]
    if use_retrieval:
        encoder_model.eval()
        with torch.no_grad():
            enc_in = torch.tensor(encode(record['text'])).unsqueeze(0).to(DEVICE)
            _, _, q_emb = encoder_model(enc_in)
        idx_k, _ = retrieve_top_k(q_emb.squeeze(0), TOP_K)
        for ridx in idx_k:
            ctx_summary = train_data[ridx.item()]['summary']
            tokens += [word2idx.get(t, UNK_IDX) for t in simple_tokenize(ctx_summary)[:8]]

    # 5. Target explanation
    tokens += [word2idx['[EXPLANATION]']]
    summary_tokens = [word2idx.get(t, UNK_IDX) for t in simple_tokenize(record['explanation'])[:30]]
    tokens += summary_tokens + [EOS_IDX]

    # Pad / truncate to max_total
    tokens = tokens[:max_total]
    tokens += [PAD_IDX] * (max_total - len(tokens))
    return tokens


class RAGDataset(Dataset):
    """
    Decoder training dataset.
    Input:  full RAG sequence tokens [0 .. S-2]
    Target: shifted by 1            [1 .. S-1]
    (Standard causal LM objective)
    """
    def __init__(self, records, use_retrieval=True):
        self.records       = records
        self.use_retrieval = use_retrieval

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        seq = build_rag_sequence(self.records[idx], self.use_retrieval)
        t   = torch.tensor(seq, dtype=torch.long)
        return t[:-1], t[1:]   # (input, target)


print("Building RAG datasets (this may take a few minutes for retrieval) ...")
# For speed, build a small RAG training set (10k) and test set (2k)
RAG_TRAIN_N = min(10000, len(train_data))
RAG_TEST_N  = min(2000,  len(test_data))

rag_train_ds = RAGDataset(train_data[:RAG_TRAIN_N], use_retrieval=True)
rag_test_ds  = RAGDataset(test_data[:RAG_TEST_N],   use_retrieval=True)
rag_no_ret_ds= RAGDataset(test_data[:RAG_TEST_N],   use_retrieval=False)  # ablation

RAG_BATCH = 32
rag_train_loader  = DataLoader(rag_train_ds,   batch_size=RAG_BATCH, shuffle=True,  num_workers=0)
rag_test_loader   = DataLoader(rag_test_ds,    batch_size=RAG_BATCH, shuffle=False, num_workers=0)
rag_no_ret_loader = DataLoader(rag_no_ret_ds,  batch_size=RAG_BATCH, shuffle=False, num_workers=0)

print(f"RAG train: {len(rag_train_ds)} | RAG test: {len(rag_test_ds)}")
