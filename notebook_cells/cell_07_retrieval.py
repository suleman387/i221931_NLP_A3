# ============================================================
# CELL 7 — RETRIEVAL MODULE (Part B)
# Cosine similarity search over stored training embeddings
# ============================================================

TOP_K = 5   # configurable: number of nearest neighbours to retrieve

# Load saved embeddings (or use in-memory tensors)
stored_embs = train_embeddings.to(DEVICE)   # (N_train, d_model)
# L2-normalise for fast cosine similarity via matrix multiplication
stored_embs_norm = F.normalize(stored_embs, p=2, dim=-1)

def retrieve_top_k(query_emb, k=TOP_K):
    """
    query_emb: (d_model,) tensor
    Returns indices and cosine similarity scores of top-k matches.
    Cosine similarity = (q · k) / (||q|| ||k||)
    Using pre-normalised embeddings: cos_sim = q_norm · K_norm^T
    """
    q_norm  = F.normalize(query_emb.unsqueeze(0), p=2, dim=-1)  # (1, d_model)
    sims    = (q_norm @ stored_embs_norm.T).squeeze(0)           # (N_train,)
    top_k_sims, top_k_idx = torch.topk(sims, k)
    return top_k_idx.cpu(), top_k_sims.cpu()

# --- Retrieval Quality Analysis on Test Set ---
print("Analysing retrieval quality on 200 test samples ...")
encoder_model.eval()
same_sentiment_hits = 0
same_category_hits  = 0
n_eval = min(200, len(test_data))
all_top1_sims = []

# Build index arrays for fast label lookup
train_sent_arr = np.array([r['sentiment'] for r in train_data])
train_cat_arr  = np.array([r['category']  for r in train_data])

with torch.no_grad():
    for i in range(n_eval):
        rec     = test_data[i]
        enc_in  = torch.tensor(encode(rec['text'])).unsqueeze(0).to(DEVICE)
        _, _, q_emb = encoder_model(enc_in)
        q_emb   = q_emb.squeeze(0)

        idx, sims = retrieve_top_k(q_emb, TOP_K)
        all_top1_sims.append(sims[0].item())

        # Check if top-1 retrieved sample shares sentiment / category
        same_sentiment_hits += int(train_sent_arr[idx[0].item()] == rec['sentiment'])
        same_category_hits  += int(train_cat_arr[idx[0].item()]  == rec['category'])

print(f"Top-1 Sentiment Match Rate: {same_sentiment_hits/n_eval:.3f}")
print(f"Top-1 Category  Match Rate: {same_category_hits/n_eval:.3f}")
print(f"Mean Top-1 Cosine Similarity: {np.mean(all_top1_sims):.4f}")

# --- Visualise similarity distribution ---
plt.figure(figsize=(7, 4))
plt.hist(all_top1_sims, bins=30, color='steelblue', edgecolor='white')
plt.title('Top-1 Retrieval Cosine Similarity Distribution')
plt.xlabel('Cosine Similarity'); plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('results/retrieval_similarity_hist.png', dpi=120)
plt.show()
print("Saved: results/retrieval_similarity_hist.png")

# --- Show 3 example retrievals ---
print("\n--- Example Retrievals ---")
with torch.no_grad():
    for i in range(3):
        rec    = test_data[i]
        enc_in = torch.tensor(encode(rec['text'])).unsqueeze(0).to(DEVICE)
        _, _, q_emb = encoder_model(enc_in)
        idx, sims   = retrieve_top_k(q_emb.squeeze(0), TOP_K)
        print(f"\nQuery [{SENTIMENT_LABELS[rec['sentiment']]} | {CATEGORY_LABELS[rec['category']]}]:")
        print(f"  '{rec['raw_text'][:100]}...'")
        for rank, (ridx, rsim) in enumerate(zip(idx[:3], sims[:3]), 1):
            r2 = train_data[ridx.item()]
            print(f"  Rank {rank} (sim={rsim:.4f}) [{SENTIMENT_LABELS[r2['sentiment']]}]: '{r2['raw_text'][:80]}...'")
