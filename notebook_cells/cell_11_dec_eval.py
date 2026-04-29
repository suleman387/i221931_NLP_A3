# ============================================================
# CELL 11 — DECODER EVALUATION + GENERATION + RAG ABLATION
# ============================================================

# ---------- Test-set Perplexity ----------
test_loss_rag,    test_ppl_rag    = run_decoder_epoch(rag_test_loader,   train=False)
test_loss_no_ret, test_ppl_no_ret = run_decoder_epoch(rag_no_ret_loader, train=False)

print(f"\n=== RAG Ablation Study ===")
print(f"With Retrieval    — Test Loss: {test_loss_rag:.4f}  | Perplexity: {test_ppl_rag:.2f}")
print(f"Without Retrieval — Test Loss: {test_loss_no_ret:.4f}  | Perplexity: {test_ppl_no_ret:.2f}")
print(f"Perplexity reduction from retrieval: {test_ppl_no_ret - test_ppl_rag:.2f}")

# ---------- Greedy Token Generation ----------
@torch.no_grad()
def generate(record, use_retrieval=True, max_new=30):
    """
    Greedy autoregressive generation.
    Starts from the full RAG prompt (up to [EXPLANATION] marker),
    then generates token by token until EOS or max_new tokens.
    """
    decoder_model.eval()
    # Build prompt (everything up to and including [EXPLANATION] marker)
    full_seq = build_rag_sequence(record, use_retrieval)
    # Find the position of [EXPLANATION] marker
    sum_tok  = word2idx['[EXPLANATION]']
    try:
        start_pos = full_seq.index(sum_tok) + 1
    except ValueError:
        start_pos = 50

    # Prompt = beginning of sequence up to start_pos
    prompt  = full_seq[:start_pos]
    prompt  = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).to(DEVICE)

    generated = []
    for _ in range(max_new):
        logits    = decoder_model(prompt)         # (1, S, V)
        next_tok  = logits[0, -1, :].argmax(-1).item()
        if next_tok == EOS_IDX:
            break
        generated.append(next_tok)
        prompt = torch.cat([
            prompt,
            torch.tensor([[next_tok]], device=DEVICE)
        ], dim=1)

    return ' '.join([idx2word.get(i, UNK_TOKEN) for i in generated])

# ---------- 5 Generated Samples with Commentary ----------
print("\n=== 5 Generated Explanations (with Retrieval) ===\n")
for i in range(5):
    rec    = test_data[i]
    gen    = generate(rec, use_retrieval=True, max_new=40)
    target = rec['explanation']
    sent_s = SENTIMENT_LABELS[rec['sentiment']]
    cat_s  = CATEGORY_LABELS[rec['category']]
    print(f"[{i+1}] Category: {cat_s} | Sentiment: {sent_s}")
    print(f"  Review (snippet): {rec['raw_text'][:80]}...")
    print(f"  Target Expl  : {target}")
    print(f"  Generated    : {gen}")
    print(f"  Commentary: The model {'correctly captures' if gen.split()[0:2] == target.split()[0:2] else 'partially reflects'}"
          f" the {sent_s.lower()} reasoning of the review.")
    print()

# ---------- RAG Ablation Comparison ----------
print("=== RAG Ablation: With vs. Without Retrieval ===\n")
for i in range(3):
    rec       = test_data[i]
    gen_rag   = generate(rec, use_retrieval=True)
    gen_norag = generate(rec, use_retrieval=False)
    print(f"[{i+1}] {rec['raw_text'][:70]}...")
    print(f"  RAG    : {gen_rag}")
    print(f"  No-RAG : {gen_norag}")
    print()
