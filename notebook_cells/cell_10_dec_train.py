# ============================================================
# CELL 10 — DECODER TRAINING (Part C)
# Autoregressive LM with cross-entropy loss (teacher forcing)
# ============================================================

DEC_EPOCHS = 5
DEC_LR     = 3e-4

optimizer_dec = torch.optim.AdamW(decoder_model.parameters(), lr=DEC_LR, weight_decay=1e-2)
scheduler_dec = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dec, T_max=DEC_EPOCHS)

lm_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

def compute_perplexity(loss_val):
    """Perplexity = exp(cross-entropy loss)"""
    return math.exp(min(loss_val, 20))   # cap to avoid overflow

def run_decoder_epoch(loader, train=True):
    decoder_model.train(train)
    total_loss, total_tokens = 0., 0
    with torch.set_grad_enabled(train):
        for dec_in, dec_tgt in loader:
            dec_in  = dec_in.to(DEVICE)
            dec_tgt = dec_tgt.to(DEVICE)

            logits = decoder_model(dec_in)    # (B, S, V)
            B, S, V = logits.shape
            loss = lm_loss_fn(logits.reshape(B*S, V), dec_tgt.reshape(B*S))

            if train:
                optimizer_dec.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(decoder_model.parameters(), 1.0)
                optimizer_dec.step()

            # Count non-pad tokens for proper averaging
            n_tokens = (dec_tgt != PAD_IDX).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, compute_perplexity(avg_loss)

dec_train_losses, dec_train_ppls = [], []

print("Training Decoder ...")
for epoch in range(1, DEC_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_ppl = run_decoder_epoch(rag_train_loader, train=True)
    scheduler_dec.step()
    dec_train_losses.append(tr_loss)
    dec_train_ppls.append(tr_ppl)
    print(f"Epoch {epoch}/{DEC_EPOCHS} | {time.time()-t0:.0f}s | "
          f"LM Loss: {tr_loss:.4f} | Perplexity: {tr_ppl:.2f}")

torch.save(decoder_model.state_dict(), 'models/decoder.pt')
print("Decoder weights saved to models/decoder.pt")

# Plot decoder loss
plt.figure(figsize=(8, 4))
plt.plot(dec_train_losses, marker='o', label='Train LM Loss')
plt.title('Decoder (RAG) Training Loss'); plt.xlabel('Epoch'); plt.legend()
plt.tight_layout()
plt.savefig('results/decoder_loss_curve.png', dpi=120)
plt.show()
