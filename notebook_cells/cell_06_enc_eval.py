# ============================================================
# CELL 6 — ENCODER EVALUATION + LEARNING CURVES + SAVE EMBEDDINGS
# ============================================================

# --- Plot Learning Curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(enc_train_losses, label='Train Loss', marker='o')
axes[0].plot(enc_val_losses,   label='Val Loss',   marker='s')
axes[0].set_title('Encoder Loss'); axes[0].legend(); axes[0].set_xlabel('Epoch')

axes[1].plot(enc_train_sent_acc, label='Train Sent Acc', marker='o')
axes[1].plot(enc_val_sent_acc,   label='Val Sent Acc',   marker='s')
axes[1].set_title('Encoder Sentiment Accuracy'); axes[1].legend(); axes[1].set_xlabel('Epoch')
plt.tight_layout()
plt.savefig('results/encoder_learning_curves.png', dpi=120)
plt.show()
print("Saved: results/encoder_learning_curves.png")

# --- Test Set Evaluation ---
encoder_model.eval()
all_sent_preds, all_sent_true = [], []
all_cat_preds,  all_cat_true  = [], []

with torch.no_grad():
    for enc_in, sent_lbl, cat_lbl, _, _ in test_loader:
        enc_in = enc_in.to(DEVICE)
        sp, cp, _ = encoder_model(enc_in)
        all_sent_preds.extend(sp.argmax(-1).cpu().tolist())
        all_sent_true.extend(sent_lbl.tolist())
        all_cat_preds.extend(cp.argmax(-1).cpu().tolist())
        all_cat_true.extend(cat_lbl.tolist())

print("\n=== Sentiment Classification Report ===")
print(classification_report(all_sent_true, all_sent_preds,
      target_names=['Negative','Neutral','Positive']))

print("\n=== Category Classification Report ===")
print(classification_report(all_cat_true, all_cat_preds,
      target_names=list(FILES.keys())))

# --- Save Training Embeddings to results/ ---
print("\nExtracting and saving training embeddings ...")
encoder_model.eval()
train_embeddings, train_labels = [], []

EMBED_LOADER = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0)
with torch.no_grad():
    for enc_in, sent_lbl, cat_lbl, _, _ in EMBED_LOADER:
        _, _, emb = encoder_model(enc_in.to(DEVICE))
        train_embeddings.append(emb.cpu())
        train_labels.append(sent_lbl)

train_embeddings = torch.cat(train_embeddings, dim=0)  # (N_train, d_model)
train_labels     = torch.cat(train_labels,     dim=0)

np.save('results/train_embeddings.npy', train_embeddings.numpy())
np.save('results/train_sent_labels.npy', train_labels.numpy())
print(f"Saved embeddings: shape {train_embeddings.shape}")
