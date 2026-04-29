# ============================================================
# CELL 5 — ENCODER TRAINING (Part A)
# Multi-task loss = α·CE_sentiment + (1-α)·CE_category
# LR schedule: linear warmup → cosine decay
# ============================================================

ENC_EPOCHS    = 5
ENC_LR        = 3e-4
ALPHA         = 0.6        # weight for sentiment loss
WARMUP_STEPS  = 500

optimizer_enc = torch.optim.AdamW(encoder_model.parameters(), lr=ENC_LR, weight_decay=1e-2)

def lr_lambda(step):
    if step == 0: step = 1
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, 10000 - WARMUP_STEPS)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler_enc = torch.optim.lr_scheduler.LambdaLR(optimizer_enc, lr_lambda)
ce_loss       = nn.CrossEntropyLoss(ignore_index=-1)

def run_encoder_epoch(loader, train=True):
    encoder_model.train(train)
    total_loss, sent_correct, cat_correct, total = 0., 0, 0, 0
    with torch.set_grad_enabled(train):
        for enc_in, sent_lbl, cat_lbl, _, _ in loader:
            enc_in   = enc_in.to(DEVICE)
            sent_lbl = sent_lbl.to(DEVICE)
            cat_lbl  = cat_lbl.to(DEVICE)

            sent_logits, cat_logits, _ = encoder_model(enc_in)

            loss_sent = ce_loss(sent_logits, sent_lbl)
            loss_cat  = ce_loss(cat_logits,  cat_lbl)
            loss      = ALPHA * loss_sent + (1 - ALPHA) * loss_cat

            if train:
                optimizer_enc.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(encoder_model.parameters(), 1.0)
                optimizer_enc.step()
                scheduler_enc.step()

            bs = enc_in.size(0)
            total_loss  += loss.item() * bs
            sent_correct += (sent_logits.argmax(-1) == sent_lbl).sum().item()
            cat_correct  += (cat_logits.argmax(-1)  == cat_lbl).sum().item()
            total        += bs

    return total_loss/total, sent_correct/total, cat_correct/total

enc_train_losses, enc_val_losses = [], []
enc_train_sent_acc, enc_val_sent_acc = [], []

print("Training Encoder ...")
for epoch in range(1, ENC_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_sa, tr_ca = run_encoder_epoch(train_loader, train=True)
    vl_loss, vl_sa, vl_ca = run_encoder_epoch(val_loader,   train=False)
    enc_train_losses.append(tr_loss);  enc_val_losses.append(vl_loss)
    enc_train_sent_acc.append(tr_sa);  enc_val_sent_acc.append(vl_sa)
    print(f"Epoch {epoch}/{ENC_EPOCHS} | {time.time()-t0:.0f}s | "
          f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
          f"SentAcc {tr_sa:.3f}/{vl_sa:.3f} | CatAcc {vl_ca:.3f}")

torch.save(encoder_model.state_dict(), 'models/encoder.pt')
print("Encoder weights saved to models/encoder.pt")
