# ============================================================
# CELL 12 — BONUS: Interactive UI with ipywidgets
# Allows entering a custom review and visualising the full
# retrieval + generation pipeline step-by-step.
# ============================================================

import ipywidgets as widgets
from IPython.display import display, clear_output

# --- UI Layout ---
title_html = widgets.HTML(
    value="<h2 style='color:#4A90D9;font-family:sans-serif'>"
          "🔍 RAG Pipeline — Interactive Demo</h2>"
          "<p style='font-family:sans-serif;color:#555'>"
          "Enter a product review below. The pipeline will:<br>"
          "1. Encode it with the Transformer encoder<br>"
          "2. Predict sentiment + product category<br>"
          "3. Retrieve the top-5 most similar training reviews<br>"
          "4. Generate a 1-2 sentence explanation using the decoder</p>"
)

text_box = widgets.Textarea(
    value="This product was absolutely fantastic! Great build quality and fast delivery.",
    description='Review:',
    layout=widgets.Layout(width='90%', height='80px'),
    style={'description_width': 'initial'},
)

use_ret_toggle = widgets.Checkbox(value=True, description='Use Retrieval (RAG)')
run_btn        = widgets.Button(description='▶ Run Pipeline', button_style='primary',
                                 layout=widgets.Layout(width='200px'))
output_area    = widgets.Output(layout=widgets.Layout(border='1px solid #ccc',
                                 padding='10px', width='90%', max_height='500px',
                                 overflow_y='auto'))

def on_run_clicked(b):
    with output_area:
        clear_output(wait=True)
        user_text    = text_box.value.strip()
        use_retrieval= use_ret_toggle.value

        if not user_text:
            print("⚠️ Please enter a review text.")
            return

        # --- Step 1: Encode ---
        clean  = clean_text(user_text)
        enc_in = torch.tensor(encode(clean)).unsqueeze(0).to(DEVICE)
        encoder_model.eval()
        with torch.no_grad():
            s_logits, c_logits, q_emb = encoder_model(enc_in)
        pred_sent = s_logits.argmax(-1).item()
        pred_cat  = c_logits.argmax(-1).item()

        print(f"📌 Predicted Sentiment : {SENTIMENT_LABELS[pred_sent]}")
        print(f"📌 Predicted Category  : {CATEGORY_LABELS[pred_cat]}")

        # --- Step 2: Retrieval ---
        if use_retrieval:
            idx_k, sims_k = retrieve_top_k(q_emb.squeeze(0), TOP_K)
            print(f"\n🔎 Top-{TOP_K} Retrieved Reviews:")
            for rank, (ridx, rsim) in enumerate(zip(idx_k, sims_k), 1):
                r2 = train_data[ridx.item()]
                print(f"  [{rank}] sim={rsim:.4f} | {SENTIMENT_LABELS[r2['sentiment']]} | "
                      f"'{r2['raw_text'][:70]}...'")
        else:
            print("\n[Retrieval disabled — ablation mode]")

        # --- Step 3: Generate ---
        fake_record = {
            'text':        clean,
            'sentiment':   pred_sent,
            'category':    pred_cat,
            'summary':     '',
            'explanation': '',
            'raw_text':    user_text,
        }
        generated = generate(fake_record, use_retrieval=use_retrieval, max_new=40)
        print(f"\n✨ Generated Explanation: {generated}")

run_btn.on_click(on_run_clicked)

display(title_html, text_box, use_ret_toggle, run_btn, output_area)
