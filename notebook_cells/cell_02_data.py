# ============================================================
# CELL 2 — DATA LOADING & PREPROCESSING
# Parses JSON-Lines files, balances categories, cleans text
# ============================================================

# ---------- CONFIG ----------
FILES = {
    'Beauty':       'Beauty_5.json',
    'Cellphones':   'Cell_Phones_and_Accessories_5.json',
    'Electronics':  'Electronics_5.json',
}
SAMPLES_PER_CAT = 15000   # 3 cats × 15k = 45k total
MAX_SEQ_LEN     = 128
VOCAB_SIZE_CAP  = 30000  # cap vocabulary

# ---------- SENTIMENT MAPPING ----------
def rating_to_sentiment(r):
    r = int(r)
    if r <= 2: return 0   # Negative
    if r == 3: return 1   # Neutral
    return 2              # Positive

SENTIMENT_LABELS = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
CATEGORY_LABELS  = {i: k for i, k in enumerate(FILES.keys())}
CAT2IDX          = {k: i for i, k in enumerate(FILES.keys())}

# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML
    text = re.sub(r"[^a-z0-9\s']", ' ', text)    # keep letters/digits/apostrophe
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- TOKENIZATION ----------
def simple_tokenize(text):
    return text.split()

# ---------- LOAD DATA ----------
print("Loading data ...")
all_reviews = []

for cat, fname in FILES.items():
    samples = []
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rt = rec.get('reviewText', '')
            ov = rec.get('overall', None)
            sm = rec.get('summary', '')
            if not rt or ov is None:
                continue
                
            # Synthesize a 1-2 sentence explanation to meet the PDF requirement
            sentiment_lbl = rating_to_sentiment(ov)
            sent_str = SENTIMENT_LABELS[sentiment_lbl].lower()
            clean_sum = clean_text(sm)
            if not clean_sum:
                clean_sum = "it met their expectations" if sentiment_lbl == 2 else "they had issues with it"
                
            explanation = clean_text(f"The review expresses a {sent_str} sentiment because the user noted that {clean_sum}")
            
            samples.append({
                'text':        clean_text(rt),
                'sentiment':   sentiment_lbl,
                'category':    CAT2IDX[cat],
                'summary':     clean_sum,
                'explanation': explanation,
                'raw_text':    rt[:300],
            })
            if len(samples) >= SAMPLES_PER_CAT:
                break
    print(f"  {cat}: {len(samples)} samples loaded")
    all_reviews.extend(samples)

random.shuffle(all_reviews)
print(f"\nTotal samples: {len(all_reviews)}")

# ---------- TRAIN/VAL/TEST SPLIT ----------
n     = len(all_reviews)
n_tr  = int(0.70 * n)
n_val = int(0.15 * n)

train_data = all_reviews[:n_tr]
val_data   = all_reviews[n_tr : n_tr + n_val]
test_data  = all_reviews[n_tr + n_val:]
print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
