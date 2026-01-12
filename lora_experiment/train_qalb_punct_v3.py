#!/usr/bin/env python3
"""
Train punct classifier on QALB - v3.
Fix: Arabic punct is SPACE-SEPARATED, not attached to words.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from collections import Counter, defaultdict
from pathlib import Path
import pickle

QALB_TRAIN = "/home/ubuntu/nahawi/data/qalb_real_train.json"
QALB_DEV = "/home/ubuntu/nahawi/data/qalb_real_dev.json"
OUTPUT_DIR = "/home/ubuntu/nahawi/punct_classifier_qalb"

PUNCT_SET = set('،.؟!؛:,;?')
PUNCT_MAP = {
    '،': 0, ',': 0,
    '.': 1,
    '؟': 2, '?': 2,
    '!': 3,
    '؛': 4, ';': 4,
    ':': 5,
}
NUM_PUNCT = 6
NONE_IDX = NUM_PUNCT


def is_punct_token(token):
    """Check if token is purely punctuation."""
    return all(c in PUNCT_SET for c in token) and len(token) > 0


def get_punct_type(token):
    """Get punct type from token."""
    for c in token:
        if c in PUNCT_MAP:
            return PUNCT_MAP[c]
    return NONE_IDX


def extract_word_punct_pairs(text):
    """
    Extract (word, following_punct_type) pairs.
    Handles space-separated punct like: "كلمة ، كلمة"
    """
    tokens = text.split()
    pairs = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Skip pure punct tokens
        if is_punct_token(token):
            i += 1
            continue

        # This is a word - check if next token is punct
        punct_type = NONE_IDX
        if i + 1 < len(tokens) and is_punct_token(tokens[i + 1]):
            punct_type = get_punct_type(tokens[i + 1])

        # Also check if punct is attached to end of word
        word = token.rstrip('،.؟!؛:,;?')
        trailing_punct = token[len(word):]
        if trailing_punct:
            punct_type = get_punct_type(trailing_punct)
            token = word

        if token:  # Only add non-empty words
            pairs.append((token, punct_type))

        i += 1

    return pairs


def test_tokenizer():
    """Test the tokenizer."""
    tests = [
        "هذا اختبار ، وهذا آخر .",
        "هل فهمت ؟ نعم !",
        "الجملة الأولى ؛ الجملة الثانية",
        "كلمة، أخرى",  # attached punct
    ]
    print("Test tokenization:")
    for test in tests:
        print(f"  Input: {test}")
        pairs = extract_word_punct_pairs(test)
        punct_names = ['،', '.', '؟', '!', '؛', ':', 'NONE']
        for word, punct in pairs:
            print(f"    '{word}' → '{punct_names[punct]}'")
        print()


def build_vocab(data, min_freq=3):
    """Build vocabulary."""
    word_counts = Counter()
    for item in data:
        tgt = item.get('target', item.get('tgt', ''))
        pairs = extract_word_punct_pairs(tgt)
        for word, _ in pairs:
            word_counts[word] += 1

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


class PunctDataset(Dataset):
    def __init__(self, data, vocab, context_size=2):
        self.samples = []
        punct_counts = Counter()

        for item in data:
            tgt = item.get('target', item.get('tgt', ''))
            pairs = extract_word_punct_pairs(tgt)

            if len(pairs) < 2:
                continue

            words = [p[0] for p in pairs]
            puncts = [p[1] for p in pairs]

            for i in range(len(words)):
                start = max(0, i - context_size)
                context_words = words[start:i+1]

                while len(context_words) < context_size + 1:
                    context_words.insert(0, '<PAD>')

                punct_label = puncts[i]
                punct_counts[punct_label] += 1

                self.samples.append({
                    'context': context_words,
                    'punct': punct_label,
                })

        print(f"  Punct distribution: {dict(punct_counts)}")
        total_punct = sum(c for p, c in punct_counts.items() if p != NONE_IDX)
        print(f"  Total punct: {total_punct}, Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context_ids = [vocab.get(w, 1) for w in sample['context']]
        return {
            'context': torch.tensor(context_ids),
            'punct': torch.tensor(sample['punct']),
        }


class PunctClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, context_size=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        input_dim = embed_dim * (context_size + 1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, NUM_PUNCT + 1)

    def forward(self, context):
        embedded = self.embedding(context)
        flat = embedded.view(embedded.size(0), -1)
        h = torch.relu(self.fc1(flat))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


if __name__ == "__main__":
    print("=" * 70)
    print("QALB PUNCT CLASSIFIER v3")
    print("=" * 70)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    test_tokenizer()

    print("Loading QALB data...")
    with open(QALB_TRAIN, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(QALB_DEV, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}")

    # Analyze patterns
    print("\nAnalyzing punct patterns...")
    word_punct = defaultdict(Counter)
    for item in train_data:
        tgt = item.get('target', item.get('tgt', ''))
        pairs = extract_word_punct_pairs(tgt)
        for word, punct in pairs:
            if punct != NONE_IDX:
                word_punct[word][punct] += 1

    print("\nTop words followed by punct:")
    top_words = sorted(word_punct.items(), key=lambda x: sum(x[1].values()), reverse=True)[:20]
    punct_names = ['،', '.', '؟', '!', '؛', ':']
    for word, counts in top_words:
        total = sum(counts.values())
        dist = ', '.join(f"{punct_names[p]}:{c}" for p, c in counts.most_common(3))
        print(f"  '{word}' ({total}x): {dist}")

    # Build vocab
    print("\nBuilding vocab...")
    vocab = build_vocab(train_data)
    print(f"Vocab size: {len(vocab)}")

    # Create datasets
    print("\nCreating datasets...")
    print("Train:")
    train_ds = PunctDataset(train_data, vocab)
    print("Dev:")
    dev_ds = PunctDataset(dev_data, vocab)

    # Class weights
    train_punct_counts = Counter(s['punct'].item() for s in train_ds)
    total = sum(train_punct_counts.values())
    weights = []
    for i in range(NUM_PUNCT + 1):
        count = train_punct_counts.get(i, 1)
        w = (total / count) ** 0.5  # sqrt to dampen extreme weights
        weights.append(w)
    weights = torch.tensor(weights)
    weights = weights / weights.mean()

    print(f"\nClass weights: {[f'{w:.2f}' for w in weights.tolist()]}")

    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on {device}...")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=256)

    model = PunctClassifier(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_f1 = 0

    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch in train_loader:
            context = batch['context'].to(device)
            punct = batch['punct'].to(device)

            logits = model(context)
            loss = criterion(logits, punct)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Eval
        model.eval()
        tp, fp, fn = Counter(), Counter(), Counter()

        with torch.no_grad():
            for batch in dev_loader:
                context = batch['context'].to(device)
                punct = batch['punct'].to(device)
                preds = model(context).argmax(dim=-1)

                for p, g in zip(preds.cpu().tolist(), punct.cpu().tolist()):
                    if p == g and p != NONE_IDX:
                        tp[p] += 1
                    elif p != NONE_IDX and p != g:
                        fp[p] += 1
                    if g != NONE_IDX and p != g:
                        fn[g] += 1

        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} P={precision:.3f} R={recall:.3f} F1={f1:.3f} (TP={total_tp} FP={total_fp} FN={total_fn})")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
            with open(f"{OUTPUT_DIR}/vocab.pkl", 'wb') as f:
                pickle.dump(vocab, f)

    print(f"\nBest F1: {best_f1:.3f}")
    print(f"Model saved to {OUTPUT_DIR}/best_model.pt")
