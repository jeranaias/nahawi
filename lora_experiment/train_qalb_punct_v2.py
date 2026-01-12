#!/usr/bin/env python3
"""
Train punct classifier on QALB - FIXED tokenization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import pickle

QALB_TRAIN = "/home/ubuntu/nahawi/data/qalb_real_train.json"
QALB_DEV = "/home/ubuntu/nahawi/data/qalb_real_dev.json"
OUTPUT_DIR = "/home/ubuntu/nahawi/punct_classifier_qalb"

# All punct we care about
PUNCT_CHARS = set('،.؟!؛:,;?')
PUNCT_MAP = {
    '،': 0, ',': 0,  # comma
    '.': 1,           # period
    '؟': 2, '?': 2,  # question
    '!': 3,           # exclamation
    '؛': 4, ';': 4,  # semicolon
    ':': 5,           # colon
}
NUM_PUNCT = 6  # 0-5 for punct types
NONE_IDX = NUM_PUNCT  # 6 = no punct


def extract_word_punct_pairs(text):
    """Extract (word, following_punct_type) pairs from text."""
    pairs = []

    # Split into tokens (whitespace separated)
    tokens = text.split()

    for token in tokens:
        # Find where punct starts (if any)
        word_end = len(token)
        punct_type = NONE_IDX

        for i, char in enumerate(token):
            if char in PUNCT_CHARS:
                word_end = i
                punct_type = PUNCT_MAP.get(char, NONE_IDX)
                break

        word = token[:word_end]

        # Skip empty words
        if word and not all(c in PUNCT_CHARS for c in word):
            pairs.append((word, punct_type))

    return pairs


def test_tokenizer():
    """Test the tokenizer on sample sentences."""
    test = "هذا اختبار ، وهذا آخر . هل فهمت ؟"
    pairs = extract_word_punct_pairs(test)
    print("Test tokenization:")
    for word, punct in pairs:
        punct_char = ['،', '.', '؟', '!', '؛', ':', 'NONE'][punct]
        print(f"  '{word}' → '{punct_char}'")
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
        self.vocab = vocab
        self.context_size = context_size

        punct_counts = Counter()

        for item in data:
            tgt = item.get('target', item.get('tgt', ''))
            pairs = extract_word_punct_pairs(tgt)

            if len(pairs) < 2:
                continue

            words = [p[0] for p in pairs]
            puncts = [p[1] for p in pairs]

            for i in range(len(words)):
                # Get context: prev words + current word
                start = max(0, i - context_size)
                context_words = words[start:i+1]

                # Pad left if needed
                while len(context_words) < context_size + 1:
                    context_words.insert(0, '<PAD>')

                punct_label = puncts[i]
                punct_counts[punct_label] += 1

                self.samples.append({
                    'context': context_words,
                    'punct': punct_label,
                })

        print(f"  Punct distribution: {dict(punct_counts)}")
        print(f"  Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context_ids = [self.vocab.get(w, 1) for w in sample['context']]
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
        self.fc3 = nn.Linear(hidden_dim // 2, NUM_PUNCT + 1)  # +1 for NONE

    def forward(self, context):
        embedded = self.embedding(context)  # (batch, seq, embed)
        flat = embedded.view(embedded.size(0), -1)
        h = torch.relu(self.fc1(flat))
        h = self.dropout(h)
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


def main():
    print("=" * 70)
    print("QALB PUNCT CLASSIFIER v2")
    print("=" * 70)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Test tokenizer first
    test_tokenizer()

    # Load data
    print("Loading QALB data...")
    with open(QALB_TRAIN, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(QALB_DEV, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}")

    # Show punct patterns
    print("\nAnalyzing punct patterns in gold...")
    phrase_punct = defaultdict(Counter)
    for item in train_data[:5000]:  # Sample
        tgt = item.get('target', item.get('tgt', ''))
        pairs = extract_word_punct_pairs(tgt)
        for i in range(1, len(pairs)):
            prev_word = pairs[i-1][0]
            curr_word = pairs[i][0]
            punct = pairs[i-1][1]
            if punct != NONE_IDX:
                phrase_punct[prev_word][punct] += 1

    print("\nTop words before punct:")
    top_words = sorted(phrase_punct.items(), key=lambda x: sum(x[1].values()), reverse=True)[:15]
    for word, counts in top_words:
        total = sum(counts.values())
        punct_names = ['،', '.', '؟', '!', '؛', ':']
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

    # Compute class weights
    train_punct_counts = Counter(s['punct'].item() for s in train_ds)
    total = sum(train_punct_counts.values())
    weights = []
    for i in range(NUM_PUNCT + 1):
        count = train_punct_counts.get(i, 1)
        weights.append(total / count)
    weights = torch.tensor(weights)
    weights = weights / weights.sum() * (NUM_PUNCT + 1)

    print(f"\nClass weights: {weights.tolist()}")

    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on {device}...")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=256)

    model = PunctClassifier(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_f1 = 0

    for epoch in range(15):
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
        tp = Counter()
        fp = Counter()
        fn = Counter()

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

        # Compute F1 per class
        punct_names = ['،', '.', '؟', '!', '؛', ':']
        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f} P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
            with open(f"{OUTPUT_DIR}/vocab.pkl", 'wb') as f:
                pickle.dump(vocab, f)

    print(f"\nBest F1: {best_f1:.3f}")
    print(f"Model saved to {OUTPUT_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
