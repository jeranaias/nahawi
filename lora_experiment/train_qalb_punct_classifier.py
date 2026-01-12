#!/usr/bin/env python3
"""
Train punct classifier on QALB only.
QALB is forum text with its own punct conventions - different from MSA corpus.

Approach:
1. Extract (word, next_punct) pairs from QALB gold
2. Train classifier: given word + context → predict punct (or none)
3. Apply to LoRA output to fix punct positions
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

PUNCT_CHARS = ['،', '.', '؟', '!', '؛', ':', 'NONE']
PUNCT_TO_IDX = {p: i for i, p in enumerate(PUNCT_CHARS)}
IDX_TO_PUNCT = {i: p for p, i in PUNCT_TO_IDX.items()}

def tokenize_with_punct(text):
    """Split text into tokens, tracking punct after each."""
    # Split on whitespace but keep punct attached
    tokens = []
    current = ""

    for char in text:
        if char.isspace():
            if current:
                tokens.append(current)
                current = ""
        else:
            current += char
    if current:
        tokens.append(current)

    # Now extract (word, following_punct) pairs
    pairs = []
    for token in tokens:
        # Strip trailing punct
        word = token.rstrip('،.؟!؛:,;?')
        punct = token[len(word):] if len(word) < len(token) else ""

        # Normalize punct
        if punct:
            # Take first punct char
            p = punct[0]
            if p in ['،', ',']:
                p = '،'
            elif p in ['.']:
                p = '.'
            elif p in ['؟', '?']:
                p = '؟'
            elif p in ['!']:
                p = '!'
            elif p in ['؛', ';']:
                p = '؛'
            elif p in [':']:
                p = ':'
            else:
                p = 'NONE'
        else:
            p = 'NONE'

        if word:  # Only add if there's a word
            pairs.append((word, p))

    return pairs


def extract_patterns(data):
    """Extract punct patterns from QALB data."""
    # Count (word, punct) occurrences
    word_punct_counts = defaultdict(Counter)

    # Count (prev_word, word, punct) for bigram context
    bigram_punct_counts = defaultdict(Counter)

    # Count punct after specific phrases
    phrase_punct_counts = defaultdict(Counter)

    for item in data:
        tgt = item.get('target', item.get('tgt', ''))
        pairs = tokenize_with_punct(tgt)

        prev_word = "<START>"
        for i, (word, punct) in enumerate(pairs):
            # Unigram
            word_punct_counts[word][punct] += 1

            # Bigram
            bigram_punct_counts[(prev_word, word)][punct] += 1

            # Common phrases (2-3 words before punct)
            if punct != 'NONE' and i >= 1:
                phrase = f"{prev_word} {word}"
                phrase_punct_counts[phrase][punct] += 1

            prev_word = word

    return word_punct_counts, bigram_punct_counts, phrase_punct_counts


def build_vocab(data, min_freq=5):
    """Build vocabulary from QALB data."""
    word_counts = Counter()
    for item in data:
        tgt = item.get('target', item.get('tgt', ''))
        pairs = tokenize_with_punct(tgt)
        for word, _ in pairs:
            word_counts[word] += 1

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab


class PunctDataset(Dataset):
    def __init__(self, data, vocab, context_size=3):
        self.samples = []
        self.vocab = vocab
        self.context_size = context_size

        for item in data:
            tgt = item.get('target', item.get('tgt', ''))
            pairs = tokenize_with_punct(tgt)

            # Create samples with context
            words = [p[0] for p in pairs]
            puncts = [p[1] for p in pairs]

            for i in range(len(words)):
                # Get context window
                start = max(0, i - context_size)
                end = min(len(words), i + context_size + 1)

                context = words[start:end]
                center_idx = i - start

                # Pad if needed
                while len(context) < 2 * context_size + 1:
                    if len(context) <= center_idx:
                        context.append('<PAD>')
                    else:
                        context.insert(0, '<PAD>')
                        center_idx += 1

                punct_label = PUNCT_TO_IDX.get(puncts[i], PUNCT_TO_IDX['NONE'])

                self.samples.append({
                    'context': context,
                    'center_idx': context_size,  # Always middle after padding
                    'punct': punct_label,
                    'word': words[i]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert words to indices
        context_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in sample['context']]

        return {
            'context': torch.tensor(context_ids),
            'punct': torch.tensor(sample['punct']),
        }


class PunctClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, context_size=3, num_classes=7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.context_size = context_size

        # Process context with attention to center word
        self.context_proj = nn.Linear(embed_dim * (2 * context_size + 1), hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, context):
        # context: (batch, 2*context_size+1)
        embedded = self.embedding(context)  # (batch, seq, embed)

        # Flatten context
        batch_size = embedded.size(0)
        flat = embedded.view(batch_size, -1)  # (batch, seq*embed)

        hidden = torch.relu(self.context_proj(flat))
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.fc1(hidden))
        logits = self.fc2(hidden)

        return logits


def train_classifier(train_data, dev_data, vocab, epochs=10, batch_size=64):
    """Train the punct classifier."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create datasets
    train_dataset = PunctDataset(train_data, vocab)
    dev_dataset = PunctDataset(dev_data, vocab)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")

    # Class weights (punct is rare)
    class_counts = Counter(s['punct'].item() for s in train_dataset)
    total = sum(class_counts.values())
    weights = torch.tensor([total / (class_counts[i] + 1) for i in range(len(PUNCT_CHARS))])
    weights = weights / weights.sum() * len(PUNCT_CHARS)
    weights = weights.to(device)

    print(f"Class distribution: {dict(class_counts)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    model = PunctClassifier(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_acc = 0

    for epoch in range(epochs):
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

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        punct_correct = 0
        punct_total = 0

        with torch.no_grad():
            for batch in dev_loader:
                context = batch['context'].to(device)
                punct = batch['punct'].to(device)

                logits = model(context)
                preds = logits.argmax(dim=-1)

                correct += (preds == punct).sum().item()
                total += punct.size(0)

                # Punct-only accuracy (excluding NONE)
                mask = punct != PUNCT_TO_IDX['NONE']
                punct_correct += ((preds == punct) & mask).sum().item()
                punct_total += mask.sum().item()

        acc = 100 * correct / total
        punct_acc = 100 * punct_correct / punct_total if punct_total > 0 else 0

        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f} acc={acc:.1f}% punct_acc={punct_acc:.1f}%")

        if punct_acc > best_acc:
            best_acc = punct_acc
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")

    return model


def main():
    print("=" * 70)
    print("QALB-ONLY PUNCT CLASSIFIER")
    print("=" * 70)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading QALB data...")
    with open(QALB_TRAIN, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(QALB_DEV, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    print(f"Train: {len(train_data)} examples")
    print(f"Dev: {len(dev_data)} examples")

    # Extract patterns
    print("\nExtracting punct patterns...")
    word_punct, bigram_punct, phrase_punct = extract_patterns(train_data)

    # Show top patterns
    print("\nTop patterns before punct:")
    phrase_counts = [(phrase, counts.most_common(1)[0]) for phrase, counts in phrase_punct.items()
                     if counts.most_common(1)[0][1] >= 50]
    phrase_counts.sort(key=lambda x: x[1][1], reverse=True)

    for phrase, (punct, count) in phrase_counts[:20]:
        print(f"  '{phrase}' → '{punct}' ({count}x)")

    # Build vocab
    print("\nBuilding vocabulary...")
    vocab = build_vocab(train_data)
    print(f"Vocab size: {len(vocab)}")

    # Save vocab
    with open(f"{OUTPUT_DIR}/vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)

    # Train classifier
    print("\nTraining classifier...")
    model = train_classifier(train_data, dev_data, vocab, epochs=10)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {OUTPUT_DIR}/best_model.pt")
    print(f"Vocab saved to: {OUTPUT_DIR}/vocab.pkl")


if __name__ == "__main__":
    main()
