#!/usr/bin/env python3
"""
word2vec skip-gram with negative sampling — full pipeline
"""

import sys
import os
import json
import time
import struct
import ctypes
from collections import Counter
from pathlib import Path

import numpy as np

# ── Constants ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CORPUS_PATH = DATA_DIR / "text8"
QUESTIONS_PATH = DATA_DIR / "questions-words.txt"
LIB_PATH = Path(__file__).parent / "libtrain.so"

MIN_COUNT = 5
VOCAB_SIZE = None  # set after building vocab


# ── C library binding ────────────────────────────────────────────────
def load_lib():
    lib = ctypes.CDLL(str(LIB_PATH))

    lib.train_skipgram.argtypes = [
        ctypes.c_void_p,                   # corpus (int*)
        ctypes.c_longlong,                 # corpus_len
        ctypes.c_void_p,                   # counts (int*)
        ctypes.c_int,                      # vocab_size
        ctypes.c_void_p,                   # w_in (float*)
        ctypes.c_void_p,                   # w_out (float*)
        ctypes.c_int,                      # dim
        ctypes.c_longlong,                 # total_words
        ctypes.c_int,                      # epochs
        ctypes.c_int,                      # window
        ctypes.c_int,                      # neg_samples
        ctypes.c_double,                   # subsample_t
        ctypes.c_float,                    # lr_start
        ctypes.c_int,                      # num_threads
        ctypes.c_uint64,                   # seed
    ]
    lib.train_skipgram.restype = ctypes.c_int

    return lib


# ── Vocabulary ───────────────────────────────────────────────────────
def build_vocab(corpus_path, min_count=5):
    """Read text8, return (word_to_id, id_to_word, counts, corpus_ids, total_tokens_raw)."""
    print(f"Reading corpus: {corpus_path}")
    text = open(corpus_path).read()
    words = text.split()
    print(f"Total tokens: {len(words)}")

    counter = Counter(words)
    total_unique = len(counter)
    print(f"Unique tokens: {total_unique}")

    # Filter by min_count
    vocab_words = sorted(
        [w for w, c in counter.items() if c >= min_count],
        key=lambda w: counter[w], reverse=True
    )
    print(f"Vocabulary size (min_count={min_count}): {len(vocab_words)}")

    word_to_id = {w: i for i, w in enumerate(vocab_words)}
    id_to_word = vocab_words
    counts = [counter[w] for w in vocab_words]
    total_tokens_raw = sum(counter.values())

    # Build corpus as list of word IDs (skip OOV)
    corpus_ids = np.array(
        [word_to_id[w] for w in words if w in word_to_id],
        dtype=np.int32
    )
    print(f"Corpus IDs: {len(corpus_ids)} (skipped {len(words) - len(corpus_ids)} OOV)")

    return word_to_id, id_to_word, counts, corpus_ids, total_tokens_raw


# ── Initialization ───────────────────────────────────────────────────
def init_vectors(vocab_size, dim, seed):
    """Initialize input vectors uniformly in [-0.5/dim, 0.5/dim], output vectors to zero."""
    rng = np.random.RandomState(seed)
    scale = 0.5 / dim
    w_in = rng.uniform(-scale, scale, (vocab_size, dim)).astype(np.float32)
    w_out = np.zeros((vocab_size, dim), dtype=np.float32)
    return w_in, w_out


# ── Training ─────────────────────────────────────────────────────────
def train(lib, corpus_ids, counts, w_in, w_out, dim, window, neg_samples,
          subsample_t, lr_start, epochs, seed, num_threads):
    """Call C training function."""
    total_words = sum(counts)
    print(f"Starting training: dim={dim}, window={window}, neg={neg_samples}, "
          f"subsample_t={subsample_t}, lr={lr_start}, epochs={epochs}, "
          f"threads={num_threads}, seed={seed}")

    t0 = time.time()
    ret = lib.train_skipgram(
        corpus_ids.ctypes.data_as(ctypes.c_void_p),
        len(corpus_ids),
        (ctypes.c_int * len(counts))(*counts),
        len(counts),
        w_in.ctypes.data_as(ctypes.c_void_p),
        w_out.ctypes.data_as(ctypes.c_void_p),
        dim,
        total_words,
        epochs,
        window,
        neg_samples,
        subsample_t,
        lr_start,
        num_threads,
        seed,
    )
    elapsed = time.time() - t0

    if ret != 0:
        raise RuntimeError(f"Training failed with return code {ret}")

    print(f"Training finished in {elapsed:.1f}s")
    return elapsed


# ── Save vectors ─────────────────────────────────────────────────────
def save_vectors(path, id_to_word, w_in, w_out):
    """Save word vectors in word2vec text format (input + output averaged)."""
    vectors = w_in + w_out  # common practice: sum input and output vectors
    n, d = vectors.shape
    with open(path, "w") as f:
        f.write(f"{n} {d}\n")
        for i, word in enumerate(id_to_word):
            vec_str = " ".join(f"{v:.6f}" for v in vectors[i])
            f.write(f"{word} {vec_str}\n")
    print(f"Vectors saved to {path}")


# ── Evaluation ───────────────────────────────────────────────────────
def load_questions(path):
    """Parse questions-words.txt. Returns list of (category, list_of_quadruples)."""
    categories = []
    current_category = None
    current_quads = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                if current_category is not None and current_quads:
                    categories.append((current_category, current_quads))
                current_category = line[1:].strip()
                current_quads = []
            else:
                parts = line.split()
                if len(parts) == 4:
                    current_quads.append(tuple(p.lower() for p in parts))

    if current_category is not None and current_quads:
        categories.append((current_category, current_quads))

    return categories


def evaluate_analogies(vectors, word_to_id, id_to_word, questions_path):
    """Evaluate word analogy accuracy. Returns semantic, syntactic, total accuracies."""
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v_norm = vectors / norms

    categories = load_questions(questions_path)

    semantic_cats = {
        "capital-common-countries", "capital-world", "currency",
        "city-in-state", "family"
    }

    total_correct = 0
    total_questions = 0
    total_skipped = 0
    sem_correct = 0
    sem_questions = 0
    syn_correct = 0
    syn_questions = 0

    for cat_name, quads in categories:
        cat_correct = 0
        cat_total = 0
        cat_skipped = 0

        for a, b, c, d in quads:
            # Check vocabulary
            if a not in word_to_id or b not in word_to_id or c not in word_to_id or d not in word_to_id:
                cat_skipped += 1
                total_skipped += 1
                continue

            id_a = word_to_id[a]
            id_b = word_to_id[b]
            id_c = word_to_id[c]
            id_d = word_to_id[d]

            # vec(b) - vec(a) + vec(c)
            query = v_norm[id_b] - v_norm[id_a] + v_norm[id_c]

            # Cosine similarity with all words
            sims = np.dot(v_norm, query)

            # Exclude a, b, c
            sims[id_a] = -np.inf
            sims[id_b] = -np.inf
            sims[id_c] = -np.inf

            predicted_id = np.argmax(sims)

            cat_total += 1
            total_questions += 1
            if predicted_id == id_d:
                cat_correct += 1
                total_correct += 1

        cat_acc = cat_correct / cat_total if cat_total > 0 else 0.0
        is_sem = any(s in cat_name.lower() for s in [
            "capital", "currency", "city", "family"
        ])
        if is_sem:
            sem_correct += cat_correct
            sem_questions += cat_total
        else:
            syn_correct += cat_correct
            syn_questions += cat_total

        print(f"  {cat_name}: {cat_correct}/{cat_total} = {cat_acc:.3f}"
              f" (skipped {cat_skipped})")

    total_acc = total_correct / total_questions if total_questions > 0 else 0.0
    sem_acc = sem_correct / sem_questions if sem_questions > 0 else 0.0
    syn_acc = syn_correct / syn_questions if syn_questions > 0 else 0.0

    print(f"\nTotal: {total_correct}/{total_questions} = {total_acc:.4f}")
    print(f"Semantic: {sem_correct}/{sem_questions} = {sem_acc:.4f}")
    print(f"Syntactic: {syn_correct}/{syn_questions} = {syn_acc:.4f}")
    print(f"Skipped: {total_skipped}")

    return {
        "total_accuracy": round(total_acc, 6),
        "semantic_accuracy": round(sem_acc, 6),
        "syntactic_accuracy": round(syn_acc, 6),
        "total_questions": total_questions,
        "semantic_questions": sem_questions,
        "syntactic_questions": syn_questions,
        "skipped": total_skipped,
    }


# ── Main ─────────────────────────────────────────────────────────────
def main():
    # Hyperparameters
    DIM = 150
    WINDOW = 5
    NEG_SAMPLES = 15
    SUBSAMPLE_T = 1e-5
    LR_START = 0.025
    EPOCHS = 20
    SEED = 42
    NUM_THREADS = 4

    print("=" * 60)
    print("word2vec skip-gram with negative sampling")
    print(f"dim={DIM} window={WINDOW} neg={NEG_SAMPLES} subsample_t={SUBSAMPLE_T}")
    print(f"lr={LR_START} epochs={EPOCHS} seed={SEED} threads={NUM_THREADS}")
    print("=" * 60)

    # Build vocabulary
    word_to_id, id_to_word, counts, corpus_ids, total_tokens = build_vocab(
        CORPUS_PATH, MIN_COUNT
    )

    # Initialize vectors
    w_in, w_out = init_vectors(len(id_to_word), DIM, SEED)

    # Load C library
    lib = load_lib()

    # Train
    train_time = train(
        lib, corpus_ids, counts, w_in, w_out,
        DIM, WINDOW, NEG_SAMPLES, SUBSAMPLE_T, LR_START, EPOCHS, SEED, NUM_THREADS
    )

    # Save vectors (input vectors only — standard word2vec)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    vectors_path = ARTIFACTS_DIR / "vectors.txt"
    
    with open(vectors_path, "w") as f:
        n, d = w_in.shape
        f.write(f"{n} {d}\n")
        for i, word in enumerate(id_to_word):
            vec_str = " ".join(f"{v:.6f}" for v in w_in[i])
            f.write(f"{word} {vec_str}\n")
    print(f"Vectors saved to {vectors_path}")

    # Evaluate (use input vectors only — standard word2vec practice)
    print("\nEvaluating analogies...")
    results = evaluate_analogies(w_in, word_to_id, id_to_word, QUESTIONS_PATH)

    # Add hyperparameters to results
    results["hyperparameters"] = {
        "dimension": DIM,
        "window": WINDOW,
        "negative_samples": NEG_SAMPLES,
        "epochs": EPOCHS,
        "subsampling_threshold": SUBSAMPLE_T,
        "learning_rate": LR_START,
        "seed": SEED,
        "wall_clock_training_time_seconds": round(train_time, 1),
    }

    # Save results
    results_path = ARTIFACTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()