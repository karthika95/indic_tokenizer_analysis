import math
import sys
import os
from sentencepiece import sentencepiece_model_pb2 as sp_model
from collections import defaultdict, Counter
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm

def normalize_tokenizer(merged_model_path, cluster_count, corpus_dir, model_dir, output_path):
    """Normalize final tokenizer based on corpus statistics"""
    # === Load merged model and skip special tokens ===
    merged_model = sp_model.ModelProto()
    merged_model.ParseFromString(open(merged_model_path, "rb").read())

    special_token_count = 3
    special_tokens = merged_model.pieces[:special_token_count]
    merged_pieces = merged_model.pieces[special_token_count:]

    print(f"Loaded merged model with {len(merged_model.pieces)} total pieces")
    print(f"Special tokens: {special_token_count}, Regular pieces: {len(merged_pieces)}")

    # === Step 1: Compute token frequencies per cluster ===
    token_freqs = defaultdict(list)  # token → list of probabilities from clusters

    for i in range(1, cluster_count + 1):
        print(f"\n🔄 Processing cluster {i}...")

        model_path = Path(model_dir) / f"cluster_{i}.model"
        corpus_path = Path(corpus_dir) / f"cluster_{i}.txt"
        
        if not model_path.exists():
            print(f"Warning: Model not found: {model_path}")
            continue
            
        if not corpus_path.exists():
            print(f"Warning: Corpus not found: {corpus_path}")
            continue
            
        sp = spm.SentencePieceProcessor(model_file=str(model_path))

        counter = Counter()
        total = 0

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Tokenizing cluster {i}", unit=" lines"):
                line = line.strip()
                if line:
                    tokens = sp.encode(line, out_type=str)
                    counter.update(tokens)
                    total += len(tokens)

        for token, freq in counter.items():
            if total > 0:
                prob = freq / total
                token_freqs[token].append(prob)

    print(f"\nFound frequencies for {len(token_freqs)} unique tokens")

    # === Step 2: Reassign new scores based on average probability ===
    new_pieces = []
    tokens_with_freq = 0
    tokens_without_freq = 0

    for p in merged_pieces:
        token = p.piece
        if token in token_freqs and len(token_freqs[token]) > 0:
            avg_prob = sum(token_freqs[token]) / len(token_freqs[token])
            p.score = math.log(max(avg_prob, 1e-10))  # Avoid log(0)
            tokens_with_freq += 1
        else:
            p.score = math.log(1e-10)  # tiny fallback for unseen tokens
            tokens_without_freq += 1
        new_pieces.append(p)

    print(f"Tokens with frequency data: {tokens_with_freq}")
    print(f"Tokens without frequency data: {tokens_without_freq}")

    # === Step 3: Renormalize to make it a valid unigram LM
    unnorm_total = sum(math.exp(p.score) for p in new_pieces)
    if unnorm_total > 0:
        for p in new_pieces:
            prob = math.exp(p.score) / unnorm_total
            p.score = math.log(max(prob, 1e-10))  # Avoid log(0)

        print(f"✅ Renormalization complete. Sum(exp(score)) ≈ {sum(math.exp(p.score) for p in new_pieces):.6f}")
    else:
        print("Warning: Unnormalized total is 0, keeping original scores")

    # === Step 4: Write out new model ===
    new_model = sp_model.ModelProto()
    new_model.CopyFrom(merged_model)

    # Replace pieces safely
    del new_model.pieces[:]                    # Clear all tokens
    new_model.pieces.extend(special_tokens)    # Add special tokens
    new_model.pieces.extend(new_pieces)        # Add updated scored tokens

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(new_model.SerializeToString())

    print(f"✅ Final normalized model saved: {output_path}")
    print(f"Total pieces in final model: {len(new_model.pieces)}")

if __name__ == "__main__":
    # Allow command line usage
    if len(sys.argv) >= 6:
        merged_model_path = sys.argv[1]
        cluster_count = int(sys.argv[2])
        corpus_dir = sys.argv[3]
        model_dir = sys.argv[4]
        output_path = sys.argv[5]
    else:
        # Original hardcoded configuration for backward compatibility
        merged_model_path = "cluster_tokenizer_and_merged_tokenizer/cluster_5_l2/total_vocab_32k/cluster_merged_32.model"
        cluster_count = 5
        corpus_dir = Path("cluster_corups/monolinugal_vocab_32/cluster_5_l2")   # directory with cluster_1.txt ... cluster_5.txt
        model_dir = Path("cluster_tokenizer_and_merged_tokenizer/cluster_5_l2/total_vocab_32k/")     # directory with cluster_1.model ... cluster_5.model
        output_path = "cluster_tokenizer_and_merged_tokenizer/cluster_5_l2/total_vocab_32k/cluster_merged_normalized.model"

    normalize_tokenizer(merged_model_path, cluster_count, corpus_dir, model_dir, output_path)