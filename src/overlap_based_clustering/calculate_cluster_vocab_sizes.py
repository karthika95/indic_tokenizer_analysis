from collections import defaultdict
import numpy as np
import os
import re
import sys
import json
from pathlib import Path
import shutil
import subprocess

def load_vocab(lang, root_dir):
    vocab = []
    vocab_file = os.path.join(root_dir, f"{lang}.vocab")
    if not os.path.exists(vocab_file):
        print(f"Warning: Vocab file not found: {vocab_file}")
        return []
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            subword = line.split('\t')[0]
            vocab.append(subword)
    return vocab

def calculate_cluster_vocab_sizes(cluster_def_path, vocab_dir, total_vocab_size):
    """Calculate optimal vocab sizes for clusters based on union vocabulary analysis"""
    cluster_re = re.compile(r"Cluster\s+(\d+)\s*:\s*(.+)", re.I)
    
    # Parse the cluster-definition file
    clusters = {}            # {cluster_id: [lang1, lang2, ...]}
    with open(cluster_def_path, 'r', encoding="utf-8") as f:
        for line in f:
            m = cluster_re.match(line.strip())
            if not m:
                continue                      # skip empty / malformed lines
            cid, langs = m.groups()
            lang_list = [l.strip() for l in langs.split(",") if l.strip()]
            clusters[int(cid)] = lang_list

    vocab_sizes = []
    for cluster_id in sorted(clusters.keys()):
        langs = clusters[cluster_id]
        union_vocab_set = set()
        for lang in langs:
            vocab = load_vocab(lang, vocab_dir)
            union_vocab_set.update(vocab)
        vocab_size = len(union_vocab_set)
        vocab_sizes.append(vocab_size)
        print(f"Cluster {cluster_id}: {len(langs)} languages, union vocab size: {vocab_size}")

    total_clustered_vocab_size = sum(vocab_sizes)
    if total_clustered_vocab_size == 0:
        print("Error: Total vocabulary size is 0")
        return None, None
    
    factor = total_vocab_size / total_clustered_vocab_size
    individual_vocabs = [max(1000, round(vocab_size * factor)) for vocab_size in vocab_sizes]
    
    print(f"\nOriginal cluster vocab sizes: {vocab_sizes}")
    print(f"Scaling factor: {factor:.3f}")
    print(f"Adjusted vocab sizes: {individual_vocabs}")
    print(f"Total adjusted vocab: {sum(individual_vocabs)}")
    
    return clusters, individual_vocabs

def train_cluster_tokenizers(cluster_corpus_dir, vocab_sizes, output_dir=None, threads=128, model_type="unigram"):
    """Train tokenizers on clustered corpora with specified vocab sizes"""
    # Use separate output directory if provided, otherwise use corpus directory
    if output_dir is None:
        output_dir = cluster_corpus_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all cluster corpus files
    cluster_files = []
    for file in os.listdir(cluster_corpus_dir):
        if file.startswith("cluster_") and file.endswith(".txt"):
            cluster_files.append(file)
    
    cluster_files.sort()  # Ensure consistent ordering
    
    if len(vocab_sizes) != len(cluster_files):
        print(f"Error: Mismatch between vocab sizes ({len(vocab_sizes)}) and cluster files ({len(cluster_files)})")
        return False
    
    # Train tokenizer for each cluster
    for cluster_file, vocab_size in zip(cluster_files, vocab_sizes):
        file_path = os.path.join(cluster_corpus_dir, cluster_file)
        output_name = cluster_file.replace('.txt', '')
        output_path = os.path.join(output_dir, output_name)  # Save to output_dir, not corpus_dir
        
        print(f"Training {cluster_file} with vocab size {vocab_size}")
        print(f"Saving model to: {output_path}")
        
        try:
            subprocess.run([
                "python", "train_indic_sentpiece_v1.py",
                "-m", model_type,
                "-v", str(vocab_size),
                "-d", file_path,
                "-t", str(threads),
                "-o", output_path
            ], check=True)
            
            print(f"Finished training: {output_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error training {cluster_file}: {e}")
            return False
        
        print("-----------------------------")
    
    return True

if __name__ == "__main__":
    # Default configuration for standalone usage
    if len(sys.argv) > 1:
        cluster_def_path = sys.argv[1]
        vocab_dir = sys.argv[2] if len(sys.argv) > 2 else 'monolingual_tokenizers_and_clusters/vocab_size_32'
        total_vocab_size = int(sys.argv[3]) if len(sys.argv) > 3 else 256000
        cluster_corpus_dir = sys.argv[4] if len(sys.argv) > 4 else None
    else:
        # Original hardcoded values for backward compatibility
        cluster_def_path = "monolingual_tokenizers_and_clusters/vocab_size_32/language_clusters_l2_5.txt"
        vocab_dir = 'monolingual_tokenizers_and_clusters/vocab_size_32'
        total_vocab_size = 256000
        cluster_corpus_dir = None

    clusters, individual_vocabs = calculate_cluster_vocab_sizes(cluster_def_path, vocab_dir, total_vocab_size)
    
    if clusters and individual_vocabs and cluster_corpus_dir:
        print(f"\nTraining cluster tokenizers...")
        success = train_cluster_tokenizers(cluster_corpus_dir, individual_vocabs)
        if success:
            print("✅ Cluster tokenizer training completed!")
        else:
            print("❌ Cluster tokenizer training failed!")
    else:
        print("Vocab size calculation completed. Use the individual_vocabs for training.")
