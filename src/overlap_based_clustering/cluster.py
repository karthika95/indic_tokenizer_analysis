#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np  
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

def load_vocab(lang, root_dir, vocab_prefix, vocab_ext):
    vocab = []
    vocab_file = os.path.join(root_dir, f"{vocab_prefix}{lang}{vocab_ext}")
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            subword = line.split('\t')[0]
            vocab.append(subword)
    return vocab

def main(args):
    # Regex for matching vocab files
    pattern = re.compile(fr'^{re.escape(args.vocab_prefix)}(.+){re.escape(args.vocab_ext)}$')

    # Step 1: Gather vocabularies
    all_vocabs = {}
    union_vocab_set = set()
    language_list = []

    for dirpath, _, filenames in os.walk(args.root_dir):
        for filename in filenames:
            match = pattern.match(filename)
            if match:
                lang = match.group(1)
                print(f"Processing {lang}")
                language_list.append(lang)

    for lang in language_list:
        vocab = load_vocab(lang, args.root_dir, args.vocab_prefix, args.vocab_ext)
        all_vocabs[lang] = set(vocab)
        union_vocab_set.update(vocab)

    union_vocab = sorted(list(union_vocab_set))
    subword2idx = {sw: i for i, sw in enumerate(union_vocab)}
    print(f"Union vocab size: {len(union_vocab)}")
    print(f"Number of languages: {len(language_list)}")

    # Step 2: Create binary vectors
    lang_vectors = {}
    for lang, vocab in all_vocabs.items():
        vec = np.zeros(len(union_vocab), dtype=np.int8)
        for sw in vocab:
            vec[subword2idx[sw]] = 1
        lang_vectors[lang] = vec

    X = np.stack([lang_vectors[lang] for lang in language_list])

    # Step 3: Run clustering
    for k in range(args.min_k, args.max_k + 1):
        # KMedoids (cosine)
        kmedoids = KMedoids(n_clusters=k, metric=args.kmedoids_metric, random_state=args.random_state)
        clusters = kmedoids.fit_predict(X)
        out_path = os.path.join(args.output_dir, f"language_clusters_{args.kmedoids_metric}_{k}.txt")
        write_clusters(out_path, clusters, language_list)

        # KMeans (L2)
        kmeans = KMeans(n_clusters=k, random_state=args.random_state)
        clusters = kmeans.fit_predict(X)
        out_path = os.path.join(args.output_dir, f"language_clusters_l2_{k}.txt")
        write_clusters(out_path, clusters, language_list)

def write_clusters(filename, clusters, language_list):
    with open(filename, "w") as f:
        for i in range(max(clusters) + 1):
            f.write(f"Cluster {i+1}: ")
            langs_in_cluster = [lang for idx, lang in enumerate(language_list) if clusters[idx] == i]
            f.write(", ".join(langs_in_cluster))
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster languages by vocabulary overlap")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing vocab files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save cluster files")
    parser.add_argument("--vocab_prefix", type=str, default="", help="Prefix of vocab filenames")
    parser.add_argument("--vocab_ext", type=str, default=".vocab", help="Extension of vocab filenames")
    parser.add_argument("--min_k", type=int, default=2, help="Minimum number of clusters")
    parser.add_argument("--max_k", type=int, default=9, help="Maximum number of clusters")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for clustering")
    parser.add_argument("--kmedoids_metric", type=str, default="cosine", help="Distance metric for KMedoids")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)