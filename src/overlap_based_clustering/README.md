# Indic Language Tokenizer Pipeline

A complete end-to-end pipeline for training clustered tokenizers for Indic languages using SentencePiece.

## Overview

This pipeline trains monolingual tokenizers, clusters languages by vocabulary similarity, creates clustered corpora, and trains optimized cluster tokenizers with automatic vocabulary size adjustment.

## Main file to run
**`end_to_end.py`** - Main end to end pipeline. Change the variables
ROOT_DIR - directory contaning .txt dataset files 
threads, model_type - for tokenizer training
vocab_sizes - list of vocab sizes to train on

### Usage
```bash
python end_to_end.py
```

## Output Directory Structure

```
./
├── monolingual_tokenizers_and_clusters/    # Individual language tokenizers
│   └── vocab_4000/
│       ├── tel.model, tel.vocab
│       ├── hin.model, hin.vocab
│       └── ...
├── clusters/                               # Language clustering results
│   └── vocab_size_4000/
│       ├── language_clusters_cosine_2.txt
│       ├── language_clusters_l2_5.txt
│       └── ...
├── final_tokenizers/                       # Final cluster tokenizers (MAIN OUTPUT)
│   ├── vocab_4000_cosine_4/
│   │   ├── cluster_1.model, cluster_1.vocab
│   │   ├── cluster_2.model, cluster_2.vocab
│   │   ├── merged_tokenizer.model
│   │   ├── final_normalized_tokenizer.model  ⭐ FINAL RESULT
│   │   └── vocab_config.json
│   └── vocab_4000_l2_5/
│       └── ...
└── temp_clustered_corpus_*/                # Temporary corpus files (auto-deleted)
```

## Output

The main output is `final_normalized_tokenizer.model` in each cluster directory, optimized for the specific language grouping and ready for production use.

## Pipeline Flow

1. **Train Individual Tokenizers** - Creates monolingual tokenizers for each language
2. **Cluster Languages** - Groups languages by vocabulary similarity (cosine/L2 metrics)
3. **Process Clusters** - For each cluster configuration:
   - Create clustered corpus (temporary)
   - Calculate optimal vocabulary sizes
   - Train cluster tokenizers
   - Merge cluster tokenizers
   - Normalize final tokenizer
   - Save to permanent location
   - Clean up temporary files

## For manual execution
### Step 1:
Train monolingual tokenizers.
- **`train_indic_sentpiece_v1.py`** - SentencePiece tokenizer trainer on a corpus (adjusts vocab to max_vocab if vocab_size > max_vocab_size).
#### Usage 
```bash
python train_indic_sentpiece_v1.py -m model_type (unigram) -v vocab_size -d corpus_file_path -t threads -o output_path
```
### Step 2:
Generate clusters for the trained tokenizers
- **`cluster.py`** - Language clustering based on vocabulary overlap of trained monolingual tokenizers using KMedoids/KMeans with l2 and cosine distance metrics
#### Usage
```bash
python cluster.py --root_dir /path/to/tokenizers --output_dir cluster_output_dir --kmedoids_metric l2/cosine
```
- **`club_data_into_clusters.py`** - Groups corpus files according to cluster definitions
- **`train_clusters.py`** - Trains cluster tokenizers and stores the cluster wise tokenizer in output directory

#### Usage
```bash
python train_clusters.py <clusters_path(output of clusters.py)> <mono_lingual_tokenizers_path> <vocab_size> <cluster_corpus_dir> <output_dir>
```

- **`club_tokenizers.py`** - Merges individual cluster tokenizers into unified model

#### Usage
```bash
python club_tokenizers.py <clusters_path(output of clusters.py)> <mono_lingual_tokenizers_path> <vocab_size> <cluster_corpus_dir> <output_dir>
```

- **`normalize_final_tokenizer.py`** - Normalizes merged tokenizer using corpus statistics

#### Usage
```bash
python normalize_final_tokenizer.py merged_model_path cluster_count corpus_dir model_dir output_path
```
