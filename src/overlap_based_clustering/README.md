# Indic Language Tokenizer Pipeline

A complete end-to-end pipeline for training clustered tokenizers for Indic languages using SentencePiece.

## Overview

This pipeline trains monolingual tokenizers, clusters languages by vocabulary similarity, creates clustered corpora, and trains optimized cluster tokenizers with automatic vocabulary size adjustment.

## Files

### Core Pipeline
- **`end_to_end.py`** - Main end to end pipeline
- **`train_indic_sentpiece_v1.py`** - SentencePiece tokenizer trainer on a corpus (adjusts vocab to max_vocab if vocab_size > max_vocab_size).
- **`cluster.py`** - Language clustering based on vocabulary overlap of trained monolingual tokenizers using KMedoids/KMeans with l2 and cosine distance metrics
- **`club_data_into_clusters.py`** - Groups corpus files according to cluster definitions

### Cluster Tokenizer Training
- **`calculate_cluster_vocab_sizes.py`** - Calculates vocab sizes for the new clusters
- **`club_tokenizers.py`** - Merges individual cluster tokenizers into unified model
- **`normalize_final_tokenizer.py`** - Normalizes merged tokenizer using corpus statistics

### Configuration
- **`train_indic_sentpiece.sh`** - Bash script for single language tokenizer training

## Usage

### Run Complete Pipeline
```bash
python end_to_end.py
```

### Individual Components
```bash
# Train single tokenizer
python train_indic_sentpiece_v1.py -m unigram -v 4000 -d data.txt -t 128 -o output

# Cluster languages
python cluster.py --root_dir ./vocab_dir --output_dir ./clusters --kmedoids_metric cosine

# Train cluster tokenizers
python train_clusters.py cluster_def.txt vocab_dir 256000 corpus_dir
```

## Directory Structure

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

## Configuration

### Vocabulary Sizes
- Default: `[4000, 16000, 32000, 64000]`
- Automatically adjusted if corpus is too small

### Clustering
- **Metrics**: cosine, L2
- **Cluster counts**: 2-9 (configurable in `cluster.py`)

### Paths
- **Source corpus**: `/home/karthika/saketh/RnD/tokenizer/normalized_iso_txt`
- **Output**: `./final_tokenizers/`

## Key Features

- **🔧 Automatic vocab size adjustment** when corpus is too small
- **💾 Space-efficient processing** - one cluster at a time
- **📊 Corpus-based normalization** for realistic token probabilities
- **🔄 Error handling** continues processing despite individual failures
- **📂 Organized output** with clear naming conventions

## Output

The main output is `final_normalized_tokenizer.model` in each cluster directory, optimized for the specific language grouping and ready for production use.