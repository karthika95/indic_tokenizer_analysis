#!/bin/bash

ROOT_DIR="/home/karthika/saketh/RnD/tokenizer/normalized_iso_txt"

# Iterate through all .txt files under ROOT_DIR
find "$ROOT_DIR" -type f -name "tel.txt" | while read -r file; do
    threads=128
    model_type="unigram"
    vocab_size=4000

    data_path="$file"
    filename="monolingual_tokenizers_and_clusters/vocab_size_4_$(basename "$file" .txt)"
    output="$filename"

    echo "Processing: $file"
    
    python train_indic_sentpiece_v1.py -m "$model_type" \
        -v "$vocab_size" \
        -d "$data_path" \
        -t "$threads" \
        -o "$output"

    echo "Finished: $output"
    echo "-----------------------------"
done
