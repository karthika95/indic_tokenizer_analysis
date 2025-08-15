#!/usr/bin/env python3
import argparse
from sentencepiece import sentencepiece_model_pb2 as sp_model
from pathlib import Path

def merge_sentencepiece_models(vocab_size, cluster_count, cluster_dir, output_path):
    cluster_models = [
        str(Path(cluster_dir) / f"cluster_{i}.model")
        for i in range(1, cluster_count + 1)
    ]

    base_model_path = cluster_models[0]
    if not Path(base_model_path).exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    # Load base model
    merged_model = sp_model.ModelProto()
    merged_model.ParseFromString(open(base_model_path, "rb").read())

    # Track existing tokens
    existing_tokens = set(piece.piece for piece in merged_model.pieces)

    # Merge other models
    for model_path in cluster_models[1:]:
        if not Path(model_path).exists():
            print(f"[WARN] Skipping missing model: {model_path}")
            continue
        print(f"[INFO] Adding from {model_path}...")
        model_proto = sp_model.ModelProto()
        model_proto.ParseFromString(open(model_path, "rb").read())

        for piece in model_proto.pieces:
            if piece.piece not in existing_tokens:
                merged_model.pieces.append(piece)
                existing_tokens.add(piece.piece)

    # Save merged model
    with open(output_path, "wb") as f:
        f.write(merged_model.SerializeToString())
    print(f"✅ Merged model saved as {output_path} with {len(merged_model.pieces)} tokens.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge SentencePiece models from multiple clusters.")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size per cluster.")
    parser.add_argument("--cluster_count", type=int, required=True, help="Number of clusters.")
    parser.add_argument("--cluster_dir", type=str, required=True, help="Directory containing cluster .model files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path for merged output .model file.")

    args = parser.parse_args()

    merge_sentencepiece_models(args.vocab_size, args.cluster_count, args.cluster_dir, args.output_path)