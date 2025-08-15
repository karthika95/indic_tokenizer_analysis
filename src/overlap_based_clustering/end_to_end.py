import os
import subprocess
from pathlib import Path

ROOT_DIR = "/home/karthika/saketh/RnD/tokenizer/normalized_iso_txt"
threads = 128
model_type = "unigram"
vocab_sizes = [4000,16000,32000,64000]

# Find all .txt files under ROOT_DIR
for vocab_size in vocab_sizes:
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                output_path = Path(f"./monolingual_tokenizers_and_clusters/vocab_{vocab_size}/{Path(file).stem}")

                # Skip if output directory already exists
                if Path(f"{output_path}.vocab").exists():
                    print(f"Skipping {file_path} (output already exists)")
                    continue

                # Create directory for output
                os.makedirs(output_path.parent, exist_ok=True)

                print(f"Processing: {file_path}")
                
                try:
                    subprocess.run([
                        "python", "train_indic_sentpiece_v1.py",
                        "-m", model_type,
                        "-v", str(vocab_size),
                        "-d", file_path,
                        "-t", str(threads),
                        "-o", str(output_path)
                    ], check=True)
                    print(f"Finished: {output_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {file_path}: {e}")
                    print(f"Skipping to next file...")
                    continue
                print("-----------------------------")
    for kmedoids_metric in ["cosine", "l2"]:
        print(f"Running clustering for metric: {kmedoids_metric}")
        subprocess.run([
            "python", "cluster.py",
            "--root_dir", f"./monolingual_tokenizers_and_clusters/vocab_{vocab_size}",
            "--output_dir", f"./clusters/vocab_size_{vocab_size}",
            "--vocab_ext", ".vocab",
            "--kmedoids_metric", kmedoids_metric,
        ])

    # Process each cluster separately to save space
    cluster_dir = f"./clusters/vocab_size_{vocab_size}"
    source_dir = "/home/karthika/saketh/RnD/tokenizer/normalized_iso_txt"
    
    # Find all cluster definition files
    cluster_files = []
    if os.path.exists(cluster_dir):
        for file in os.listdir(cluster_dir):
            if file.startswith("language_clusters_") and file.endswith(".txt"):
                cluster_files.append(file)
    
    for cluster_file in cluster_files:
        cluster_def_path = os.path.join(cluster_dir, cluster_file)
        cluster_name = cluster_file.replace("language_clusters_", "").replace(".txt", "")
        
        # Create temporary clustered corpus directory
        temp_corpus_dir = f"./temp_clustered_corpus_{cluster_name}"
        final_tokenizer_dir = f"./final_tokenizers/vocab_{vocab_size}_{cluster_name}"
        
        print(f"Processing cluster: {cluster_name}")
        
        try:
            # Step 1: Create clustered corpus for this specific cluster
            print(f"Creating clustered corpus for: {cluster_name}")
            subprocess.run([
                "python", "club_data_into_clusters.py",
                cluster_def_path,
                source_dir,
                temp_corpus_dir
            ], check=True)
            print(f"✅ Clustered corpus created in: {temp_corpus_dir}")
            
            # Step 2: Create final tokenizer directory early (before training)
            os.makedirs(final_tokenizer_dir, exist_ok=True)
            
            # Step 3: Calculate vocab sizes and train tokenizers
            print(f"Training tokenizers for cluster: {cluster_name}")
            
            # Import the functions from train_clusters.py
            import sys
            sys.path.append('.')
            from calculate_cluster_vocab_sizes import calculate_cluster_vocab_sizes, train_cluster_tokenizers
            
            # Calculate optimal vocab sizes for this cluster configuration
            vocab_dir = f"./monolingual_tokenizers_and_clusters/vocab_{vocab_size}"
            target_vocab = vocab_size * 4  # 4x the original for merged tokenizer
            
            clusters, individual_vocabs = calculate_cluster_vocab_sizes(
                cluster_def_path, vocab_dir, target_vocab
            )
            
            if clusters and individual_vocabs:
                # Train cluster tokenizers with calculated vocab sizes
                # Save models directly to final directory, read corpus from temp directory
                success = train_cluster_tokenizers(
                    temp_corpus_dir, individual_vocabs, final_tokenizer_dir, threads, model_type
                )
                
                if success:
                    # Step 4: Merge cluster tokenizers
                    cluster_count = len(individual_vocabs)
                    merged_output = f"{final_tokenizer_dir}/merged_tokenizer.model"
                    
                    subprocess.run([
                        "python", "club_tokenizers.py",
                        "--vocab_size", str(vocab_size),
                        "--cluster_count", str(cluster_count),
                        "--cluster_dir", final_tokenizer_dir,  # Models are now in final_tokenizer_dir
                        "--output_path", merged_output
                    ], check=True)
                    print(f"✅ Merged tokenizer created: {merged_output}")
                    
                    # Step 5: Normalize the merged tokenizer
                    final_output = f"{final_tokenizer_dir}/final_normalized_tokenizer.model"
                    
                    subprocess.run([
                        "python", "normalize_final_tokenizer.py",
                        merged_output,
                        str(cluster_count),
                        temp_corpus_dir,      # Corpus is still in temp directory
                        final_tokenizer_dir,  # Models are in final directory
                        final_output
                    ], check=True)
                    print(f"✅ Normalized tokenizer created: {final_output}")
                    
                    # Step 6: Save vocab configuration for reference
                    import json
                    vocab_config = {
                        'clusters': clusters,
                        'individual_vocab_sizes': individual_vocabs,
                        'target_total_vocab': target_vocab,
                        'cluster_count': cluster_count
                    }
                    with open(f"{final_tokenizer_dir}/vocab_config.json", 'w') as f:
                        json.dump(vocab_config, f, indent=2)
                    
                    print(f"✅ Final tokenizers saved to: {final_tokenizer_dir}")
                    
                else:
                    print(f"❌ Failed to train cluster tokenizers for: {cluster_name}")
            else:
                print(f"❌ Failed to calculate vocab sizes for: {cluster_name}")
            
        except Exception as e:
            print(f"❌ Error processing cluster {cluster_name}: {e}")
        
        finally:
            # Step 7: Clean up temporary corpus directory to save space
            if os.path.exists(temp_corpus_dir):
                print(f"🧹 Cleaning up temporary corpus directory: {temp_corpus_dir}")
                subprocess.run(['rm', '-rf', temp_corpus_dir], check=True)
                print(f"✅ Cleanup completed")
        
        print(f"Completed processing cluster: {cluster_name}")
        print("-" * 60)
    
    print(f"Completed all processing for vocab_size: {vocab_size}")
    print("="*50)