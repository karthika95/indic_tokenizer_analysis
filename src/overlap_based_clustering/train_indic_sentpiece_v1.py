import sentencepiece as spm
import time
import argparse
import os
import gc

def main(data_path, model_type, vocab_size, model_prefix, num_threads):
    # Check file size and adjust settings accordingly  
    # Force garbage collection to free memory
    gc.collect()

    # SentencePiece training configuration
    split_by_unicode_script = True
    split_by_number = True
    split_by_whitespace = True
    split_digits = False
    train_extremely_large_corpus = True
    
    # Add memory management for very large files
    additional_params = {}
    start_time = time.time()

    try:
        base_params = {
            'input': data_path,
            'model_prefix': model_prefix,
            'model_type': model_type,
            'vocab_size': vocab_size,
            'split_by_unicode_script': split_by_unicode_script,
            'split_by_number': split_by_number,
            'split_by_whitespace': split_by_whitespace,
            'split_digits': split_digits,
            'num_threads': num_threads,
            'train_extremely_large_corpus': train_extremely_large_corpus
        }
        
        # Merge additional memory management parameters
        train_params = {**base_params, **additional_params}
        
        spm.SentencePieceTrainer.train(**train_params)
        print(f"Training completed with vocab size: {vocab_size}")
        
    except RuntimeError as e:
        if "Vocabulary size too high" in str(e):
            print(f"Error: {e}")
            
            # Extract the maximum allowed vocab size from error message
            import re
            match = re.search(r'set it to a value <= (\d+)', str(e))
            if match:
                max_vocab_size = int(match.group(1))
                print(f"Adjusting vocab size from {vocab_size} to maximum possible: {max_vocab_size}")
                
                # Use the same parameters for retry
                retry_params = {**base_params, **additional_params}
                retry_params['vocab_size'] = max_vocab_size
                
                spm.SentencePieceTrainer.train(**retry_params)
                print(f"Training completed with adjusted vocab size: {max_vocab_size}")
            else:
                print(f"Could not extract max vocab size from error: {e}")
                raise e
        else:
            raise e

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, required=True)
    parser.add_argument("-v", "--vocab_size", type=int, required=True)
    parser.add_argument("-d", "--data_path", type=str, required=True)
    parser.add_argument("-t", "--threads", type=int, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    
    args = parser.parse_args()

    data_path = args.data_path
    model_type = args.model_type
    vocab_size = args.vocab_size
    model_prefix = args.output

    num_threads = args.threads

    main(data_path, model_type, vocab_size, model_prefix, num_threads)