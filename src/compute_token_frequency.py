import logging
import os
import shutil
import argparse
import sys
import json
from tqdm import tqdm
from collections import OrderedDict, Counter

from transformers import XLMRobertaTokenizerFast, AutoTokenizer

logging.basicConfig(level=logging.INFO)


def get_tokenizer_path(tokenizer_dir, tokenizer_type, lang, alpha, NV):
    return os.path.join(tokenizer_dir, tokenizer_type, lang, f"alpha-{alpha}_N-{NV}")


# getting tokenizer
def get_tokenizer(tokenizer_path):
    logging.info(f"Loading tokenizer from {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except ValueError:
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer not found at {tokenizer_path}")
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_path, unk_token="<unk>")
    return tokenizer


def save_token_frequency(tokens_with_freq, decoded_tokens_with_freq, out_path, name):
    """Function to save token frequencies and log arguments to a file"""

    # copy current script to the output directory
    shutil.copyfile(sys.argv[0], os.path.join(out_path, f"{name}_script.py"))
    # save the arguments
    with open(os.path.join(out_path, f"{name}_args.txt"), "w") as log_file:
        log_file.write(" ".join(sys.argv[1:]))

    for save_name, save_object in [
        (f"{name}.json", tokens_with_freq),
        (f"{name}_decoded.json", decoded_tokens_with_freq),
    ]:
        if not save_object:
            continue
        save_path = os.path.join(out_path, save_name)
        with open(save_path, "w", encoding="utf-8") as outfile:
            logging.info(f"Writing frequencies to {save_path}")
            json.dump(
                OrderedDict(save_object),
                outfile,
                indent=2,
                ensure_ascii=False,
            )


def batch(iterator, batch_size):
    """Yield elements from iterator in batches of batch_size."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def compute_frequencies(data_list, tokenizer, name="token_frequencies", pretokenized=False, output_path=None):
    """Compute token frequencies for a given tokenizer and data."""

    # open the train data
    batch_size = 10000
    vocab = tokenizer.get_vocab()
    counter = {token_id: 0 for token_id in vocab.values()}
    for data_path in data_list:
        logging.info(f"Reading lines from {data_path}")
        with open(data_path, "r") as f:
            # go through the file line by line in batches
            # NOTE: we strip the newline character from the end of each line
            # TODO: maybe we shouldn't do this?
            for line_batch in tqdm(batch(map(lambda s: s.rstrip(), f), batch_size)):
                if pretokenized:
                    for tokenized_line in line_batch:
                        for tok in tokenized_line.split():
                            idx = vocab[tok]
                            counter[idx] += 1
                else:
                    for tokenized_line in tokenizer(line_batch)["input_ids"]:
                        for idx in tokenized_line:
                            counter[idx] += 1

    id_to_token = {v: k for k, v in vocab.items()}
    tokens_with_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    decoded_tokens_with_freq = [
        (id_to_token[token_id], freq) for token_id, freq in tokens_with_freq
    ]

    save_token_frequency(
        tokens_with_freq, decoded_tokens_with_freq, output_path, name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_list", nargs="+", help="<Required> Set flag", required=True
    )
    parser.add_argument("-o", "--out_dir", type=str, required=True)
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        required=True,
        help="List of languages the tokenizer was trained on.",
    )
    
    # tokenizer parameters
    parser.add_argument(
        "--tokenizer_path", type=str, required=False, default=None
    )
    parser.add_argument(
        "-a", "--alpha", type=str, required=False, help="Balancing coefficient alpha."
    )
    parser.add_argument("-v", "--vocab_size", type=int, required=False)
    parser.add_argument("-t", "--type", type=str, required=False, default="unigram")
    parser.add_argument(
        "-n", "--name", type=str, required=False, default="token_frequencies"
    )
    
    args = parser.parse_args()
    languages_str = "-".join(args.languages)

    # load the tokenizer
    if not args.tokenizer_path:
        assert args.alpha is not None and args.vocab_size is not None and type is not None and args.out_dir is not None, (
            "If no tokenizer path is provided, alpha, vocab_size, type and out_dir must be provided."
        )
        tokenizer_path = get_tokenizer_path(args.out_dir, type, languages_str, args.alpha, args.vocab_size)

    tokenizer = get_tokenizer(tokenizer_path)
    compute_frequencies(data_list=args.data_list, tokenizer=tokenizer, name=args.name, pretokenized=False,
                        output_path=args.out_dir or tokenizer_path)
