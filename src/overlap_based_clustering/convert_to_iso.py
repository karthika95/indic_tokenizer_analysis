import os
from datasets import load_from_disk
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Safe script detection
def detect_script(text):
    if not text or not text.strip():
        return None
    ch = text.strip()[0]
    code = ord(ch)

    if 0x0900 <= code <= 0x097F:
        return sanscript.DEVANAGARI
    elif 0x0B80 <= code <= 0x0BFF:
        return sanscript.TAMIL
    elif 0x0980 <= code <= 0x09FF:
        return sanscript.BENGALI
    elif 0x0C80 <= code <= 0x0CFF:
        return sanscript.KANNADA
    elif 0x0D00 <= code <= 0x0D7F:
        return sanscript.MALAYALAM
    elif 0x0A80 <= code <= 0x0AFF:
        return sanscript.GUJARATI
    elif 0x0A00 <= code <= 0x0A7F:
        return sanscript.GURMUKHI
    elif 0x0C00 <= code <= 0x0C7F:
        return sanscript.TELUGU
    elif 0x0B00 <= code <= 0x0B7F:
        return sanscript.ORIYA
    else:
        return None  # Script not supported or already Latin

def to_iso15919(example):
    script = detect_script(example["text"])
    if script:
        try:
            example["text"] = transliterate(example["text"], script, sanscript.ISO15919)
        except Exception:
            # Skip problematic lines silently
            pass
    return example

src_root = "/home/karthika/saketh/RnD/tokenizer/normalized_01"
dst_root = "/home/karthika/saketh/RnD/tokenizer/normalized_iso"

os.makedirs(dst_root, exist_ok=True)

for dataset_name in os.listdir(src_root):
    src_path = os.path.join(src_root, dataset_name)
    dst_path = os.path.join(dst_root, dataset_name)

    if not os.path.isdir(src_path):
        continue
    if os.path.exists(dst_path):
        print(f"Skipping {dataset_name}, already exists at {dst_path}")
        continue

    print(f"Processing {dataset_name}...")

    try:
        dataset = load_from_disk(src_path)
        dataset_iso = dataset.map(to_iso15919)
        dataset_iso.save_to_disk(dst_path)
        print(f"✅ Saved converted dataset to {dst_path}")
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")

print("🎯 All datasets processed.")
