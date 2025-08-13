import os
import json
import argparse
from transformers import AutoTokenizer

def add_decoded_to_jsonl(input_path: str, output_path: str, model_id: str, cache_dir: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, resume_download=True, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            tokens = obj.get("tokens", [])

            # Ensure token IDs are ints
            tokens = [int(t) for t in tokens]

            # Decode a single sequence -> plain string
            decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            dec_str = "".join(" " if t == "" else t for t in decoded)

            obj["decoded_generation"] = decoded
            obj["decoded_str"] = dec_str
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add decoded_generation to an existing JSONL of samples.")
    parser.add_argument("--input", default="scripts/eval/results/BV4/find_inv_eq_bvlshr0_4bit_gad.jsonl", help="Path to existing *.jsonl (e.g., results/SLIA/<id>/<id>_gad.jsonl)")
    parser.add_argument("--output", default="scripts/eval/results/BV4/updated_find_inv_eq_bvlshr0_4bit_gad.jsonl", help="Path to write new *.jsonl with decoded_generation added")
    parser.add_argument("--model-id", default="mistralai/mistral-7b-instruct-v0.2", help="HF model ID for tokenizer")
    parser.add_argument("--cache-dir", default="/content/hf_cache", help="Optional HuggingFace cache dir (e.g., /content/hf_cache)")
    args = parser.parse_args()

    add_decoded_to_jsonl(args.input, args.output, args.model_id, args.cache_dir)
