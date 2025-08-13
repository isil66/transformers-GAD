import os
# --- set env BEFORE importing transformers ---
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"   # avoid torchvision import path
os.environ["HF_HUB_ENABLE_XET"] = "0"             # avoid xet transport
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"     # faster, robust downloader
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"     # optional
CACHE_DIR = "/content/hf_cache"                   # resume-able cache

import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarAlignedOracleLogitsProcessor

# ---------------- your config ----------------
DATASET = "ebmoon/GAD-dataset"
SPLIT = "SLIA"   # or SLIA / CP
NUM_ITER = 100
MODEL_ID = "mistralai/mistral-7b-instruct-v0.2"
TRIE_PATH = f"tries/{SPLIT}"
RESULT_PATH = f"results/{SPLIT}"
DEVICE = "cuda"
DTYPE = torch.float16            # <-- T4-safe (changed from bfloat16)
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0
# ---------------------------------------------

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def eval_prob(model, tokenizer, id, prompt, grammar_str):
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar, save_log=True)
    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList([inf_nan_remove_processor, gad_oracle_processor])

    input_ids = tokenizer([prompt], add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"].to(model.device)

    history = []
    # inference-only context: faster & less memory
    with torch.inference_mode():
        for _ in tqdm(range(NUM_ITER), desc="Running Inference"):
            output = model.generate(
                input_ids,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                top_p=TOP_P,
                top_k=TOP_K,
                temperature=TEMPERATURE,
                logits_processor=logits_processors,
                repetition_penalty=REPETITION_PENALTY,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

            input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
            generated_tokens = output.sequences[0, input_length:].tolist()
            decoded_generation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            raw_likelihood = gad_oracle_processor.oracle_trie.raw_likelihood(generated_tokens)
            history.append({"tokens": generated_tokens, "raw_likelihood": raw_likelihood, "decoded_generation": decoded_generation})
            gad_oracle_processor.reset()

    make_dir(f"{RESULT_PATH}/{id}")
    with open(f"{RESULT_PATH}/{id}/{id}_gad.jsonl", "w") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")

def main():
    device = torch.device(DEVICE)

    # Tokenizer (resume + cache)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, resume_download=True, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Choose ONE of the two loading modes ----
    USE_4BIT = True  # safer on T4; set False if you prefer plain fp16

    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb,
            device_map="auto",
            resume_download=True,
            cache_dir=CACHE_DIR,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            resume_download=True,
            cache_dir=CACHE_DIR,
        )

    model.eval()  # generation only
    model.to(device)

    # NOTE: Only resize embeddings if you added new tokens; otherwise skip
    # model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset(DATASET, split=SPLIT)
    for data in dataset:
        id = data["id"]
        prompt = data["prompt"]
        grammar = data["grammar"]
        eval_prob(model, tokenizer, id, prompt, grammar)
        print(f"Evaluation finished: {id}")

if __name__ == "__main__":
    main()
