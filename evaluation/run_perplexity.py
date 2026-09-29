import argparse
import json
import sys
from pathlib import Path
import numpy as np

# הוספת שורש המאגר ל-sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from evaluation.run_probes import _load_olmo_core
from evaluation.scoring import token_perplexity

def main():
    parser = argparse.ArgumentParser(description="Calculate held-out token perplexity")
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens-path", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, model_fn = _load_olmo_core(args.run_config, args.checkpoint, random_init=False)

    print(f"Loading {args.max_tokens} tokens from {args.tokens_path} via memmap...")
    tokens_mmap = np.memmap(args.tokens_path, dtype=np.uint32, mode="r")
    tokens = np.array(tokens_mmap[:args.max_tokens])

    print("Evaluating perplexity...")
    try:
        perplexity_score = token_perplexity(model_fn, tokens)
    except Exception:
        perplexity_score = token_perplexity(model, tokens)

    result = {
        "checkpoint": str(args.checkpoint),
        "tokens_path": str(args.tokens_path),
        "evaluated_tokens": len(tokens),
        "token_perplexity": float(perplexity_score)
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Perplexity: {perplexity_score:.4f}")
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()