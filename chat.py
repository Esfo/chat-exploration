"""
interactive inference for a trained model.

loads a model saved by training.save_model (a .pt written when model_output
is set in main.py), rebuilds the tokenizer from the tokenpath that travelled
with the model, and lets you chat with it from the terminal.

note: this is a small base language model trained to continue text, not an
instruction-tuned assistant. it will continue whatever you type rather than
answer it conversationally.

usage:
    python chat.py /home/sfo/data/models/model.pt
    python chat.py /home/sfo/data/models/model.pt --tokens /path/to/bpe.json
    python chat.py /home/sfo/data/models/model.pt --max-new-tokens 60
    python chat.py /home/sfo/data/models/model.pt --temperature 0.7 --top-k 40 --top-p 0.95
"""

import argparse
import os

from training import load_model, generate
from bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="chat with a trained model")

    parser.add_argument(
        "model",
        help="path to a saved model .pt (written when model_output is set)",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="where to run inference: auto (gpu if available), cuda, or cpu",
    )

    #sampling controls (see training.generate for what each does)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="softmax temperature; <1 sharper, >1 more random, 0 = greedy")
    parser.add_argument("--top-k", type=int, default=40,
                        help="sample only from the k most likely tokens (0 = off)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="nucleus sampling cutoff (0 = off)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="discourage repeating tokens; >1 stronger, 1.0 = off")

    #the model stores the tokenpath it was trained with; this overrides it in
    #case the token file has moved since training.
    parser.add_argument(
        "--tokens",
        default=None,
        help="override path to the tokenizer JSON (defaults to the one saved "
             "with the model)",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="how many tokens to generate per reply (default: 100)",
    )

    args = parser.parse_args()

    #load the model + the architecture config saved alongside it
    model, cfg = load_model(args.model, device=args.device)

    #figure out where the tokenizer lives: explicit override, else the path the
    #model was trained with.
    tokenpath = args.tokens or cfg.tokenpath
    if not tokenpath or not os.path.exists(tokenpath):
        raise SystemExit(
            f"tokenizer file not found at {tokenpath!r}. "
            "pass --tokens with the path to the tokenizer JSON."
        )

    tokenizer = BPETokenizer.load(tokenpath)

    #the model's output head is sized to the vocab it trained on; a mismatched
    #tokenizer would produce nonsense / index errors.
    if tokenizer.vocab_size != cfg.vocab_size:
        raise SystemExit(
            f"tokenizer vocab_size ({tokenizer.vocab_size}) does not match the "
            f"model's vocab_size ({cfg.vocab_size}). use the token file the "
            "model was trained with."
        )

    print(f"loaded model from {args.model}")
    print(f"vocab_size={cfg.vocab_size} context_length={cfg.context_length} "
          f"layer_widths={cfg.layer_widths} n_layers={cfg.n_layers}")
    print(f"position_enc={'RoPE' if cfg.use_rope else 'learned'} "
          f"mlp={'SwiGLU' if cfg.use_swiglu else 'GELU'} "
          f"device={next(model.parameters()).device}")
    print("type a prompt and press enter. ctrl-c or empty line + enter to quit.\n")

    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        #empty line exits, matching the prompt hint above
        if prompt.strip() == "":
            break

        text = generate(
            prompt,
            tokenizer,
            model,
            cfg,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"model> {text}\n")


if __name__ == "__main__":
    main()
