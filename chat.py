"""
interactive inference for a trained PyTorch model.

loads a checkpoint saved by training.save_model (a .pt file written when
model_output is set in main.py), rebuilds the tokenizer from the tokenpath that
travelled with the model, and lets you chat with it from the terminal.

note: this is a small base language model trained to continue text, not an
instruction-tuned assistant. it will continue whatever you type rather than
answer it conversationally.

usage:
    python chat.py /home/sfo/data/models/model.pt
    python chat.py /home/sfo/data/models/model.pt --tokens /path/to/tokens.jsonl
    python chat.py /home/sfo/data/models/model.pt --max-new-tokens 60
    python chat.py /home/sfo/data/models/model.pt --temperature 0.7 --top-k 40 --top-p 0.9
"""

import argparse
import os

from training import (
    build_tokenizer_from_jsonl,
    generate,
    load_model,
    resolve_device,
)


def main():
    parser = argparse.ArgumentParser(description="chat with a trained model")

    parser.add_argument(
        "model",
        help="path to a saved PyTorch model .pt (written when model_output is set)",
    )

    #the model stores the tokenpath it was trained with; this overrides it in
    #case the token file has moved since training.
    parser.add_argument(
        "--tokens",
        default=None,
        help="override path to the tokenizer JSONL (defaults to the one saved with the model)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="where to run inference (default: auto)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="how many tokens to generate per reply (defaults to checkpoint config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature; 0 means greedy (defaults to checkpoint config)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="keep only this many most likely tokens; 0 disables (defaults to checkpoint config)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="nucleus sampling probability mass; 1.0 disables (defaults to checkpoint config)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="always choose the highest-logit token instead of sampling",
    )

    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, _ = load_model(args.model, map_location=device)
    model.to(device)
    model.eval()

    #figure out where the tokenizer lives: explicit override, else the path the
    #model was trained with.
    tokenpath = args.tokens or cfg.tokenpath
    if not tokenpath or not os.path.exists(tokenpath):
        raise SystemExit(
            f"tokenizer file not found at {tokenpath!r}. "
            "pass --tokens with the path to the token JSONL."
        )

    tokenizer = build_tokenizer_from_jsonl(tokenpath)

    #the model's output head is sized to the vocab it trained on; a mismatched
    #token file would produce nonsense / index errors.
    if tokenizer.vocab_size != cfg.vocab_size:
        raise SystemExit(
            f"tokenizer vocab_size ({tokenizer.vocab_size}) does not match the "
            f"model's vocab_size ({cfg.vocab_size}). use the token file the "
            "model was trained with."
        )

    print(f"loaded model from {args.model}")
    print(
        f"vocab_size={cfg.vocab_size} context_length={cfg.context_length} "
        f"d_model={cfg.d_model} n_layers={cfg.n_layers} device={device}"
    )
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
            greedy=args.greedy or None,
            device=device,
        )
        print(f"model> {text}\n")


if __name__ == "__main__":
    main()
