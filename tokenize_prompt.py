#!/usr/bin/env python3
"""
tokenize_prompt.py — Tokeniza um texto e imprime os IDs separados por espaço.
Os IDs incluem o token BOS (128000) como primeiro token.

Saída pronta para passar ao binário C:
    ./bitnet $(python3.14 tokenize_prompt.py "seu prompt aqui") --max-gen 10

Uso:
    python3.14 tokenize_prompt.py "Birds fly south in winter"
    python3.14 tokenize_prompt.py --no-bos "Birds fly south in winter"
"""

import sys
import os

TIKTOKEN_PATH = "/home/cami/Downloads/HLS_min_dist_V0/venv/lib/python3.14/site-packages"
TOKENIZER_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.model")

def main():
    args = sys.argv[1:]
    bos  = True

    if "--no-bos" in args:
        bos  = False
        args = [a for a in args if a != "--no-bos"]

    if not args:
        print("Usage: python3.14 tokenize_prompt.py [--no-bos] \"prompt text\"", file=sys.stderr)
        sys.exit(1)

    prompt = " ".join(args)

    # Add tiktoken to path
    sys.path.insert(0, TIKTOKEN_PATH)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from tokenizer import Tokenizer
    except ImportError as e:
        print(f"ERROR: Cannot import Tokenizer: {e}", file=sys.stderr)
        print(f"  Tried tiktoken path: {TIKTOKEN_PATH}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(TOKENIZER_MODEL):
        print(f"ERROR: tokenizer.model not found at {TOKENIZER_MODEL}", file=sys.stderr)
        sys.exit(1)

    tok = Tokenizer(TOKENIZER_MODEL)
    ids = tok.encode(prompt, bos=bos, eos=False)

    # Print space-separated IDs for shell substitution
    print(" ".join(str(i) for i in ids))

if __name__ == "__main__":
    main()
