"""
SimpleStories-1.25M (or any small HF causal LM) の重みを疑似量子化して、
ビット幅ごとの劣化具合を perplexity と生成テキストの両方で比較するスクリプト。

疑似量子化(fake quantization)方式:
  実際にint8/int4/int2の型に詰め替えるのではなく、
  「量子化して丸めてから再度float に戻す」ことで、
  量子化誤差だけをシミュレートする。
  Bloxd移植前の「どのビット幅までなら実用か」の見極めに使う。

使い方:
  pip install torch transformers --break-system-packages
  python quantize_eval.py

  # 特定ビット幅だけ試したい場合
  python quantize_eval.py --bits 8 4 3 2

  # 埋め込み層/出力層だけ精度を落とさず、隠れ層だけ低ビットにする(混合精度)
  python quantize_eval.py --bits 2 --skip-embeddings
"""

import argparse
import copy
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "SimpleStories/SimpleStories-1.25M"

# perplexity計測用の評価テキスト(TinyStories/SimpleStories風の短い物語)。
# 実際のデータセットのサンプルに差し替えるとより正確な評価になる。
EVAL_TEXT = """
Once upon a time, there was a little girl named Lily. She loved to play in the
garden with her dog, Max. One sunny day, Lily found a small, shiny key under a
big tree. She picked it up and wondered what it could open. Max barked happily
and wagged his tail. Together, they decided to look for a door that the key
might fit. They searched all around the garden, behind the flowers and under
the old wooden bench, but they could not find anything. Lily was a little sad,
but then she smiled and said, "Maybe the key is just for keeping, not for
opening." She put the key in her pocket and continued to play with Max in the
warm afternoon sun.
""".strip()

PROMPTS = [
    "Once upon a time, there was a",
    "The little cat wanted to",
    "One day, a boy named",
]


def fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """対称量子化(symmetric, per-tensor)で丸めてから float に戻す。

    bits=8 -> [-127, 127] へマップ
    bits=2 -> [-1, 1] (実質ternary) へマップ
    """
    if bits >= 32:
        return tensor

    qmax = 2 ** (bits - 1) - 1  # 例: bits=8 -> 127, bits=2 -> 1
    if qmax < 1:
        qmax = 1

    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor

    scale = max_val / qmax
    q = torch.clamp(torch.round(tensor / scale), -qmax, qmax)
    return q * scale


def quantize_model(model, bits: int, skip_embeddings: bool):
    """モデルをin-placeで疑似量子化したコピーを返す。"""
    qmodel = copy.deepcopy(model)
    for name, param in qmodel.named_parameters():
        is_embed_or_head = ("embed" in name) or ("lm_head" in name)
        if skip_embeddings and is_embed_or_head:
            continue
        with torch.no_grad():
            param.data = fake_quantize_tensor(param.data, bits)
    return qmodel


@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str) -> float:
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"]
    outputs = model(input_ids, labels=input_ids)
    # outputs.loss は平均 cross-entropy (nats)。perplexity = exp(loss)
    return math.exp(outputs.loss.item())


@torch.no_grad()
def sample_generations(model, tokenizer, prompts, max_new_tokens=40):
    texts = []
    for p in prompts:
        input_ids = tokenizer(p, return_tensors="pt")["input_ids"]
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: ビット幅間の比較を安定させるため
        )
        texts.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[32, 8, 4, 3, 2],
        help="比較したいビット幅のリスト (32=量子化なし)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="埋め込み層とlm_headは量子化せず精度を保つ(混合精度)",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    print(f"モデル読み込み中: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(args.model)
    base_model.eval()

    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"パラメータ数: {n_params:,}")
    print(f"混合精度(埋め込み/出力層は非量子化): {args.skip_embeddings}")
    print("=" * 70)

    results = []
    for bits in args.bits:
        label = "fp32 (量子化なし)" if bits >= 32 else f"int{bits}"
        print(f"\n--- {label} ---")

        if bits >= 32:
            model = base_model
        else:
            model = quantize_model(base_model, bits, args.skip_embeddings)
        model.eval()

        ppl = compute_perplexity(model, tokenizer, EVAL_TEXT)
        print(f"Perplexity: {ppl:.3f}")

        gens = sample_generations(model, tokenizer, PROMPTS)
        for prompt, gen in zip(PROMPTS, gens):
            print(f'  [{prompt!r}] -> {gen!r}')

        results.append((label, ppl))

    print("\n" + "=" * 70)
    print("まとめ (perplexity: 低いほど良い)")
    print("=" * 70)
    base_ppl = results[0][1]
    for label, ppl in results:
        delta = ppl - base_ppl
        print(f"{label:20s} ppl={ppl:8.3f}  (Δ from fp32: {delta:+.3f})")


if __name__ == "__main__":
    main()