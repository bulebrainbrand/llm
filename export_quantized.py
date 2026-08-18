"""
SimpleStories-1.25M の重みを int8 に量子化し、Bloxd.io の Code Block
(1ブロックあたり16,000文字制限) に貼り付けられる形に Base64 チャンク分割して
書き出すスクリプト。

出力:
  out/manifest.json          -> 各テンソルの name/shape/scale/オフセット情報
  out/weights_chunks/chunk_0000.txt, chunk_0001.txt, ...
                              -> 量子化済み重み全体を1本のバイト列にして
                                 Base64化し、16,000文字ごとに分割したもの
  out/tokenizer_vocab.json   -> トークンID -> 文字列 の対応表(そのままJS配列に変換可能)

使い方:
  pip install torch transformers --break-system-packages
  python export_quantized.py

manifest.jsonの中身:
  {
    "chunk_size": 16000,
    "num_chunks": 83,
    "total_b64_len": 1327000,
    "tensors": [
      {"name": "model.embed_tokens.weight", "shape": [4096, 64], "scale": 0.00331, "offset": 0, "length": 262144},
      ...
    ]
  }
  offset/length は「量子化後の生バイト列(int8)」内でのバイト位置。
  Bloxd側でチャンクを連結してBase64デコードしたバイト列に対して、
  このoffset/lengthでスライスすれば各テンソルの生データを取り出せる。
"""

import base64
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "SimpleStories/SimpleStories-1.25M"
OUT_DIR = "out"
CHUNK_SIZE = 16000  # Bloxd Code Block 1つあたりの文字数上限


def quantize_tensor_int8(tensor: torch.Tensor):
    """対称量子化(per-tensor)。int8配列とscale(float)を返す。"""
    flat = tensor.detach().float()
    max_val = flat.abs().max().item()
    if max_val == 0:
        scale = 1.0
    else:
        scale = max_val / 127.0

    q = torch.clamp(torch.round(flat / scale), -127, 127).to(torch.int8)
    return q.cpu().numpy(), scale


def chunk_string(s: str, chunk_size: int):
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]


def export_weights(model, out_dir: str):
    manifest_tensors = []
    raw_chunks = []  # バイト列の断片を集めて最後に連結
    offset = 0

    # 名前順で固定(復元側と順序を一致させるため)
    named_params = sorted(model.named_parameters(), key=lambda kv: kv[0])

    for name, param in named_params:
        q_np, scale = quantize_tensor_int8(param.data)
        raw_bytes = q_np.tobytes()  # int8なので1要素=1バイト
        length = len(raw_bytes)

        manifest_tensors.append({
            "name": name,
            "shape": list(param.shape),
            "scale": scale,
            "offset": offset,
            "length": length,
        })

        raw_chunks.append(raw_bytes)
        offset += length

    full_bytes = b"".join(raw_chunks)
    b64_str = base64.b64encode(full_bytes).decode("ascii")
    chunks = chunk_string(b64_str, CHUNK_SIZE)

    chunks_dir = os.path.join(out_dir, "weights_chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    for i, c in enumerate(chunks):
        with open(os.path.join(chunks_dir, f"chunk_{i:04d}.txt"), "w") as f:
            f.write(c)

    manifest = {
        "chunk_size": CHUNK_SIZE,
        "num_chunks": len(chunks),
        "total_b64_len": len(b64_str),
        "total_raw_bytes": len(full_bytes),
        "tensors": manifest_tensors,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"量子化後の生バイト数: {len(full_bytes):,} bytes")
    print(f"Base64後の文字数:     {len(b64_str):,} chars")
    print(f"チャンク数:            {len(chunks)} 個 (chunk_size={CHUNK_SIZE})")
    print(f"-> {chunks_dir}/ に書き出し完了")
    print(f"-> {os.path.join(out_dir, 'manifest.json')} に書き出し完了")


def export_tokenizer(tokenizer, out_dir: str):
    vocab = tokenizer.get_vocab()  # token(str) -> id(int)
    id_to_token = [None] * len(vocab)
    for token, idx in vocab.items():
        id_to_token[idx] = token

    path = os.path.join(out_dir, "tokenizer_vocab.json")
    with open(path, "w") as f:
        json.dump(id_to_token, f, ensure_ascii=False)

    total_len = sum(len(t) for t in id_to_token if t)
    print(f"語彙サイズ: {len(id_to_token)} トークン (概算文字数: {total_len:,})")
    print(f"-> {path} に書き出し完了")
    if total_len > CHUNK_SIZE:
        print(
            "  ※ このJSONは16,000文字を超える可能性があります。"
            " 必要ならchunk_string()で同様に分割してください。"
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"モデル読み込み中: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {n_params:,}\n")

    print("=== 重みの量子化・書き出し ===")
    export_weights(model, OUT_DIR)

    print("\n=== トークナイザー語彙の書き出し ===")
    export_tokenizer(tokenizer, OUT_DIR)


if __name__ == "__main__":
    main()