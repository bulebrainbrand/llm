"""
SimpleStories-1.25M の重みを int8 に量子化し、Bloxd.io の Code Block
(1ブロックあたり16,000文字制限) に貼り付けられる形にチャンク分割して
書き出すスクリプト。

※Base64ではなく「固定長3桁の10進数」でエンコードしている。
  Bloxdには伏せ字(NGワード)フィルタが存在し、Base64/Hexのアルファベット
  (英字を含む)だと偶然NGワードの綴りが出現して保存時に文字が書き換わる
  リスクがあるため。数字のみ(0-9)なら英単語を構成しようがなく安全。
  トレードオフとしてサイズはBase64比で約1.7倍(3文字/バイト vs 1.33文字/バイト)
  に膨らむが、ストレージ容量は問題にならない前提なので許容している。

出力:
  out/manifest.json          -> 各テンソルの name/shape/scale/オフセット情報
  out/weights_chunks/chunk_0000.txt, chunk_0001.txt, ...
                              -> 量子化済み重み全体を1本のバイト列にして
                                 10進数(1バイト=3桁)化し、チャンク分割したもの
  out/tokenizer_vocab.json   -> トークンID -> 文字列 の対応表(そのままJS配列に変換可能)

使い方:
  pip install torch transformers --break-system-packages
  python export_quantized.py

manifest.jsonの中身:
  {
    "encoding": "decimal3",
    "chunk_size": 15999,
    "num_chunks": 125,
    "total_encoded_len": 1990000,
    "tensors": [
      {"name": "model.embed_tokens.weight", "shape": [4096, 64], "scale": 0.00331, "offset": 0, "length": 262144},
      ...
    ]
  }
  offset/length は「量子化後の生バイト列(int8)」内でのバイト位置(10進数化前)。
  Bloxd側で全チャンクを文字列として連結し、3文字ずつ区切ってparseIntすれば
  バイト値(0-255)の配列に戻る。そのバイト列に対してoffset/lengthでスライス
  すれば各テンソルの生データを取り出せる。
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "SimpleStories/SimpleStories-1.25M"
OUT_DIR = "out"
# Bloxdの伏せ字フィルタ回避のため、Base64/Hexではなく固定長3桁の10進数で
# 1バイトを表現する(例: 5 -> "005", 255 -> "255")。文字が数字のみになるため
# 単語を構成しようがなくなる。
BYTES_PER_CHAR_UNIT = 3
# 16,000文字上限のうち、3の倍数に切り詰めてバイト境界がチャンクをまたがない
# ようにする(1バイト分の桁が2つのチャンクに分割されるのを防ぐ)。
CHUNK_SIZE = 16000 - (16000 % BYTES_PER_CHAR_UNIT)  # -> 15999


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


def decimal_encode(raw_bytes: bytes) -> str:
    """バイト列を固定長3桁の10進数文字列に変換する(数字のみ、文字を含まない)。

    raw_bytesの各要素はPythonのbytesインデックスにより自動的に0-255の
    符号なし整数として得られる(int8の2の補数表現がそのままバイトパターン
    として保存されているため、符号付き/符号なしの変換は復元側でオフセット
    を戻すだけでよい)。
    """
    return "".join(f"{b:03d}" for b in raw_bytes)


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
    dec_str = decimal_encode(full_bytes)
    chunks = chunk_string(dec_str, CHUNK_SIZE)

    chunks_dir = os.path.join(out_dir, "weights_chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    for i, c in enumerate(chunks):
        with open(os.path.join(chunks_dir, f"chunk_{i:04d}.txt"), "w") as f:
            f.write(c)

    manifest = {
        "encoding": "decimal3",  # 1バイト = 固定長3桁の10進数文字列
        "chunk_size": CHUNK_SIZE,
        "num_chunks": len(chunks),
        "total_encoded_len": len(dec_str),
        "total_raw_bytes": len(full_bytes),
        "tensors": manifest_tensors,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"量子化後の生バイト数:   {len(full_bytes):,} bytes")
    print(f"10進数エンコード後の文字数: {len(dec_str):,} chars (3文字/バイト)")
    print(f"チャンク数:              {len(chunks)} 個 (chunk_size={CHUNK_SIZE})")
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