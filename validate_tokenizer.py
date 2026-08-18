"""
SimpleStoriesの実際のtokenizerの挙動(normalizer/pre_tokenizer設定)を確認し、
JS移植版で想定しているロジック(greedy longest-match WordPiece +
簡易プレトークナイズ)を素朴にPythonで再実装して、本物のtokenizerと
出力が一致するか自動比較するスクリプト。

使い方:
  pip install torch transformers --break-system-packages
  python validate_tokenizer.py
"""

import re

from transformers import AutoTokenizer

MODEL_NAME = "SimpleStories/SimpleStories-1.25M"

TEST_SENTENCES = [
    "Once upon a time, there was a little girl named Lily.",
    "The dog barked loudly at the mailman!",
    "She wondered, \"What could it be?\"",
    "He's running quickly to the store.",
    "It's a beautiful, sunny day today.",
    "Unbelievably, the cat started talking.",
    "12345 and some_weird-input.",
]


def inspect_tokenizer_internals(tokenizer):
    print("=== tokenizer本体の設定確認 ===")
    print(f"class: {type(tokenizer).__name__}")
    print(f"unk_token: {tokenizer.unk_token!r}  id={tokenizer.unk_token_id}")
    print(f"eos_token: {tokenizer.eos_token!r}  id={tokenizer.eos_token_id}")

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        print(f"normalizer: {backend.normalizer}")
        print(f"pre_tokenizer: {backend.pre_tokenizer}")
    else:
        print("(backend_tokenizer属性なし。Fast tokenizerでない可能性)")
    print()


# --- ここから、JS実装のロジックをそのままPythonに写したもの ---

def _is_ascii_punctuation(cp: int) -> bool:
    return (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126)


def pretokenize_js_version(text: str):
    """JS版 preTokenize() の再現(v2)。
    - 小文字化
    - 幅広いASCII句読点(アンダースコア/ハイフン含む)を個別トークンとして分割
    - 数字は1文字ずつ独立した単語として分割
    """
    text = text.lower()
    words = []
    current = ""
    for ch in text:
        cp = ord(ch)
        if ch in (" ", "\t", "\n"):
            if current:
                words.append(current)
                current = ""
        elif _is_ascii_punctuation(cp):
            if current:
                words.append(current)
                current = ""
            words.append(ch)
        elif 48 <= cp <= 57:  # 0-9
            if current:
                words.append(current)
                current = ""
            words.append(ch)
        else:
            current += ch
    if current:
        words.append(current)
    return words


def wordpiece_tokenize_word_js_version(word: str, vocab: dict, unk_token: str, max_chars=100):
    """JS版 wordpieceTokenizeWord() の素朴な再現。"""
    if len(word) > max_chars:
        return [unk_token]

    output_tokens = []
    start = 0
    while start < len(word):
        end = len(word)
        cur_token = None
        while start < end:
            substr = word[start:end]
            if start > 0:
                substr = "##" + substr
            if substr in vocab:
                cur_token = substr
                break
            end -= 1
        if cur_token is None:
            return [unk_token]
        output_tokens.append(cur_token)
        start = end
    return output_tokens


def encode_js_version(text: str, vocab: dict, unk_token: str):
    tokens = []
    for w in pretokenize_js_version(text):
        tokens.extend(wordpiece_tokenize_word_js_version(w, vocab, unk_token))
    return tokens


# --- ここまで ---


def compare_on_sentence(tokenizer, vocab, sentence: str):
    # 本物のtokenizer
    real_ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
    real_tokens = tokenizer.convert_ids_to_tokens(real_ids)

    # JS移植ロジックの素朴なPython再現
    js_tokens = encode_js_version(sentence, vocab, tokenizer.unk_token)

    match = real_tokens == js_tokens
    status = "OK " if match else "NG "

    print(f"[{status}] {sentence!r}")
    print(f"      本物 : {real_tokens}")
    if not match:
        print(f"      JS版 : {js_tokens}")
    return match


def main():
    print(f"モデル読み込み中: {MODEL_NAME}\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab = tokenizer.get_vocab()

    inspect_tokenizer_internals(tokenizer)

    print("=== 本物のtokenizer vs JS移植ロジック(Python再現版) 比較 ===\n")
    results = []
    for sentence in TEST_SENTENCES:
        results.append(compare_on_sentence(tokenizer, vocab, sentence))
        print()

    ok_count = sum(results)
    print("=" * 60)
    print(f"一致: {ok_count}/{len(results)} 文")
    if ok_count < len(results):
        print(
            "不一致がある場合、上のnormalizer/pre_tokenizer設定を見て "
            "JS版のpreTokenize()を実際の挙動に合わせて修正してください。"
            "(例: 大文字小文字を正規化しているか、アポストロフィの扱い、"
            "数字の分割ルールなど)"
        )
    else:
        print("全文一致。JS移植ロジックはそのままBloxdに持ち込んで大丈夫そうです。")


if __name__ == "__main__":
    main()