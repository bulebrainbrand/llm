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

def compare_on_sentence(tokenizer, vocab, sentence: str):
    # 本物のtokenizer
    real_ids = tokenizer(sentence, add_special_tokens=False)["input_ids"]
    real_tokens = tokenizer.convert_ids_to_tokens(real_ids)
    print(real_ids)

def main():
    print(f"モデル読み込み中: {MODEL_NAME}\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab = tokenizer.get_vocab()

    inspect_tokenizer_internals(tokenizer)
    results = []
    for sentence in TEST_SENTENCES:
        results.append(compare_on_sentence(tokenizer, vocab, sentence))
        print()
        
if __name__ == "__main__":
    main()