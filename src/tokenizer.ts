"use worldcode";

import { readBlockData } from "./blockData";
import { TOKENIZER_VOCAB_FIRST, TOKENIZER_VOCAB_SECOND } from "./positions";

export function* loadVocab() {
  const firstBlockData = yield* readBlockData(TOKENIZER_VOCAB_FIRST);
  const firstArray: string[] = JSON.parse(firstBlockData).map((s: string) => {
    if (!s.startsWith("|")) return s;
    let result = "";
    for (let i = 0; i < s.length / 2; i++) {
      result += s[i * 2 + 1];
    }
    return result;
  });
  const secondBlockData = yield* readBlockData(TOKENIZER_VOCAB_SECOND);
  const secondArray: string[] = JSON.parse(secondBlockData).map((s: string) => {
    if (!s.startsWith("|")) return s;
    let result = "";
    for (let i = 0; i < s.length / 2; i++) {
      result += s[i * 2 + 1];
    }
    return result;
  });
  const map = new Map();
  for (const [i, s] of firstArray.entries()) {
    map.set(s, i);
  }
  for (const [i, s] of secondArray.entries()) {
    map.set(s, firstArray.length + i);
  }
  return { map, array: firstArray.concat(secondArray) };
}
function isAsciiPunctuation(cp: number) {
  return (
    (cp >= 33 && cp <= 47) ||
    (cp >= 58 && cp <= 64) ||
    (cp >= 91 && cp <= 96) ||
    (cp >= 123 && cp <= 126)
  );
}

const preTokenize = (text: string): string[] => {
  text = text.toLowerCase();
  const words = [];
  let current = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0)!;
    if (ch === " " || ch === "\t" || ch === "\n") {
      if (current) {
        words.push(current);
        current = "";
      }
    } else if (isAsciiPunctuation(cp)) {
      if (current) {
        words.push(current);
        current = "";
      }
      words.push(ch);
    } else if (cp >= 48 && cp <= 57) {
      // 0-9
      if (current) {
        words.push(current);
        current = "";
      }
      words.push(ch); // 数字は1文字ずつ独立した単語として切り離す
    } else {
      current += ch;
    }
  }
  if (current) words.push(current);
  return words;
};
const wordpieceTokenizeWord = (
  word: string,
  vocabMap: Map<string, number>,
  unkId: number = vocabMap.get("[UNK]")!,
  maxChars: number = 1000,
) => {
  if (word.length > maxChars) return [unkId];
  const outputIds: number[] = [];
  let start = 0;
  while (start < word.length) {
    let end = word.length;
    let curId = -1;
    while (start < end) {
      let substr = word.slice(start, end);
      if (start > 0) substr = "##" + substr;
      if (vocabMap.has(substr)) {
        curId = vocabMap.get(substr)!;
        break;
      }
      end -= 1;
    }
    if (curId === -1) return [unkId]; // 単語全体をUNK扱い(WordPiece標準動作)
    outputIds.push(curId);
    start = end;
  }
  return outputIds;
};

export const encode = (text: string, map: Map<string, number>) => {
  let ids: number[] = [];
  for (const w of preTokenize(text)) {
    ids = ids.concat(wordpieceTokenizeWord(w, map));
  }
  return ids;
};

export const decode = (ids: number[], array: string[]) => {
  let out = "";
  for (const id of ids) {
    const tok = array[id];
    if (tok === "[UNK]" || tok === "[EOS]") continue;
    if (tok.startsWith("##")) out += tok.slice(2);
    else out += (out.length > 0 ? " " : "") + tok;
  }
  return out;
};
