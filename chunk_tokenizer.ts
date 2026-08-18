/// <reference types="@types/node" />
// 多分半分にするだけでおけ
// 一部単語は伏せ字を食らうので
// |から始まる単語については偶数文字だけ読み、間は無視するアルゴリズム

import fs from "node:fs";
const ngWords = [
  "butt",
  "puzz",
  "fu",
  "kiss",
  "kis",
  "stroke",
  "horiz",
  "##ities",
  "crack",
  "##tering",
  "bite",
  "##fu",
  "##pp",
  "af",
  "bra",
  "cu",
  "##ching",
  "##ily",
  "##ass",
  "twink",
  "shi",
  "fing",
];
const text = fs.readFileSync("./out/tokenizer_vocab.json").toString();
const array = JSON.parse(text) as Array<string>;
const middle = Math.floor(array.length / 2);
const firstArray = array.slice(0, middle).map((s) => {
  if (ngWords.includes(s)) {
    let result = "|";
    for (const c of s) {
      result += c;
      result += "v";
    }
    return result.slice(0, -1);
  }
  return s;
});
const secondArray = array.slice(middle).map((s) => {
  if (ngWords.includes(s)) {
    let result = "|";
    for (const c of s) {
      result += c;
      result += "v";
    }
    return result.slice(0, -1);
  }
  return s;
});
fs.mkdirSync("./chunked_tokenizer", { recursive: true });
fs.writeFileSync("./chunked_tokenizer/first.json", JSON.stringify(firstArray));
fs.writeFileSync(
  "./chunked_tokenizer/second.json",
  JSON.stringify(secondArray),
);
