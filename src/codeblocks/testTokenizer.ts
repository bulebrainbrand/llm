"use codeblock{text_tokenizer}";

import { decode, encode, loadVocab } from "../tokenizer";
(function* () {
  const { map, array } = yield* loadVocab();
  encode;
  decode;
})();
