"use worldcode";
import { readBlockData } from "./blockData";
import { ByteCursor } from "./byte_cursur";
import { generate } from "./main";
import {
  getModelWeight,
  MANIFEST_POSITION,
  MODEL_CONFIG_POSITION,
  MODEL_WRIGHT_POSITION,
} from "./positions";
import { decode, encode, loadVocab } from "./tokenizer";

export function* ai(input: string, addToken: number) {
  let result: number[];
  const { array } = yield* loadVocab();
  {
    let ids: number[];
    {
      const { map } = yield* loadVocab();
      ids = encode(input, map);
    }

    {
      const modelWright = JSON.parse(
        yield* readBlockData(MODEL_WRIGHT_POSITION),
      );
      const modelConfig = JSON.parse(
        yield* readBlockData(MODEL_CONFIG_POSITION),
      );
      const manifest = JSON.parse(yield* readBlockData(MANIFEST_POSITION));
      const byteCursor = new ByteCursor(manifest, getModelWeight);
      const generator = generate(
        byteCursor,
        modelWright,
        modelConfig,
        ids,
        addToken,
        2,
        (token) => {
          const text = decode(token, array);
          for (const id of api.getPlayerIds()) {
            api.setClientOption(id, "middleTextLower", text);
          }
        },
      );
      result = yield* generator;
    }
  }

  return decode(result, array);
}
