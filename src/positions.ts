"use worldcode";
export const TOKENIZER_VOCAB_FIRST = [-1, 1, 64] as const;
export const TOKENIZER_VOCAB_SECOND = [-1, 1, 66] as const;
export const MANIFEST_POSITION = [-1, 1, 96] as const;

export const MODEL_WRIGHT_POSITION = [-1, 1, 98] as const;

export const MODEL_CONFIG_POSITION = [-1, 1, 100] as const;

export const getModelWeight = (index: number): [number, number, number] => [
  32 + index * 4,
  1,
  96,
];
