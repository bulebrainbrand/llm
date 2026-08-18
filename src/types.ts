"use worldcode";
export interface TensorMeta {
  name: string;
  shape: number[];
  scale: number;
  offset: number; // 生バイト列内でのバイトオフセット
  length: number; // バイト数
}

export interface Manifest {
  encoding: "decimal3";
  chunk_size: number;
  num_chunks: number;
  total_encoded_len: number;
  total_raw_bytes: number;
  tensors: TensorMeta[];
}
