/**
 * SimpleStories-1.25M (Llamaアーキテクチャ) の1トークン分forwardパス。
 * 重みは全て ByteCursor 経由で必要な範囲だけ遅延読み込みする。
 *
 * 前提インターフェース(既存の ByteCursor 実装がこの形と違う場合は
 * readBytes の呼び出し部分だけ合わせてください):
 *
 *   class ByteCursor {
 *     readBytes(tensorOffset: number, length: number): Uint8Array;
 *   }
 *
 * config.json の実測値:
 *   hiddenSize=128, numLayers=4, numAttentionHeads=4, numKeyValueHeads=2,
 *   headDim=32, intermediateSize=341, vocabSize=4019, tieWordEmbeddings=true
 */

import { TensorMeta } from "./types";

// ==== 型定義 ====

interface LayerWeights {
  inputNorm: TensorMeta; // [hiddenSize]
  postAttnNorm: TensorMeta; // [hiddenSize]
  qProj: TensorMeta; // [numAttentionHeads*headDim, hiddenSize]
  kProj: TensorMeta; // [numKeyValueHeads*headDim, hiddenSize]
  vProj: TensorMeta; // [numKeyValueHeads*headDim, hiddenSize]
  oProj: TensorMeta; // [hiddenSize, numAttentionHeads*headDim]
  gateProj: TensorMeta; // [intermediateSize, hiddenSize]
  upProj: TensorMeta; // [intermediateSize, hiddenSize]
  downProj: TensorMeta; // [hiddenSize, intermediateSize]
}

interface ModelConfig {
  hiddenSize: number;
  numLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  ropeTheta: number;
  rmsNormEps: number;
}

interface ModelWeights {
  embedTokens: TensorMeta; // [vocabSize, hiddenSize] tie_word_embeddings=trueなので出力層と共用
  layers: LayerWeights[];
  finalNorm: TensorMeta; // [hiddenSize]
}

// KVキャッシュ: レイヤーごとに、これまでの全トークン分のkey/valueを保持
interface KVCache {
  keys: Float32Array[][]; // [layer][position] -> [numKeyValueHeads*headDim]
  values: Float32Array[][]; // 同上
}
import type { ByteCursor as ByteCursorLike } from "./byte_cursur";
// ==== 量子化復元ユーティリティ ====

function toSignedInt8(v: number): number {
  return v > 127 ? v - 256 : v;
}

/** テンソルのrow行目(長さrowLen)を読んでdequantizeする。 */
function* readRowDequantized(
  cursor: ByteCursorLike,
  tensor: TensorMeta,
  row: number,
  rowLen: number,
): Generator<any, Float32Array<ArrayBuffer>, any> {
  const raw = yield* cursor.readBytes(tensor.offset + row * rowLen, rowLen);
  const out = new Float32Array(rowLen);
  for (let i = 0; i < rowLen; i++) {
    out[i] = toSignedInt8(raw[i]) * tensor.scale;
  }
  return out;
}

/** ベクトルをそのままdequantizeして返す(RMSNormの重みなど、1次元の小さいテンソル用)。 */
function readVectorDequantized(cursor: ByteCursorLike, tensor: TensorMeta) {
  return readRowDequantized(cursor, tensor, 0, tensor.length);
}

/**
 * y = W x  (Wの形状は [outDim, inDim]、PyTorch Linearと同じ行優先レイアウト)
 * 行を1本ずつ順番に読みながら計算するので、ByteCursorの連続アクセス最適化が効く。
 */
function* linear(
  cursor: ByteCursorLike,
  weight: TensorMeta,
  input: Float32Array,
  outDim: number,
  inDim: number,
) {
  const out = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    const row = yield* readRowDequantized(cursor, weight, o, inDim);
    let sum = 0;
    for (let i = 0; i < inDim; i++) sum += row[i] * input[i];
    out[o] = sum;
  }
  return out;
}

// ==== RMSNorm ====

function rmsNorm(
  x: Float32Array,
  weight: Float32Array,
  eps: number,
): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < x.length; i++) sumSq += x[i] * x[i];
  const rms = Math.sqrt(sumSq / x.length + eps);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] / rms) * weight[i];
  return out;
}

// ==== RoPE (Rotary Position Embedding) ====

/** headDim次元のベクトルに対し、位置posに応じた回転を適用する(in-place)。 */
function applyRope(
  vec: Float32Array,
  headDim: number,
  pos: number,
  theta: number,
): void {
  const half = headDim / 2;
  for (let i = 0; i < half; i++) {
    const freq = 1.0 / Math.pow(theta, (2 * i) / headDim);
    const angle = pos * freq;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const x0 = vec[i];
    const x1 = vec[i + half];
    vec[i] = x0 * cos - x1 * sin;
    vec[i + half] = x0 * sin + x1 * cos;
  }
}

// ==== SiLU (Swish) 活性化関数 ====

function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

// ==== Attention (Grouped Query Attention) ====

function* attention(
  cursor: ByteCursorLike,
  layerWeights: LayerWeights,
  layerIdx: number,
  normedInput: Float32Array,
  pos: number,
  cfg: ModelConfig,
  kvCache: KVCache,
) {
  const {
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    hiddenSize,
    ropeTheta,
  } = cfg;
  const qDim = numAttentionHeads * headDim;
  const kvDim = numKeyValueHeads * headDim;
  const groupSize = numAttentionHeads / numKeyValueHeads; // 1クエリヘッドグループあたりのクエリヘッド数

  // Q, K, V を計算
  const q = yield* linear(
    cursor,
    layerWeights.qProj,
    normedInput,
    qDim,
    hiddenSize,
  );
  const k = yield* linear(
    cursor,
    layerWeights.kProj,
    normedInput,
    kvDim,
    hiddenSize,
  );
  const v = yield* linear(
    cursor,
    layerWeights.vProj,
    normedInput,
    kvDim,
    hiddenSize,
  );

  // ヘッドごとにRoPEを適用
  for (let h = 0; h < numAttentionHeads; h++) {
    applyRope(
      q.subarray(h * headDim, (h + 1) * headDim),
      headDim,
      pos,
      ropeTheta,
    );
  }
  for (let h = 0; h < numKeyValueHeads; h++) {
    applyRope(
      k.subarray(h * headDim, (h + 1) * headDim),
      headDim,
      pos,
      ropeTheta,
    );
  }

  // KVキャッシュに今回のk, vを追加
  kvCache.keys[layerIdx].push(k);
  kvCache.values[layerIdx].push(v);

  const seqLen = kvCache.keys[layerIdx].length; // 現在位置を含めた系列長
  const attnOut = new Float32Array(qDim);
  const scale = 1 / Math.sqrt(headDim);

  for (let qh = 0; qh < numAttentionHeads; qh++) {
    const kvh = Math.floor(qh / groupSize); // GQA: 複数のクエリヘッドが同じKVヘッドを共有
    const qVec = q.subarray(qh * headDim, (qh + 1) * headDim);

    // スコア計算 (causal: 自分より後ろの位置は見ない。今回は毎回最新posまでなので全部見てOK)
    const scores = new Float32Array(seqLen);
    let maxScore = -Infinity;
    for (let t = 0; t < seqLen; t++) {
      const kVec = kvCache.keys[layerIdx][t].subarray(
        kvh * headDim,
        (kvh + 1) * headDim,
      );
      let dot = 0;
      for (let i = 0; i < headDim; i++) dot += qVec[i] * kVec[i];
      scores[t] = dot * scale;
      if (scores[t] > maxScore) maxScore = scores[t];
    }

    // softmax
    let sumExp = 0;
    for (let t = 0; t < seqLen; t++) {
      scores[t] = Math.exp(scores[t] - maxScore);
      sumExp += scores[t];
    }
    for (let t = 0; t < seqLen; t++) scores[t] /= sumExp;

    // 加重和
    const outSlice = attnOut.subarray(qh * headDim, (qh + 1) * headDim);
    for (let t = 0; t < seqLen; t++) {
      const vVec = kvCache.values[layerIdx][t].subarray(
        kvh * headDim,
        (kvh + 1) * headDim,
      );
      const w = scores[t];
      for (let i = 0; i < headDim; i++) outSlice[i] += w * vVec[i];
    }
  }

  // 出力射影
  return yield* linear(cursor, layerWeights.oProj, attnOut, hiddenSize, qDim);
}

// ==== SwiGLU MLP ====

function* mlp(
  cursor: ByteCursorLike,
  layerWeights: LayerWeights,
  normedInput: Float32Array,
  cfg: ModelConfig,
) {
  const { hiddenSize, intermediateSize } = cfg;

  const gate = yield* linear(
    cursor,
    layerWeights.gateProj,
    normedInput,
    intermediateSize,
    hiddenSize,
  );
  const up = yield* linear(
    cursor,
    layerWeights.upProj,
    normedInput,
    intermediateSize,
    hiddenSize,
  );

  const swiglu = new Float32Array(intermediateSize);
  for (let i = 0; i < intermediateSize; i++) swiglu[i] = silu(gate[i]) * up[i];

  return yield* linear(
    cursor,
    layerWeights.downProj,
    swiglu,
    hiddenSize,
    intermediateSize,
  );
}

// ==== 埋め込み取得 (1トークン分だけ読む: 軽い) ====

function getTokenEmbedding(
  cursor: ByteCursorLike,
  embedTokens: TensorMeta,
  tokenId: number,
  hiddenSize: number,
) {
  return readRowDequantized(cursor, embedTokens, tokenId, hiddenSize);
}

// ==== 出力層 (tie_word_embeddings=true: 埋め込み行列を転置して使い回す) ====
// vocabSize行すべてを読む必要があるため、ここが唯一「全部読む」コスト。

function* computeLogits(
  cursor: ByteCursorLike,
  embedTokens: TensorMeta,
  hiddenState: Float32Array,
  vocabSize: number,
  hiddenSize: number,
) {
  const logits = new Float32Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    const row = yield* readRowDequantized(cursor, embedTokens, v, hiddenSize);
    let sum = 0;
    for (let i = 0; i < hiddenSize; i++) sum += row[i] * hiddenState[i];
    logits[v] = sum;
  }
  return logits;
}

// ==== 1トークン分のforwardパス本体 ====

function* forwardStep(
  cursor: ByteCursorLike,
  weights: ModelWeights,
  cfg: ModelConfig,
  tokenId: number,
  pos: number,
  kvCache: KVCache,
) {
  let hidden = yield* getTokenEmbedding(
    cursor,
    weights.embedTokens,
    tokenId,
    cfg.hiddenSize,
  );

  for (let l = 0; l < cfg.numLayers; l++) {
    const layerWeights = weights.layers[l];

    const inputNormWeight = yield* readVectorDequantized(
      cursor,
      layerWeights.inputNorm,
    );
    const normed1 = rmsNorm(hidden, inputNormWeight, cfg.rmsNormEps);
    const attnOut = yield* attention(
      cursor,
      layerWeights,
      l,
      normed1,
      pos,
      cfg,
      kvCache,
    );

    const hiddenAfterAttn = new Float32Array(cfg.hiddenSize);
    for (let i = 0; i < cfg.hiddenSize; i++)
      hiddenAfterAttn[i] = hidden[i] + attnOut[i];

    const postAttnNormWeight = yield* readVectorDequantized(
      cursor,
      layerWeights.postAttnNorm,
    );
    const normed2 = rmsNorm(
      hiddenAfterAttn,
      postAttnNormWeight,
      cfg.rmsNormEps,
    );
    const mlpOut = yield* mlp(cursor, layerWeights, normed2, cfg);

    hidden = new Float32Array(cfg.hiddenSize);
    for (let i = 0; i < cfg.hiddenSize; i++)
      hidden[i] = hiddenAfterAttn[i] + mlpOut[i];
  }

  const finalNormWeight = yield* readVectorDequantized(
    cursor,
    weights.finalNorm,
  );
  const normedFinal = rmsNorm(hidden, finalNormWeight, cfg.rmsNormEps);

  return yield* computeLogits(
    cursor,
    weights.embedTokens,
    normedFinal,
    cfg.vocabSize,
    cfg.hiddenSize,
  );
}

// ==== サンプリング (greedy: 最も確率の高いトークンを選ぶ) ====

function argmax(logits: Float32Array): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

// ==== 生成ループ ====

function* generate(
  cursor: ByteCursorLike,
  weights: ModelWeights,
  cfg: ModelConfig,
  promptTokenIds: number[],
  maxNewTokens: number,
  eosTokenId: number,
) {
  const kvCache: KVCache = {
    keys: Array.from({ length: cfg.numLayers }, () => []),
    values: Array.from({ length: cfg.numLayers }, () => []),
  };

  const generated: number[] = [...promptTokenIds];
  let logits: Float32Array = new Float32Array(cfg.vocabSize);

  // プロンプト部分を1トークンずつ流してKVキャッシュを構築
  for (let pos = 0; pos < promptTokenIds.length; pos++) {
    logits = yield* forwardStep(
      cursor,
      weights,
      cfg,
      promptTokenIds[pos],
      pos,
      kvCache,
    );
  }

  // 新規トークンを1個ずつ生成
  for (let step = 0; step < maxNewTokens; step++) {
    const nextToken = argmax(logits);
    generated.push(nextToken);
    if (nextToken === eosTokenId) break;

    const pos = generated.length - 1;
    logits = yield* forwardStep(cursor, weights, cfg, nextToken, pos, kvCache);
  }

  return generated;
}

export {
  ModelConfig,
  ModelWeights,
  LayerWeights,
  TensorMeta,
  KVCache,
  ByteCursorLike,
  forwardStep,
  generate,
  argmax,
};
