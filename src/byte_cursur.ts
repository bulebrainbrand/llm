"use worldcode";
import { readBlockData } from "./blockData";
import { Manifest } from "./types";

export class ByteCursor {
  readonly manifest: Manifest;
  readonly coordFn: (index: number) => [number, number, number];
  chunkIndex: number;
  chunkText: string;
  constructor(
    manifest: Manifest,
    coordFn: (index: number) => [number, number, number],
  ) {
    this.manifest = manifest;
    this.coordFn = coordFn;
    this.chunkIndex = -1;
    this.chunkText = "";
  }

  *_ensureChunkLoaded(byteOffset: number) {
    const charOffset = byteOffset * 3;
    const neededChunk = Math.floor(charOffset / this.manifest.chunk_size);
    if (neededChunk !== this.chunkIndex) {
      const pos = this.coordFn(neededChunk);
      this.chunkText = yield* readBlockData(pos); // 境界を跨いだ時だけ読む
      this.chunkIndex = neededChunk;
    }
    return charOffset - this.chunkIndex * this.manifest.chunk_size;
  }

  // tensor内オフセットからlengthバイト読む(チャンク境界跨ぎも対応)
  *readBytes(tensorOffset: number, length: number) {
    const out = new Uint8Array(length);
    let remaining = length,
      srcOffset = tensorOffset,
      outIdx = 0;

    while (remaining > 0) {
      const localPos = yield* this._ensureChunkLoaded(srcOffset);
      const bytesLeftInChunk = Math.floor(
        (this.chunkText.length - localPos) / 3,
      );
      const bytesToRead = Math.min(remaining, bytesLeftInChunk);
      if (bytesToRead === 0) {
        throw new Error(
          `stalled read: chunk ${this.chunkIndex} exhausted at localPos=${localPos}, ` +
            `remaining=${remaining}, srcOffset=${srcOffset}, chunkIndex:${this.chunkIndex}`,
        );
      }
      for (let i = 0; i < bytesToRead; i++) {
        const p = localPos + i * 3;
        out[outIdx++] =
          (this.chunkText.charCodeAt(p) - 48) * 100 +
          (this.chunkText.charCodeAt(p + 1) - 48) * 10 +
          (this.chunkText.charCodeAt(p + 2) - 48);
      }
      srcOffset += bytesToRead;
      remaining -= bytesToRead;
    }
    return out;
  }
}
