"use worldcode";
export function* waitLoad(pos: Readonly<[number, number, number]>) {
  while (!api.isBlockInLoadedChunk(pos[0], pos[1], pos[2])) {
    yield api.getBlock(pos[0], pos[1], pos[2]);
  }
}
export function* readBlockData(
  pos: Readonly<[number, number, number]>,
): Generator<unknown, string, any> {
  yield* waitLoad(pos);
  return api.getBlockData(pos[0], pos[1], pos[2]).persisted.shared.text;
}
