"use worldcode";
export const readBlockData = (pos: Readonly<[number, number, number]>): any =>
  api.getBlockData(pos[0], pos[1], pos[2]).persisted.shared.text;
