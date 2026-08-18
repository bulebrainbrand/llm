"use worldcode";
import "@bloxdjs/api";
import "./tokenizer";
import "./all";
globalThis.queue = [];
tick = () => {
  if (queue.length > 0) {
    for (let i = 0; i < 100; i++) {
      const result = queue[0].next();
      if (result.done) {
        queue.shift();
        break;
      }
    }
    for (let i = 0; i < 30; i++) {
      api.getBlock(32 + i * 32, 1, 96);
    }
  }
};
