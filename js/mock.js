// mock.js — simulates what your real backend will return
// Replace calls to mockQuery() with real fetch() calls later

const MOCK_RESPONSES = [
  {
    answer: "The attention mechanism computes compatibility between queries and keys using scaled dot-products. Scaling by the square root of the key dimension prevents vanishing gradients in high-dimensional spaces [1]. The Transformer replaces recurrence entirely with multi-head attention, enabling full parallelization [2].",
    citations: [
      { id: 1, doc: "attention_is_all_you_need.pdf", page: 3 },
      { id: 2, doc: "attention_is_all_you_need.pdf", page: 5 },
    ],
    insufficient: false,
  },
  {
    answer: null,
    citations: [],
    insufficient: true,
  },
  {
    answer: "FlashAttention rewrites the attention computation to be IO-aware, tiling the softmax operation to avoid materializing the full N×N attention matrix in HBM [1]. This reduces memory from O(N²) to O(N) and achieves 2–4× wall-clock speedup on standard benchmarks [2].",
    citations: [
      { id: 1, doc: "flash_attention.pdf", page: 2 },
      { id: 2, doc: "flash_attention.pdf", page: 7 },
    ],
    insufficient: false,
  },
];

let _mockIndex = 0;

/**
 * Simulates a backend POST /chats/:id/messages
 * Returns a promise that resolves after a fake delay.
 *
 * @param {string} query  - the user's question
 * @param {string[]} docs - list of uploaded doc names
 * @returns {Promise<{answer, citations, insufficient}>}
 */
function mockQuery(query, docs) {
  return new Promise((resolve) => {
    const delay = 900 + Math.random() * 600;
    setTimeout(() => {
      // Cycle through mock responses so you can see all states
      const response = MOCK_RESPONSES[_mockIndex % MOCK_RESPONSES.length];
      _mockIndex++;
      resolve(response);
    }, delay);
  });
}

/**
 * Simulates a backend POST /documents/upload
 * In reality this would send FormData and return doc metadata.
 */
function mockUpload(file) {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({ name: file.name, pages: Math.floor(Math.random() * 20) + 3 });
    }, 400);
  });
}