import torch
from draft_adapter.benchmark import benchmark_speculative

PROMPTS = [
      "请用中文介绍量子计算的基本原理",
      "Write a function that finds the k-th largest element in an array",
      "What are the key differences between Python and Rust?",
      "解释一下深度学习中的反向传播算法",
      "How does speculative decoding work in large language models?",]


def main():
    benchmark_speculative(
        target_model_id="Qwen/Qwen3-1.7B",
        draft_model_path="./draft_qwen",
        prompts=PROMPTS,
        max_new_tokens=128,
        num_speculative_tokens=5,
        temperature=0.0,
        require_svd_hybrid=True,
    )


if __name__ == "__main__":
    main()
