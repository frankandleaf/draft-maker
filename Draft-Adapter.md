# Draft-Adapter：让标准大模型快速获得专属草稿模型

目标：一个能设置参数+一键启动的软件包，对标准Transformer模型（Llama/Qwen/Mistral等）进行处理，产出一个显存占用极小、接受率达标、可直接接入vLLM的Draft Model。

核心原则：显存峰值压到最低（尤其是Teacher侧），速度优化到最高（以实测Wall-clock为准），残差流一致性绝对不可破坏。

工作：
1. 接受原模型与tokenizer，获取head dim，head size等模型信息。**放弃“任意nn.Module”的设想**，首版锁定标准GQA Decoder架构，明确不支持MLA/MoE/Mamba等非标结构。
2. 设置es因子，对embed_dim和FFN中间维度做整除，得到新草稿模型结构。head_dim保持冻结以确保RoPE兼容性。
3. **放弃逐层独立SVD**（会破坏残差流维度一致性导致模型崩溃）。采用SliceGPT风格正交投影+切片，或仅在FFN中间维（SwiGLU）做激活感知低秩分解。计算时采用Swift-SVD的增量协方差聚合策略，将显存开销降到最低。
4. 设置ls因子。**放弃迭代贪心删“相似度最高层”**（非全局最优且重算成本高）。采用ShortGPT的Block Influence(BI)全局排序，保护首尾层，直接一次性裁剪至目标层数。删层后需显式处理hidden states偏移。
5. 设置训练器进行蒸馏。主损失采用on-policy的top-K稀疏KL散度，辅以AdaSpec风格的token过滤与相对排序蒸馏。**打破显存瓶颈的关键在Teacher**：Teacher侧可选使用FP8/INT4量化推理。
6. 将产出的模型封装为标准HF格式，对齐词表与KV cache结构，直接适配vLLM/HF的draft_model proposer接口，避免重造解码轮子。
7. 进行投机解码推理测试。**拒绝只看平均接受率**，必须监控端到端Wall-clock加速比和P95延迟波动（接受率波动大反而会导致整体变慢）。

【避坑与差异化总结】
- 上述流程本质是 SliceGPT(宽度压缩) + ShortGPT/KnapSpec(深度剪枝) + DistillSpec(蒸馏) 的工程化集成，单点技术无新意。
- 核心卖点在于“端到端自动化流水线 + 显存优先约束 + vLLM一键启动”。
- 目标客户：给一个HF checkpoint，半小时内产出一个vLLM可直接加载、接受率≥60%的draft。