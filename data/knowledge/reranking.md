# Reranker 与两阶段检索

## 1. 为什么需要 reranker

第一阶段检索器通常追求召回，也就是先把可能相关的 chunk 找出来。BM25、vector 或 hybrid 都可能在 top-20 里包含正确资料，但排序不一定最理想。Reranker 的作用是在候选集上做第二阶段精排，把最相关的 chunk 放到更靠前的位置。

Reranker 通常会同时看到 query 和 candidate chunk，然后判断二者是否真正相关。它比单独 embedding 更细，因为它不是把 query 和 chunk 分别编码后比较距离，而是把两者放在一起理解。常见形式包括 cross-encoder、LLM rerank 和轻量打分模型。

## 2. 两阶段检索流程

典型流程是：第一阶段粗召回 top-20 或 top-50，第二阶段 rerank，最后只取 top-3 到 top-5 给 LLM。粗召回负责不要漏，reranker 负责排序干净。这个模式能同时兼顾 recall 和 precision。

如果直接让 reranker 扫描全库，成本会非常高。它需要逐个比较 query 和 chunk，延迟会随候选数量线性增长。因此生产系统一般不会对全部 chunk 做 rerank，而是先用便宜的检索器缩小范围。

## 3. reranker 能改善什么

Reranker 对相似概念干扰很有帮助。比如用户问 “precision@k 低代表什么”，粗召回可能同时返回 recall、MRR、nDCG 的说明。reranker 可以根据 query 更精确地判断 precision 相关段落应该排在前面。

Reranker 也能改善 hybrid 融合后的排序。RRF 只能根据排名融合，不理解文本细节。如果 BM25 和 vector 都给出一批候选，reranker 可以重新判断每个候选是否真正回答了问题，而不是只看原始名次。

## 4. reranker 的成本

Reranker 会增加延迟和费用。候选数越多，调用次数或批处理成本越高。大型 cross-encoder 或 LLM reranker 的效果通常更好，但在线服务中可能需要缓存、批量推理、超时降级和成本控制。

一个常见折中是：粗召回 top-30，rerank 前 30 个，最终给 LLM top-5。对于延迟敏感的场景，可以只 rerank top-10，或者只在 query 风险较高时 rerank。例如检索分数很接近、问题很长、用户要求严格引用时再启用。

## 5. 阈值过滤

除了 rerank，还可以使用 score threshold。阈值过滤的目标是：分数太低的 chunk 不要为了凑满 top-k 被塞进上下文。这样可以减少噪音，但也可能降低 recall，因为低分结果里有时也藏着正确资料。

阈值应该基于评估集调，而不是凭感觉写死。vector、BM25 和 hybrid 的分数尺度不同，阈值不能直接共用。vector 的 0.3、BM25 的 5.0、RRF 的 0.03 没有可比性。

## 6. 多路召回与候选池

生产系统常见做法是多路召回：BM25 召回一批，向量召回一批，标题或标签召回一批，最近更新文档再补一批。然后统一去重、rerank、截断。这样比单一路径更稳，但也更需要评估。

候选池大小很关键。候选池太小，正确 chunk 根本进不了 reranker；候选池太大，成本和延迟上升。候选池通常需要结合业务容忍度、模型上下文长度、文档规模和评估结果来调。

## 7. 什么时候不急着加 reranker

如果知识库很小、问题简单、chunk 已经很聚焦，reranker 的收益可能不明显。学习项目一开始应该先把文档、chunk、BM25、vector、hybrid 和基础指标跑明白，再加 reranker。

当你看到 recall 已经不错但 precision 或 MRR 不理想时，reranker 才更有价值。换句话说，正确资料已经被找回来了，只是排序不够靠前或上下文太脏，这正是 reranker 擅长处理的问题。
