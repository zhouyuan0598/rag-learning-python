# RAG 检索与回答评估指标

## 1. 为什么要评估

RAG 系统不能只靠感觉调参。一个问题看起来答对了，可能只是碰巧；一次检索看起来命中，可能只是因为知识库太小。评估集的作用是把检索质量、排序质量、回答质量和延迟变成可比较的数字。

一个好的评估集应该包含真实问题、期望答案、期望 citation 和问题类型。它不需要一开始很大，但要覆盖常见任务、容易混淆的概念、精确关键词、语义改写和生产场景。每次改 chunk、embedding、BM25 参数或 reranker，都应该重新跑评估。

## 2. retrieval_hit_rate

retrieval_hit_rate 衡量每个问题的 top-k 结果里是否至少有一个正确 citation。只要命中一个，它就算这个问题 hit。这个指标直观，适合作为 smoke test，但不够细。

如果一个问题有三个正确 chunk，系统只找回一个，retrieval_hit_rate 仍然认为这个问题命中。它不能告诉你找回得全不全，也不能告诉你上下文里混了多少无关内容。因此它适合保留，但不应该是唯一指标。

## 3. recall@k

recall@k 衡量正确 chunk 有没有被前 k 个结果找回来。计算方式是：前 k 个结果命中的正确 citation 数，除以这个问题全部正确 citation 数。RAG 里 recall 很关键，因为正确资料没进入上下文，LLM 后面很难可靠回答。

recall@k 高说明系统不容易漏资料。代价是如果盲目增大 k，recall 往往会上升，但上下文噪音也会上升。因此 recall 必须和 precision、MRR、回答质量一起看。

## 4. precision@k

precision@k 衡量前 k 个结果里有多少是真的相关。计算方式是：前 k 个结果中正确 citation 数，除以前 k 个实际返回结果数。precision 低说明塞给 LLM 的上下文噪音多，模型更容易被无关段落干扰。

在很小的知识库里，precision 容易失真。比如全库只有 4 个 chunk，而 top-k 设置成 5，系统几乎会返回整个知识库。此时 recall 可能是 100%，但 precision 会被大量无关 chunk 拉低。扩大数据集后，这个指标才更接近真实表现。

## 5. MRR

MRR 是 Mean Reciprocal Rank，关注第一个正确结果排第几。第一个正确结果排第 1，得分是 1；排第 2，得分是 0.5；排第 3，得分是 0.333；没有找到，得分是 0。所有问题取平均就是 MRR。

MRR 对 RAG 很有用，因为 LLM 通常更容易受到前面上下文影响。即使 top-k 里有正确资料，如果它排在最后，模型可能先被无关内容带偏。MRR 下降通常说明排序需要优化。

## 6. nDCG

nDCG 适合有相关性等级的评测集。它不只判断相关或不相关，还可以表达“非常相关”“部分相关”“弱相关”。越相关的结果越应该排在越前面。搜索系统和推荐系统经常使用 nDCG。

学习项目第一阶段可以先不实现 nDCG，因为它需要更精细的标注。等评测集从 binary citation 扩展到 graded relevance 后，再加入 nDCG 会更有价值。

## 7. answer_contains_expected_rate

answer_contains_expected_rate 是一个简单的回答检查指标：生成答案里是否包含 expected_answer 字符串。在本项目的离线 LLM 模式下，它主要反映 expected_answer 是否出现在检索上下文里，不等同于真实 LLM 回答质量。

真实生产评估还需要人工标注、LLM-as-judge、引用一致性检查和事实一致性检查。比如答案包含关键词不代表完整正确，答案引用了正确文档也不代表推理没有错误。离线指标只是学习检索链路的第一步。

## 8. 延迟指标

avg_retrieval_ms 衡量平均检索耗时，avg_answer_ms 衡量平均回答生成耗时。学习项目里 answer_ms 可能接近 0，因为 OfflineContextLLM 不调用外部模型。真实系统中，LLM 生成通常是最大延迟来源。

检索延迟也不能忽略。BM25、vector、hybrid、reranker 的成本不同。生产系统需要在质量和延迟之间做取舍：召回更多候选、rerank 更多 chunk 往往更准，但也更慢。
