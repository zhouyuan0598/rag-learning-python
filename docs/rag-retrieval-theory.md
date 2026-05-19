# RAG 检索与评估学习笔记

这份笔记用于复习 RAG 质量评估实验台后续要学习的核心概念。当前项目已经有基础的 `ingest`、`search`、`ask` 和 `eval` 命令，下一步会围绕检索质量展开。

## 1. 为什么要先学检索评估

RAG 的回答质量很大程度取决于检索质量。如果正确资料没有被检索出来，后面的 prompt、LLM、引用格式都救不了结果。

基础链路是：

```text
load documents -> split chunks -> embed -> store -> retrieve -> build prompt -> answer
```

质量评估要回答的是：

```text
检索有没有找到正确 chunk？
正确 chunk 排得够不够靠前？
top-k 里有多少结果是真相关？
换 chunk size、top-k、检索算法后，结果是变好还是变差？
```

## 2. BM25 是什么

BM25 是经典的关键词检索算法，可以理解为更实用的 TF-IDF 排序方法。它不理解深层语义，而是根据 query 里的词和文档 chunk 里的词来打分。

BM25 主要考虑四件事：

- 词是否出现：query 里的词出现在 chunk 里，相关性更高。
- 词出现次数：出现多次通常更相关，但不是线性增加，会做饱和处理。
- 词是否稀有：常见词区分度低，专有词、术语、编号、代码符号区分度高。
- 文档长度：长 chunk 更容易碰巧包含关键词，所以 BM25 会做长度归一化。

简化理解：

```text
BM25 分数 = 关键词匹配程度 * 关键词稀有度 * 文档长度修正
```

BM25 擅长：

- 精确术语，比如 `BM25`、`RAG`、`Chroma`。
- 人名、产品名、错误码、函数名、配置项。
- 用户问题和资料里使用相同关键词的场景。

BM25 不擅长：

- 同义改写。
- 概念相关但字面词不重合的问题。
- 需要语义理解的模糊问题。

## 3. 向量检索是什么

向量检索会先把文本变成 embedding，也就是一组数字向量。语义接近的文本，向量距离也应该更接近。

例如：

```text
用户问：RAG 适合什么场景？
文档写：企业知识库问答、文档助手、客服助手
```

即使文档里没有完全重复“适合什么场景”，向量检索也可能找到它，因为语义相近。

向量检索擅长：

- 语义相似。
- 问法和资料表达不完全一致。
- 概念解释类问题。

向量检索不擅长：

- 精确字符串、代码符号、编号。
- 新术语、冷门专有名词。
- embedding 模型没有很好覆盖的领域词。

## 4. BM25 和向量检索的差异

| 方法 | 核心依据 | 擅长 | 风险 |
| --- | --- | --- | --- |
| BM25 | 关键词匹配 | 精确词、术语、编号、代码符号 | 不理解同义表达 |
| 向量检索 | 语义相似度 | 概念匹配、自然语言改写 | 可能漏掉精确词 |
| Hybrid Search | 关键词 + 语义 | 兼顾召回稳定性和语义泛化 | 需要融合排序 |

真实 RAG 系统一般不会只押一种检索方式。关键词检索和向量检索互补，混合后通常更稳。

## 5. Hybrid Search 是什么

Hybrid Search 是混合检索。它把 BM25 和向量检索的结果合并，再重新排序。

常见流程：

```text
BM25 找一批候选
向量检索找一批候选
合并候选
融合分数或排名
输出最终 top-k
```

常见融合方式有两类。

### 分数融合

把 BM25 分数和向量相似度归一化后加权：

```text
final_score = 0.5 * bm25_score + 0.5 * vector_score
```

问题是 BM25 分数和向量相似度不是同一种量纲，直接相加容易失真，所以需要归一化。

### RRF

RRF 是 Reciprocal Rank Fusion。它不直接相信原始分数，只看排名。

直觉是：

```text
排第 1 的结果加很多分
排第 10 的结果加少一点分
多个检索器都排得靠前的结果，最终更靠前
```

简化公式：

```text
score = 1 / (k + rank)
```

其中 `rank` 是某个结果在某个检索器里的排名，`k` 是平滑常数，常见取值是 `60`。RRF 很适合学习阶段，因为实现简单，也避免了不同检索器分数不可比的问题。

## 6. Reranker 是什么

Reranker 是第二阶段排序器。

第一阶段先粗召回：

```text
BM25 / vector / hybrid -> top-20
```

第二阶段再精排：

```text
query + chunk -> reranker -> relevance score
```

最后把最相关的 top-5 交给 LLM。

Reranker 通常比单纯向量检索更准，因为它直接把问题和候选 chunk 放在一起判断相关性。代价是更慢，所以不会对全库所有 chunk 做 rerank，而是只重排第一阶段召回出来的一小批候选。

典型生产流程：

```text
粗召回 top-20 -> rerank top-5 -> build prompt -> LLM answer
```

## 7. 标准检索指标

当前项目第一版 eval 已经有 `retrieval_hit_rate`，后续可以升级为更标准的检索指标。

### recall@k

`recall@k` 看正确结果有没有出现在前 k 个结果里。

```text
recall@5 = 前 5 个结果命中的正确 citation 数 / 全部正确 citation 数
```

它回答：

```text
该找的资料有没有被找回来？
```

RAG 里 recall 很关键，因为正确资料找不回来，后面无法回答。

### precision@k

`precision@k` 看前 k 个结果里有多少是真的相关。

```text
precision@5 = 前 5 个结果中正确 citation 数 / 5
```

它回答：

```text
塞给 LLM 的上下文有多少是干净的？
```

precision 低会导致 prompt 里噪音多，LLM 更容易跑偏。

### MRR

MRR 是 Mean Reciprocal Rank，关注第一个正确结果排第几。

单个问题的 reciprocal rank：

```text
第一个正确结果排第 1 -> 1 / 1 = 1.0
第一个正确结果排第 2 -> 1 / 2 = 0.5
第一个正确结果排第 5 -> 1 / 5 = 0.2
没找到 -> 0
```

MRR 是所有问题的平均值。它回答：

```text
正确结果是不是排得足够靠前？
```

### nDCG

nDCG 是 Normalized Discounted Cumulative Gain。它适合一个问题有多个相关结果，并且相关程度不同的场景。

直觉是：

```text
越相关的结果越应该排前面
排得越靠后，收益打折越多
```

学习阶段可以先实现 `recall@k`、`precision@k`、`MRR`，等评测集支持相关性等级后再实现 nDCG。

## 8. 下一步项目改造建议

下一版可以把当前项目升级成更完整的检索实验台：

```text
BM25 retriever
vector retriever
hybrid retriever
recall@k / precision@k / MRR
experiment comparison
```

建议命令形态：

```bash
uv run rag eval data/eval/questions.jsonl --retriever vector --top-k 5
uv run rag eval data/eval/questions.jsonl --retriever bm25 --top-k 5
uv run rag eval data/eval/questions.jsonl --retriever hybrid --top-k 5
```

再进一步可以做实验矩阵：

```bash
uv run rag experiment data/eval/questions.jsonl \
  --retriever bm25,vector,hybrid \
  --chunk-size 300,500,800 \
  --top-k 3,5,10
```

重点不是命令变多，而是能回答：

```text
BM25 和向量检索分别在哪些问题上更强？
Hybrid 是否比单一检索更稳定？
chunk_size 改变后 recall@k 是升还是降？
top_k 增大后 precision 是否下降？
```

## 9. 当前项目中的对应位置

- `src/rag_learning/vector_store.py`：当前向量检索和 JSON/Chroma 存储位置。
- `src/rag_learning/evaluation.py`：当前评估数据解析和指标计算位置。
- `src/rag_learning/cli.py`：`rag ingest`、`rag search`、`rag ask`、`rag eval` 命令入口。
- `data/eval/questions.jsonl`：当前最小评测集样例。

后续如果实现 BM25，建议新增独立模块，而不是把逻辑塞进现有向量存储：

```text
src/rag_learning/retrievers.py
```

这样可以让 BM25、vector、hybrid 都实现同一种检索接口，方便 `rag eval --retriever ...` 做横向对比。
