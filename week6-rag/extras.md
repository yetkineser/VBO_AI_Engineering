# Week 6 — Extras (Spec'in Üstüne Yapılanlar + Yol Haritası)

Spec'i karşılayan **base ödev** [`hr_rag_chatbot/`](hr_rag_chatbot/) altında — `main.py ingest` + `main.py chat` ile çalışır. Bu dosya **base'in üstüne ne eklendi** ve **sırada ne var** sorusunu cevaplıyor. Week 5'te yaptığın `--local` + `model_comparison.md` pattern'i Week 6'da daha kapsamlı hale getirildi.

---

## Mimari Genel Görünüm

```mermaid
flowchart TB
    subgraph Spec["📋 Homework Spec — Base"]
        S1["DirectoryLoader-style ingest<br/>13 metadata fields"]
        S2["Chroma persist<br/>vbo-aillm-bc-rag"]
        S3["create_agent + @tool retrieval"]
        S4["PostgresSaver + thread_id"]
        S1 --> S2 --> S3 --> S4
    end

    subgraph Done["✅ Already added (this repo)"]
        D1["3-profile chat: gemini / deepseek / ollama"]
        D2["Dual embedding profiles<br/>OpenRouter ↔ Ollama"]
        D3["Per-row try/except<br/>(Week 5 lesson)"]
        D4["JSONL test logger<br/>+ scorer.py"]
        D5["Mermaid'd model_comparison.md"]
    end

    subgraph Planned["🛠️ Planned (next iterations)"]
        P1["BGE/Cohere Reranker<br/>top-20 → top-4"]
        P2["Hybrid retrieval<br/>BM25 + vector + RRF"]
        P3["RAGAS evaluation<br/>(faithfulness, relevancy)"]
        P4["Streaming responses"]
        P5["Citation verification"]
        P6["Query rewriter"]
        P7["GraphRAG<br/>(only if dataset grows)"]
    end

    Spec --> Done --> Planned
```

---

## ✅ Şu ana kadar eklenenler

### 1. `--model {gemini,deepseek,ollama}` profilleri

Spec sadece Gemini istiyor. [`rag_agent.py:30-34`](hr_rag_chatbot/rag_agent.py#L30-L34)'te `PROFILES` dict ile üç model arkasını bağladım:

```mermaid
flowchart LR
    U["main.py test --model X"] --> P{PROFILES dict}
    P -->|gemini| G["init_chat_model<br/>openai:google/gemini-2.5-flash-lite<br/>OpenRouter"]
    P -->|deepseek| D["init_chat_model<br/>openai:deepseek/deepseek-chat<br/>OpenRouter"]
    P -->|ollama| O["init_chat_model<br/>openai:qwen2.5:14b<br/>localhost:11434/v1"]
    G --> A["create_agent(...)"]
    D --> A
    O --> A
```

**Neden değer:** Aynı kod, aynı sistem prompt, aynı retriever — sadece chat modeli değişiyor. Bu sayede karşılaştırma "model kalitesi" değişkenini izole ediyor (literatürde **single-variable ablation** denir, [BAIR'ın LLM eval guide'ı](https://bair.berkeley.edu/blog/2023/04/03/koala/) buna vurgu yapar).

### 2. İki embedding profili (OpenRouter ↔ Ollama)

[`vector_store.py:42-58`](hr_rag_chatbot/vector_store.py#L42-L58)'de `_embeddings(profile)` switch'i:

| Profile | Embedding | Dim | Koleksiyon |
|---|---|---|---|
| `openrouter` (spec) | `openai/text-embedding-3-small` | 1536 | `vbo-aillm-bc-rag` |
| `ollama` | `nomic-embed-text` | 768 | `vbo-aillm-bc-rag-ollama` |

**Neden iki koleksiyon?** Chroma dim-mismatched yazımı reddeder. Aynı koleksiyona iki farklı boyutta vektör koymaya çalışırsan `ValueError: Embedding dimension X does not match collection dimensionality Y`.

### 3. Per-row try/except + retry (Week 5 dersi)

[`main.py:100-119`](hr_rag_chatbot/main.py#L100-L119)'da test loop'u her sorunun çevresini try/except ile sarıyor — bir soru çökse diğerleri devam eder, hata JSONL'e yazılır. Week 5 [`feedback.md`](../week5-structured-output/feedback.md)'te aldığın `-1p`'nin doğal devamı.

Ollama testinde memory turn 3 `KeyError` attı — pipeline çökmedi, JSONL'e error olarak düştü. Bu sayede `scorer.py` hatayı görüp `memory_ok=2/3` bastı.

### 4. JSONL test logger + scorer.py

Her test runı `results/test_<model>.jsonl` üretiyor — her satır:
```json
{"model": "gemini", "section": "smoke", "question": "...", "answer": "...",
 "ok": true, "latency_s": 1.69, "timestamp": "..."}
```

`scorer.py` 6 metrik basıyor: `smoke_ok`, `memory_ok`, `cite_rate`, `hallu_cite`, `lang_consistency`, `avg_latency_s`. **`hallu_cite` ve `lang_consistency` özellikle değerli** — `smoke_ok=6/6` çıktığı halde Ollama'nın gerçekten broken olduğunu yakalayan iki metrik bunlar.

JSONL formatı [JSON Lines](https://jsonlines.org/) standardı — append-friendly, her satır bağımsız parse edilir.

### 5. 3-Model karşılaştırma raporu

[`model_comparison.md`](model_comparison.md): Week 5 stilinde tablolar + per-question breakdown + Ollama'nın neden battığının teknik analizi ([Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) referansıyla).

---

## 🛠️ Planlanan eklemeler

### 1. Reranker (BGE veya Cohere)

```mermaid
flowchart LR
    Q["query"] --> R1["vector retriever<br/>k=20"]
    R1 -->|20 chunks| RR["BGE Reranker<br/>cross-encoder"]
    RR -->|top 4 by relevance| A["agent prompt"]
```

**Neden:** Vector retrieval similarity'den sıralı geliyor, ama "iki vector birbirine yakın" ≠ "ikisi soruyu eşit iyi cevaplar". Cross-encoder reranker `(query, chunk)` çiftine birlikte bakıp gerçek alaka skoru üretir. **Tek en büyük ucuz kazanç** literatürde — [Pinecone'un benchmark'ında](https://www.pinecone.io/learn/series/rag/rerankers/) Recall@4 %15-30 artıyor.

LangChain entegrasyonu: [`ContextualCompressionRetriever` + `CrossEncoderReranker`](https://python.langchain.com/docs/integrations/retrievers/bge-rerank/). HuggingFace'deki [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) ücretsiz, lokalde çalışır.

### 2. Hybrid Retrieval (BM25 + Vector + RRF)

```mermaid
flowchart LR
    Q["query"] --> V["vector retriever<br/>(semantik)"]
    Q --> B["BM25 retriever<br/>(keyword)"]
    V --> RRF["Reciprocal Rank Fusion"]
    B --> RRF
    RRF --> Top["top-k birleşik liste"]
```

**Neden:** Embeddings semantik benzerliği yakalar ama exact-match'leri sıklıkla kaçırır. "Section 4.2" veya "TKT-88123" gibi tokenler vector space'de seyreltir. BM25 (keyword TF-IDF) o boşluğu doldurur. [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) iki listeyi tek skorla birleştirir.

LangChain'de [`EnsembleRetriever`](https://python.langchain.com/docs/how_to/ensemble_retriever/) bunu out-of-the-box veriyor.

### 3. RAGAS Evaluation Harness

```mermaid
flowchart LR
    JSONL["test_*.jsonl"] --> RAGAS["RAGAS"]
    Docs["retrieved chunks"] --> RAGAS
    RAGAS --> F["faithfulness<br/>(cevap retrieved'dan mı?)"]
    RAGAS --> AR["answer_relevancy<br/>(soruya cevap mı?)"]
    RAGAS --> CP["context_precision<br/>(retrieved alakalı mı?)"]
    RAGAS --> CR["context_recall<br/>(eksik retrieved var mı?)"]
```

**Neden:** Şu an `model_comparison.md`'deki manuel "PASS/FAIL" gözlemi sübjektif. [RAGAS](https://docs.ragas.io/) bu dört metriği LLM-as-judge ile otomatikleştiriyor — ödevi "Yetkin gözüyle iyi" değil "RAGAS skoru 0.87" diye savunabilirsin. Aynı 3-model karşılaştırması bilimsel zemine oturur.

Paper: [Es et al., 2023 — RAGAS](https://arxiv.org/abs/2309.15217).

### 4. Streaming Responses

[`agent.stream()`](https://python.langchain.com/docs/how_to/streaming/) ile token-by-token. CLI experience anında iyileşir, Gemini flash zaten 1.8s ortalama — streaming'le hissedilen latency yarıya iner.

### 5. Citation Verification

Her cevabın citation'ını **post-hoc doğrula** — claim'in `file_name` içindeki bir chunk'tan geldiğini kontrol et. Ollama'nın hayali `employee_benefits_policy.docx` halüsinasyonunu otomatik yakalar.

Pattern: [Anthropic'in Citations API'sinin](https://docs.anthropic.com/en/docs/build-with-claude/citations) tersine mühendisliği — model citation üretir, sen vector store'a sorup verify edersin.

### 6. Query Rewriter (Multi-Query Expansion)

Memory turn 2'de `"What about sick leave?"` Gemini'de çalıştı çünkü Gemini iyi paraphrase yaptı. Ollama'da çökmesi, onun bu adımı atlaması. **Pre-retrieval rewrite**: küçük bir LLM çağrısı ile pronoun-resolution + 3 paraphrase üret, hepsinin retrieval'ını birleştir.

LangChain: [`MultiQueryRetriever`](https://python.langchain.com/docs/how_to/MultiQueryRetriever/).

### 7. Knowledge Graph (sadece dataset büyürse)

Kullanıcı sordu — şu durumda ekosistem **overkill**. 8 kısa policy doc, birbirinden bağımsız konular. Vector RAG yeterli.

KG'nin parlayacağı senaryolar:
- Çalışan → manager → dept → policy → onay zinciri (multi-hop)
- "Travel approve eden kişinin manager'ı kim?"
- Cross-policy referansları (X policy'si Y policy'sine atıfta bulunuyor)

Canon: [Microsoft GraphRAG](https://github.com/microsoft/graphrag), [LangChain GraphRAG entegrasyonu](https://python.langchain.com/docs/integrations/graphs/), [Neo4j + LangChain](https://neo4j.com/labs/genai-ecosystem/langchain/).

---

## Değer Sıralaması (en yüksekten en düşüğe)

| # | Ekleme | Etki | Effort | Çalışan örnek var mı? |
|---|---|:---:|:---:|:---:|
| 1 | RAGAS eval | 🔴🔴🔴 | 1h | [docs](https://docs.ragas.io/en/stable/getstarted/evals/) |
| 2 | Reranker (BGE) | 🔴🔴🔴 | 30m | [LangChain how-to](https://python.langchain.com/docs/integrations/retrievers/bge-rerank/) |
| 3 | Hybrid retrieval | 🔴🔴 | 1h | [EnsembleRetriever](https://python.langchain.com/docs/how_to/ensemble_retriever/) |
| 4 | Citation verify | 🔴🔴 | 45m | yok — yazmak gerek |
| 5 | Query rewriter | 🔴🔴 | 30m | [MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) |
| 6 | Streaming | 🔴 | 20m | [agent.stream()](https://python.langchain.com/docs/how_to/streaming/) |
| 7 | GraphRAG | 🔴 (bu data için) | 4h+ | [Microsoft GraphRAG](https://github.com/microsoft/graphrag) |

---

## Referanslar (genel)

- [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/) — canonical
- [LangChain Retrievers concept](https://python.langchain.com/docs/concepts/retrievers/)
- [Pinecone — Chunking & Reranking series](https://www.pinecone.io/learn/series/rag/)
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Lewis et al., 2020 — RAG paper](https://arxiv.org/abs/2005.11401)
- [Es et al., 2023 — RAGAS paper](https://arxiv.org/abs/2309.15217)
- [Microsoft GraphRAG paper](https://arxiv.org/abs/2404.16130)
- [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
