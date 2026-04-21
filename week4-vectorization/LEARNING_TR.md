# Hafta 4 Öğrenme Rehberi — Ne Çalışmalı ve Neden

Bu sayfa Hafta 4 ödevinin arkasındaki kavramları açıklar ve her biri için okuma linkleri verir. Bölümleri sırayla oku — her biri bir öncekinin üzerine inşa eder.

---

## 1. Yapay Zeka Sistemleri İçin Doküman Depolama

Herhangi bir AI sistemi verileriniz hakkında soruları cevaplayabilmek için önce o verileri **aranabilir** bir şekilde **depolamak** zorundadır. Üç tamamlayıcı yaklaşım var ve üretim sistemleri üçünü birlikte kullanır.

### Okuma Linkleri
- [MongoDB Getting Started Tutorial](https://www.mongodb.com/docs/manual/tutorial/getting-started/) — resmi 30 dakikalık rehber
- [What is NoSQL? (MongoDB)](https://www.mongodb.com/nosql-explained) — esnek şemaların neden önemli olduğu

### Ana Çıkarım
MongoDB **yapılandırılmış metadata**'yı (yazar, boyut, tarih, etiketler) esnek JSON dokümanları olarak saklar. SQL gibi önceden sütun tanımlamanız gerekmez — her dosya tipinin farklı özellikleri olduğu durumlar için ideal.

---

## 2. Tam Metin Arama ve Ters İndeksler

"Transfer learning" gibi bir sorgu yazdığınızda, Elasticsearch o kelimeleri içeren tüm dokümanları **ters indeks** kullanarak bulur — her kelimeden onu içeren dokümanlara bir harita. Bu, Hafta 2'deki TF-IDF ile aynı fikir, ama hız için optimize edilmiş hali.

### Okuma Linkleri
- [Elasticsearch: Documents and Indices](https://www.elastic.co/guide/en/elasticsearch/reference/current/documents-indices.html) — ters indeks gerçekte nedir
- [BM25: Elasticsearch'ün Arkasındaki Algoritma](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) — BM25, geliştirilmiş bir TF-IDF; ES'nin neden ham TF-IDF yerine bunu kullandığını anla
- [Fireship: Elasticsearch in 100 Seconds](https://www.youtube.com/watch?v=ZP0NmfyfsoM) — en hızlı genel bakış (video)

### Ana Çıkarım
Anahtar kelime araması **kesin**dir — tam kelimeleri ve cümleleri bulur. Ama kullanıcının kelimeleri dokümanın kelimelerinden farklı olduğunda başarısız olur ("bildirim sistemi" vs "push tahsis pipeline'ı").

---

## 3. Yoğun Vektör Gömmeleri ve Semantik Arama

Gömmeleri (embeddings) zaten Hafta 2'den (transformer embeddings) ve Hafta 3'ten (FastText/GloVe) biliyorsun. Bu hafta, bunları hesaplayıp atmak yerine, anlam bazlı arama yapabilmek için **saklıyorsun**.

### Okuma Linkleri
- [What is Similarity Search? (Pinecone)](https://www.pinecone.io/learn/what-is-similarity-search/) — vektör aramaya en iyi görsel giriş
- [Sentence Transformers Quickstart](https://www.sbert.net/docs/quickstart.html) — embedding üretmek için kullanacağın kütüphane
- [Elasticsearch kNN Search](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html) — Elasticsearch'te yoğun vektörleri nasıl saklayıp sorgularsın

### Ana Çıkarım
Vektör araması içeriği tam kelimelerle değil **anlamla** bulur. Eş anlamlıları, diller arası sorguları ve soruları doğal olarak işler. Dezavantaj: tam cümle eşleştirmesi yapamaz.

---

## 4. Üçünü Birleştirmek (Hibrit Arama)

Üretim RAG sistemleri tek bir yaklaşım seçmez — üçünü birleştirir:

1. Metadata ile **filtrele** (MongoDB): "sadece 2020 sonrası yayınlanan kitaplar"
2. Anahtar kelime uygunluğu ile **sırala** (Elasticsearch BM25): "hangilerinde transfer learning var?"
3. Semantik benzerlik ile **tekrar sırala** (vektör arama): "sorgunun anlamına en yakın hangileri?"

### Okuma Linkleri
- [Hybrid Search Explained (Qdrant)](https://qdrant.tech/articles/hybrid-search/) — anahtar kelime ve vektör sonuçlarını nasıl birleştirirsin
- [Reciprocal Rank Fusion](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) — Elasticsearch'ün keyword + vektör skorlarını birleştirmek için kullandığı algoritma

### Ana Çıkarım
Her sistem diğerinin zayıflığını kapatır. Metadata filtreleme kapsamı daraltır, anahtar kelime araması tam eşleşmeleri bulur, vektör araması semantik yakın eşleşmeleri yakalar.

---

## 5. Doküman Ayrıştırma (Parsing)

Herhangi bir şey saklamadan önce, dosyalardan metin çıkarman gerekiyor. Farklı formatlar farklı ayrıştırıcılar gerektirir.

### Okuma Linkleri
- [PyMuPDF (fitz) Tutorial](https://pymupdf.readthedocs.io/en/latest/tutorial.html) — kullandığımız PDF ayrıştırıcı (hızlı, güvenilir)
- [EbookLib Documentation](https://docs.sourcefabric.org/projects/ebooklib/en/latest/) — EPUB dosyaları için

### Ana Çıkarım
PDF çıkarma kusursuz değildir — taranmış PDF'ler OCR gerektirir, bazı PDF'lerin karakter kodlaması bozuktur. Güvenmeden önce çıkarılan metni her zaman kontrol et.

---

## 6. Bu Nereye Gidiyor: RAG

Bu pipeline, Retrieval-Augmented Generation'ın **erişim** (retrieval) yarısıdır. Hafta 5–6'da bu depolama katmanını bir LLM'e bağlayarak dokümanlarına dayalı cevaplar **üretmeyi** öğreneceksin.

### Okuma Linkleri
- [RAG from Scratch (LangChain YouTube)](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) — erişimin üretimi nasıl beslediğini gösteren video serisi
- [LangChain Document Loaders](https://python.langchain.com/docs/how_to/#document-loaders) — LangChain'in aynı dosya yükleme problemini bir framework ile nasıl çözdüğü

### Ana Çıkarım
Bir erişim katmanı olmadan LLM'ler sadece eğitim verilerinden cevap verebilir. Bu katman ile **senin** verilerinden cevap verebilir — kurumsal uygulamalarda RAG'ı bu kadar güçlü yapan da budur.

---

## Bootcamp Yolculuğunla Bağlantı

```
Hafta 1: Metin İŞLE    →  tokenize, encode, sınıflandır (IMDb'de BERT)
Hafta 2: Metin KODLA   →  integer → one-hot → TF-IDF → transformer → zero-shot
Hafta 3: Metin GÖMSÜNLE →  FastText/GloVe, kosinüs benzerliği, kümeleme
Hafta 4: Metin SAKLA   →  MongoDB + Elasticsearch + vektörler     ← BURADASIN
Hafta 5: Metin ERİŞ    →  LangChain retrieval chain'leri
Hafta 6: ÜRET          →  RAG (erişim + LLM = dayanaklı cevaplar)
Hafta 7: ORKESTRA ET   →  Agent'lar ve LangGraph
```

Her hafta bir öncekinin üzerine inşa eder. Hafta 2–3'te öğrendiğin gömmeler bu hafta sakladığın vektörler olur. Bu hafta inşa ettiğin depolama katmanı Hafta 5–6'nın erişim backend'i olur.

---

## 7. Bu Pipeline Nasıl Geliştirilir?

Alakasız bir sorgu ("En iyi pizza tarifi nedir?") çalıştırdığımızda pipeline'ın dört zayıflığı ortaya çıktı. Her biri üretim kalitesinde bir erişim sistemine doğru bir adım.

### 7a. Chunking — Dokümanları Küçük Parçalara Böl

Şu an her dokümanın sadece ilk 512 karakterini embed ediyoruz. 300 sayfalık bir kitap ilk paragrafıyla temsil ediliyor. Çözüm: dokümanları örtüşen parçalara (500–1000 token) böl ve her parçayı ayrı embed et.

**Okuma Linkleri**:
- [Chunking Strategies for LLM Applications (Pinecone)](https://www.pinecone.io/learn/chunking-strategies/) — farklı yaklaşımların görsel rehberi
- [LangChain Text Splitters](https://python.langchain.com/docs/how_to/#text-splitters) — pratik uygulamalar
- [Greg Kamradt: 5 Levels of Text Splitting (YouTube)](https://www.youtube.com/watch?v=8OJC21T2SL4) — karakter bölmeden semantik bölmeye 20 dakikada

### 7b. Hibrit Arama — Kelime + Vektör Sonuçlarını Birleştir

Üç ayrı arama yerine, BM25 ve vektör sonuçlarını Reciprocal Rank Fusion (RRF) ile tek bir sıralamada birleştir.

**Okuma Linkleri**:
- [Reciprocal Rank Fusion (Elasticsearch)](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) — resmi rehber
- [Hybrid Search Explained (Qdrant)](https://qdrant.tech/articles/hybrid-search/) — diyagramlarla kavram açıklayıcı

### 7c. Skor Eşiği — Sistemin "Bilmiyorum" Demesini Sağla

Pipeline'ımız her zaman sonuç döndürüyor, alakasız olsa bile. Minimum benzerlik skoru (örneğin 0.65) koymak, alakasız dokümanların LLM'e ulaşmasını engeller — halüsinasyona karşı ilk savunma hattı.

**Okuma Linkleri**:
- [Reducing Hallucination in RAG (LlamaIndex)](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) — üretim RAG rehberi
- [RAGAS: Evaluating RAG Pipelines](https://docs.ragas.io/en/latest/) — erişim kalitesini ölçme

### 7d. Daha Güçlü Embedding Modelleri

Modelimiz (`MiniLM`, 384 boyut) hızlı ama küçük. Daha büyük modeller anlamı daha iyi yakalar, özellikle teknik içerik için. Seçenekleri karşılaştırmak için MTEB liderlik tablosuna bak.

**Okuma Linkleri**:
- [MTEB Leaderboard (Hugging Face)](https://huggingface.co/spaces/mteb/leaderboard) — 100+ embedding modelini karşılaştıran benchmark
- [Choosing an Embedding Model for RAG (Pinecone)](https://www.pinecone.io/learn/series/rag/embedding-models-rag/) — pratik seçim rehberi

---

*VBO AI & LLM Bootcamp, Hafta 4 (Nisan 2026) için yazılmıştır.*
