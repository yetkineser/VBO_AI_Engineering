# Week 3 — Ek Öneriler: Başka Neler Yapılabilirdi?

Bu belge, **ödevin ötesinde** hem eğitici hem de pratik uzantıları kapsar. Her öneri zorluk derecesi, neyi öğrettiği ve başlangıç bağlantısı içerir.

---

## 1. Kelime Analojileri (kral - erkek + kadın = kraliçe)

**Zorluk:** Kolay | **Ne öğretir:** Vektör aritmetiği anlamsal ilişkileri yakalar

Kelime gömmeleri, ilişkileri vektör uzayında yönler olarak kodlar. Klasik test:

```
vec("kral") - vec("erkek") + vec("kadın") ≈ vec("kraliçe")
```

Bu çalışır çünkü "cinsiyet yönü" sözlükte tutarlıdır. Ülke-başkent, fiil zamanı ve sıfat derecesi analojilerini de aynı şekilde test edebilirsiniz.

**Bu projede zaten uygulandı** — `--analogy` bayrağıyla.

**Bağlantılar:**
- [The Illustrated Word2Vec — Analogy Bölümü](https://jalammar.github.io/illustrated-word2vec/#analogy)
- [Gensim — most_similar (positive/negative ile)](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar)

---

## 2. t-SNE / PCA Görselleştirme

**Zorluk:** Kolay | **Ne öğretir:** Yüksek boyutlu yapıyı görünür kılar

300 boyutlu kelime vektörlerini PCA veya t-SNE ile 2D'ye indirgeyin ve bir dağılım grafiğinde çizin. Aynı anlamsal kategoriden kelimeler görünür kümeler oluşturmalıdır.

PCA deterministik ve hızlıdır ama yalnızca doğrusal varyansı yakalar. t-SNE yerel komşuluk yapısını daha iyi korur ama deterministik değildir ve daha yavaştır.

**Bu projede zaten uygulandı** — `--visualise` bayrağıyla.

**Bağlantılar:**
- [scikit-learn — t-SNE Kullanıcı Kılavuzu](https://scikit-learn.org/stable/modules/manifold.html#t-sne)
- [Distill.pub — t-SNE'yi Etkili Kullanma](https://distill.pub/2016/misread-tsne/) — t-SNE yorumlama üzerine temel okuma
- [Google Embedding Projector](https://projector.tensorflow.org/) — tarayıcıda interaktif keşif

---

## 3. Türkçe Kelime Gömmelerinde Önyargı Tespiti

**Zorluk:** Orta | **Ne öğretir:** Adillik, toplumsal önyargı, sorumlu yapay zeka

Kelime gömmeleri internet metninden öğrenir ve bu metinler kalıp yargılar içerir. Hangi meslek kelimelerinin (doktor, hemşire, mühendis, öğretmen) "erkek"e mi yoksa "kadın"a mı daha yakın olduğunu kontrol ederek cinsiyet önyargısını ölçebilirsiniz.

Bu etik açıdan önemlidir ve uygulaması şaşırtıcı derecede kolaydır — hassas kelime çiftlerine uygulanan kosinüs benzerliğidir.

**Bağlantılar:**
- [Bolukbasi ve ark., 2016 — Man is to Computer Programmer as Woman is to Homemaker](https://arxiv.org/abs/1607.06520) — temel önyargı makalesi
- [Caliskan ve ark., 2017 — Semantics derived automatically from language corpora contain human-like biases](https://arxiv.org/abs/1608.07187)
- [Google AI Blog — Kelime Gömmelerinde Cinsiyet Önyargısını Azaltma](https://ai.googleblog.com/2016/07/reducing-gender-bias-in-word-embeddings.html)

---

## 4. FastText vs GloVe vs Word2Vec Karşılaştırma

**Zorluk:** Orta | **Ne öğretir:** Model seçimi, Türkçe morfolojisinin etkisi

İki veya üç farklı gömme modeli yükleyin, aynı benzerlik ve kümeleme görevlerini çalıştırın ve bir karşılaştırma tablosu oluşturun. Temel sorular:

- Türkçe kelime çiftlerinde hangi model sezginize daha çok uyar?
- OOV oranları nasıl karşılaştırılır? (Alt-kelime desteği sayesinde FastText kazanmalı)
- Küme atamaları farklı mı?

**Bağlantılar:**
- [FastText — 157 Dil İçin Önceden Eğitilmiş Vektörler](https://fasttext.cc/docs/en/crawl-vectors.html)
- [Gensim Word2Vec Turkish](https://radimrehurek.com/gensim/models/word2vec.html)

---

## 5. Ortalama Havuzlama ile Cümle Gömmeleri

**Zorluk:** Orta | **Ne öğretir:** Kelime vektörlerinden cümle vektörlerine

Bir cümledeki tüm kelime vektörlerinin ortalamasını alarak cümle düzeyinde bir gömme oluşturun. Sonra cümleler arasında kosinüs benzerliği hesaplayın. Bu kaba bir yöntemdir ama birçok görev için şaşırtıcı derecede etkilidir.

```python
def sentence_embedding(model, sentence: str) -> np.ndarray:
    words = sentence.lower().split()
    vectors = [model[w] for w in words if w in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
```

Aradaki farkı görmek için Week 2'den `sentence-transformers` ile karşılaştırın.

**Bağlantılar:**
- [Arora ve ark., 2017 — A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx) — SIF ağırlıklandırma ortalama havuzlamayı iyileştirir
- [Sentence-Transformers](https://www.sbert.net/) — modern alternatif

---

## 6. Sıfırdan Kendi Gömmelerinizi Eğitin

**Zorluk:** Zor | **Ne öğretir:** Gömmeler gerçekte nasıl oluşturulur

Önceden eğitilmiş vektörleri yüklemek yerine, bir Türkçe külliyat üzerinde (ör. Türkçe Vikipedi dökümü veya haber veri seti) Word2Vec veya FastText eğitin. Bu şunları öğretir:

- Külliyat boyutunun kaliteyi nasıl etkilediği
- Pencere boyutu ve vektör boyutluluğunun sonuçları nasıl değiştirdiği
- Neden önceden eğitilmiş modeller genellikle daha iyi — nadiren yeterli veriniz olur

**Bağlantılar:**
- [Gensim — Word2Vec Eğitim Rehberi](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)
- [FastText — Kendi Verinizle Eğitim](https://fasttext.cc/docs/en/unsupervised-tutorial.html)
- [Türkçe Vikipedi Dökümleri](https://dumps.wikimedia.org/trwiki/) — eğitim için ham metin

---

## 7. Gömme Tabanlı Metin Sınıflandırma

**Zorluk:** Zor | **Ne öğretir:** Gömmeleri alt görevlere bağlama

Kelime gömmelerini metin sınıflandırıcı için özellik olarak kullanın:

1. Her belgeyi kelime vektörlerinin ortalaması (veya TF-IDF ağırlıklı ortalaması) olarak kodlayın.
2. Üstüne Lojistik Regresyon veya SVM eğitin.
3. Week 2'deki TF-IDF temeli ile karşılaştırın.

Bu, Week 2'yi (duygu analizi) Week 3 ile (gömmeler) birleştirir ve bağlamsal gömmelerin (BERT vb.) neden statik kelime vektörlerinden daha iyi olduğunu gösterir.

**Bağlantılar:**
- [scikit-learn — Metin Sınıflandırma Boru Hattı](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- Bu depodaki Week 2 projesi

---

## 8. En Yakın Komşular & Anlamsal Arama

**Zorluk:** Orta | **Ne öğretir:** Gömmelerin pratik uygulaması

Basit bir anlamsal arama motoru kurun: bir sorgu kelimesi verildiğinde en benzer N kelimeyi bulun. Sonra vektörleri ortalayarak çok kelimeli sorgulara genişletin. Bu, Pinecone, Weaviate ve FAISS gibi vektör veritabanlarının arkasındaki temel fikirdir.

**Bağlantılar:**
- [Gensim — most_similar](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar)
- [FAISS — Facebook AI Benzerlik Araması](https://github.com/facebookresearch/faiss)
- [Pinecone — Vektör Veritabanı Nedir?](https://www.pinecone.io/learn/vector-database/)

---

## Özet

| Öneri | Zorluk | Uygulandı mı? |
|---|---|---|
| Kelime Analojileri | Kolay | Evet (`--analogy`) |
| t-SNE Görselleştirme | Kolay | Evet (`--visualise`) |
| Önyargı Tespiti | Orta | Hayır |
| FastText vs GloVe | Orta | Hayır (altyapı hazır) |
| Cümle Gömmeleri | Orta | Hayır |
| Sıfırdan Eğitim | Zor | Hayır |
| Gömme Sınıflandırma | Zor | Hayır |
| Anlamsal Arama | Orta | Hayır |
