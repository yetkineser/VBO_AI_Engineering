# Week 3 — Öz Değerlendirme: Neler Yaptık, Neler Öğrendik

Bu belge, her ödev gereksinimini adım adım inceler, ne uyguladığımızı gösterir, neler öğrendiğimizi yansıtır ve daha derine inmek için kaynaklara yönlendirir. Ardından ödevin ötesinde yaptığımız ekstra çalışmaları kapsar.

---

## Ödev Gereksinimleri

### 1. Önceden eğitilmiş bir model yükle

**Gereksinim:** `KeyedVectors` nesnesi döndüren `load_fasttext_model()` ve `load_glove_model()` yazın.

**Ne yaptık:**
- İki fonksiyonu da `src/embedding_utils.py` içinde uyguladık
- Tam FastText Türkçe modelini indirdik (`cc.tr.300.vec`, 4.2 GB, ~2M kelime)
- Bellek kullanımını ve yükleme süresini kontrol etmek için `limit` parametresi ekledik (varsayılan 200K)
- `load_glove_model()` dosyanın word2vec başlığı olup olmadığını otomatik algılar (GloVe dosyaları genellikle başlığı atlar)
- Dosya bulunamadığında düzgün hata mesajları ekledik

**Ne öğrendik:**
- Önceden eğitilmiş kelime vektörleri **word2vec metin formatını** takip eder: ilk satır `<kelime_sayısı> <boyut>`, sonra her satır `<kelime> <v1> ... <v300>`
- Gensim'in `KeyedVectors.load_word2vec_format()` standart yükleyicisidir — hem `.vec` hem `.vec.gz` dosyalarını ele alır
- 200K kelime yüklemek ~20 saniye ve ~600 MB RAM alır. Tam 2M dakikalar ve birkaç GB alır. `limit` parametresi pratik geliştirme için olmazsa olmaz
- GloVe ve FastText dosyaları metin formatında aynı görünür ama GloVe bazen başlık satırını atlar

**Anlayışınızı derinleştirin:**
- [Gensim — KeyedVectors dokümanları](https://radimrehurek.com/gensim/models/keyedvectors.html) — `load_word2vec_format()` parametreleri
- [FastText — Önceden eğitilmiş vektörler](https://fasttext.cc/docs/en/crawl-vectors.html) — 157 dil için modeller nereden indirilir
- [Bojanowski ve ark., 2017 — FastText makalesi](https://arxiv.org/abs/1607.04606) — FastText alt-kelime vektörlerini nasıl oluşturur

---

### 2. Kelime vektörü al

**Gereksinim:** 300 boyutlu vektörü veya OOV için `None` döndüren `get_word_vector(model, word)` yazın.

**Ne yaptık:**
- `src/embedding_utils.py` içinde uyguladık
- İki adımlı arama: önce ham kelimeyi dene, sonra normalize edilmiş sürümü dene
- Türkçeye duyarlı bir normalleştirme fonksiyonu (`normalise_word()`) oluşturduk:
  - `İ` → `i` (noktalı büyük I → noktalı küçük i)
  - `I` → `ı` (noktasız büyük I → noktasız küçük ı)
  - Noktalama ve boşluk temizliği
- OOV için istisna fırlatmak yerine `None` döndürür

**Ne öğrendik:**
- Türkçe harf dönüşümü zorludur: Python'un `casefold()` fonksiyonu `İ`'yi `i\u0307`'ye (i + birleştirici nokta) çevirir ki bu sözlük girişleriyle eşleşmez. Açık karakter değişimi kullanmak zorunda kaldık
- Eklemeli morfoloji nedeniyle OOV Türkçede çok yaygındır — `kitap` (book) mevcut olsa bile `kitaplarımızda` (in our books) bulunmayabilir
- `normalise_word()` fonksiyonu kritiktir: onsuz "Kedi" ve "kedi" farklı kelimeler olarak işlem görürdü

**Anlayışınızı derinleştirin:**
- [Unicode — Türkçe harf dönüşümü](https://www.unicode.org/versions/Unicode15.0.0/ch03.pdf#G33992) — Türkçe I/İ neden özel
- [Python dokümanları — str.casefold()](https://docs.python.org/3/library/stdtypes.html#str.casefold) — tam Unicode harf dönüşümü
- [Zeyrek — Türkçe morfolojik analizci](https://github.com/obulat/zeyrek) — aramadan önce Türkçe kelimeleri lemmatize etmek için

---

### 3. Kelime benzerliği (kosinüs)

**Gereksinim:** Kosinüs benzerliğini float olarak döndüren `word_similarity(model, word1, word2)` yazın.

**Ne yaptık:**
- `src/embedding_utils.py` içinde uyguladık
- Hesaplama için `sklearn.metrics.pairwise.cosine_similarity` kullanır
- OOV kelimeler için ödev talimatına uygun olarak `None` döndürür
- Sezgisel olarak doğru sonuçlarla 12 Türkçe kelime çiftiyle test ettik

**Sonuç öne çıkanları:**
- `kedi ↔ köpek: 0.79` (ikisi de hayvan — yüksek benzerlik)
- `kedi ↔ araba: 0.37` (ilgisiz — düşük benzerlik)
- `iyi ↔ kötü: 0.72` (zıt anlamlılar yüksek puan alır çünkü benzer bağlamlarda geçerler)

**Ne öğrendik:**
- Kosinüs benzerliği büyüklüğü değil yönü ölçer — bu önemlidir çünkü kelime sıklığı vektör normlarını etkiler
- Zıt anlamlılar (iyi/kötü, güzel/çirkin) şaşırtıcı derecede yüksek puan alır çünkü bağlamı paylaşırlar ("film iyiydi" vs "film kötüydü")
- Kosinüs benzerliği "eş anlamlılık" ile aynı şey değildir — eş anlamlıları ve zıt anlamlıları da içeren eş-oluşum kalıplarını yakalar

**Anlayışınızı derinleştirin:**
- [Machine Learning Mastery — Cosine Similarity](https://machinelearningmastery.com/cosine-similarity-for-nlp/) — çalışılmış örnekler
- [Hill ve ark., 2015 — SimLex-999](https://aclanthology.org/J15-4004/) — neden benzerlik ≠ ilişkisellik (zıt anlamlı sorununu açıklar)
- [Gensim — similarity()](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.similarity) — yerleşik alternatif

---

### 4. K-Means kümeleme

**Gereksinim:** `random_state=42` ve `n_init=10` ile `cluster_words(model, words, k=3)` yazın.

**Ne yaptık:**
- `src/embedding_utils.py` içinde uyguladık
- **Kümeleme öncesi vektörleri L2-normalize ettik** — bu ödev şartnamesinde yok ama doğru yapılacak şey, çünkü K-Means Öklid mesafesini kullanır ve biz kosinüs-benzeri davranış istiyoruz
- OOV kelimeleri log mesajıyla sessizce atlar
- Geçerli kelime kalmadığında veya `k > len(geçerli_kelimeler)` durumunda ödev talimatına uygun olarak `ValueError` fırlatır
- 24 kelimeyle test ettik (8 hayvan + 8 araç + 8 meyve) → **24'ünün 24'ü doğru kümelenmiş**

**Ne öğrendik:**
- `random_state=42` tekrarlanabilirlik için rastgele tohumu sabitler — onsuz her çalışmada farklı kümeler elde edersiniz
- `n_init=10` K-Means'i farklı başlangıç noktalarından 10 kez çalıştırır ve en iyi sonucu seçer. K-Means yerel optimumlara takılabilir, bu yüzden çoklu başlatma yardımcı olur
- L2 normalizasyonu Öklid mesafesini kosinüs mesafesine eşdeğer kılar: `||u - v||² = 2(1 - cos(u,v))` eğer `||u|| = ||v|| = 1`
- Kelime gömmeleri dikkat çekici derecede temiz kümeler oluşturur — sadece 3 kategoriyle 24/24 doğru

**Anlayışınızı derinleştirin:**
- [StatQuest — K-Means Clustering (video)](https://www.youtube.com/watch?v=4b5d3muPQmA) — 8 dakikalık görsel açıklama
- [scikit-learn — KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — API referansı
- [sklearn — Kümeleme Değerlendirmesi](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) — ARI, NMI, Purity

---

### 5. Komut satırı arayüzü

**Gereksinim:** Modeli yükleyen, vektörleri yazdıran, benzerlikleri hesaplayan, kelimeleri kümeleyen ve sonuçları kaydeden `main.py` yazın.

**Ne yaptık:**
- Tam `argparse` CLI ile `src/main.py` uyguladık
- Bayraklar: `--model`, `--model-type`, `--limit`, `--k`, `--analogy`, `--visualise`, `--all`
- Ortam değişkeni desteği: `EMBEDDING_MODEL_PATH`
- Çıktılar: `similarity.csv`, `clusters.csv`, `results.md`
- Biçimlendirilmiş tablolarla güzel konsol çıktısı

**Ne öğrendik:**
- `argparse` standart Python CLI kütüphanesidir — `--help`'i otomatik oluşturur, türleri doğrular ve varsayılanları ele alır
- Ortam değişkenleri (`os.environ.get("EMBEDDING_MODEL_PATH", varsayılan)`) yolları sabit kodlamadan yapılandırmanın temiz yoludur
- Sonuçları hem CSV (makine-okunabilir) hem Markdown (insan-okunabilir) olarak kaydetmek çıktıyı hem analiz hem dokümantasyon için yararlı kılar

**Anlayışınızı derinleştirin:**
- [Python dokümanları — argparse eğitimi](https://docs.python.org/3/howto/argparse.html) — adım adım
- [The Twelve-Factor App — III. Config](https://12factor.net/config) — yapılandırma neden ortam değişkenlerinde olmalı

---

## Ödevin Ötesinde Yaptıklarımız

### Ekstra 1: Kelime Analojileri (`--analogy` bayrağı)

**Nedir:** `vec(B) - vec(A) + vec(C)` hesaplayıp en yakın kelimeyi bulun. Gömme kalitesinin klasik testi.

**Ne uyguladık:**
- 7 kategoride 25 analoji dörtlüsü (cinsiyet, ülke-başkent, zıt anlamlılar, fiil zamanı, meslek-iş yeri, ülke-dil, ülke-para birimi)
- Girdi kelimelerini sonuçlardan filtreler
- Top-5 adaylarını puanlarıyla gösterir
- Başarı oranı: 8/24 Top-1 doğru, çoğu Top-5'te

**Ne öğrendik:**
- Cinsiyet analojileri Türkçede iyi çalışıyor (`baba→anne ✓`, `amca→teyze` Top-2'de)
- Ülke-başkent analojileri zayıf (`türkiye→ankara :: fransa→?` "paris" yerine "belçika" döndürür) çünkü Ankara baskın
- Zıt ve eş anlamlı analojiler iyi çalışıyor (`iyi→kötü :: güzel→çirkin ✓`)

**Bağlantılar:**
- [Mikolov ve ark., 2013 — Linguistic Regularities](https://aclanthology.org/N13-1090/)
- [The Illustrated Word2Vec — Analogy](https://jalammar.github.io/illustrated-word2vec/#analogy)

---

### Ekstra 2: t-SNE Görselleştirme (`--visualise` bayrağı)

**Nedir:** 300 boyutlu kelime vektörlerini t-SNE ile 2D'ye indirgeyin ve renk kodlu dağılım grafiği çizin.

**Ne uyguladık:**
- `main.py` içinde `sklearn.manifold.TSNE` ve `matplotlib` kullanan `demo_visualise()`
- Küme atamasına göre renk kodlu
- Her noktada kelime etiketleri
- `outputs/tsne_clusters.png` dosyasına kaydedilir

**Ne öğrendik:**
- t-SNE yerel komşuluk yapısını korur — 300D'de benzer olan kelimeler 2D'de yakın kalır
- Üç küme (hayvanlar, araçlar, meyveler) görsel olarak iyi ayrılmış
- t-SNE deterministik değildir ve kümeler arası mesafeler anlamlı değildir — sadece küme-içi yapı önemlidir

**Bağlantılar:**
- [Distill.pub — t-SNE'yi Etkili Kullanma](https://distill.pub/2016/misread-tsne/) — temel okuma
- [Google Embedding Projector](https://projector.tensorflow.org/) — interaktif tarayıcı aracı

---

### Ekstra 3: Tam Kıyaslama Değerlendirmesi (`evaluate.py`)

**Nedir:** Gömmelerimizi gerçek Türkçe NLP kıyaslamalarında uygun metriklerle değerlendirin.

**Ne uyguladık:**
- **AnlamVer** (500 Türkçe kelime çifti, insan puanlarıyla) → Spearman ρ
- **Türkçe Anlamsal Analojiler** (7742 soru, 7 kategori) → Top-1/5/MRR
- **Türkçe Sözdizimsel Analojiler** (206 soru) → Top-1/5/MRR
- **Genişletilmiş kümeleme** (90 kelime, 5 kategori) → ARI/NMI/Purity
- Hem `limit=200K` hem `limit=500K` ile test ettik

**Önemli sonuçlar:**
- Spearman ρ = 0.571 (orta — Türkçe statik gömmeler için beklenen)
- Anlamsal analoji Top-5 = %65.1, MRR = 0.471
- Sözdizimsel analoji Top-5 = %69.8
- Kümeleme ARI = 0.949 (mükemmel)

**Ne öğrendik:**
- Daha geniş sözlük (`limit=500K`) kapsamı artırır ama gürültü kelimelerinin doğru cevapla rekabet etmesi nedeniyle analoji doğruluğunu düşürebilir
- AnlamVer kıyaslaması birçok nadir/morfolojik olarak karmaşık kelime içerir — 200K sınırında çiftlerin %32'si OOV
- FastText modelimiz Türkçe statik gömmeler için beklenen aralıkta puan alıyor (literatür Spearman ρ için 0.45–0.60 raporluyor)

**Bağlantılar:**
- [AnlamVer Makalesi (Ercan & Yıldız, 2018)](https://aclanthology.org/C18-1323/)
- [Türkçe Statik Kelime Gömmelerinin Kapsamlı Analizi (2024)](https://arxiv.org/abs/2405.07778)

---

### Ekstra 4: Beş Model Karşılaştırması (`evaluate_advanced.py`)

**Nedir:** Statik, bağlamsal ve cümle-transformer gömmelerini aynı kıyaslamalarda karşılaştırın.

**Test edilen modeller:**
1. FastText cc.tr.300 (statik)
2. BERTurk (bağlamsal)
3. XLM-RoBERTa-base (bağlamsal, çok dilli)
4. Turkish BERT-NLI-STS (cümle-transformer)
5. Multilingual MiniLM (cümle-transformer)

**Önemli bulgular:**

| Bulgu | Ne öğretir |
|-------|-----------|
| FastText kelime benzerliğinde kazanır (ρ=0.571) | Statik gömmeler kararlı kelime-düzeyi vektörler verir |
| Ham BERT tek kelimeler için kötü (ρ=0.356) | Bağlamsal modeller bağlam ister — `[CLS] kedi [SEP]` yeterli değil |
| XLM-RoBERTa en kötü (ρ=0.014) | Çok dilli modeller dile özgü bilgiyi seyreltir |
| Turkish NLI-STS sözdizimsel analojide kazanır (%98.5) | Türkçe NLI ile ince ayar morfolojiyi derinden öğretir |
| Turkish NLI-STS eş anlamlılarda %95.7 | NLI eğitimi = iki şeyin aynı anlama geldiğini anlamak |
| Kapsam: Tüm transformerlarda %100, FastText'te %32-68 | Alt-kelime tokenizasyonu OOV'yi ortadan kaldırır |

**Merkezi ders:** Evrensel olarak "en iyi" gömme yoktur. FastText kelime-düzeyi görevlere hükmeder; cümle-transformerlar cümle-düzeyi görevlere hükmeder. Alt görev ihtiyacınıza göre seçin.

**Bağlantılar:**
- [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- [Ethayarajh, 2019 — How Contextual are Contextualized Representations?](https://aclanthology.org/D19-1006/)
- [Sentence-Transformers dokümanları](https://www.sbert.net/)
- [Bakarov, 2018 — Kelime Gömme Değerlendirme Yöntemleri Anketi](https://arxiv.org/abs/1801.09536)

---

### Ekstra 5: Kapsamlı Dokümantasyon

**Oluşturduklarımız:**
- `docs/HOMEWORK.md` + `_TR` — orijinal ödev kaydedildi ve çevrildi
- `docs/LEARNING_OBJECTIVES.md` + `_TR` — derlenmiş bağlantılarla 8 bölümlük çalışma rehberi
- `docs/EXTRA_SUGGESTIONS.md` + `_TR` — zorluk dereceleriyle 8 uzantı fikri
- `docs/EVALUATION_METRICS.md` + `_TR` — gömme değerlendirme metriklerinin tam rehberi
- `docs/RESULTS_ANALYSIS.md` + `_TR` — basit metrik örnekleriyle 5 model karşılaştırmasının yorumu
- `docs/SELF_EVALUATION.md` + `_TR` — bu belge
- `README.md` + `_TR` — mermaid diyagramlarıyla proje genel bakışı
- `outputs/model_comparison.md` — otomatik oluşturulan detaylı karşılaştırma raporu
- `outputs/evaluation_report.md` — otomatik oluşturulan kıyaslama sonuçları

---

## Özet Puan Kartı

| Ödev Gereksinimleri | Durum | Kalite |
|---------------------|-------|--------|
| `load_fasttext_model()` | Tamamlandı | .vec ve .gz destekler, limit parametresi, hata mesajları |
| `load_glove_model()` | Tamamlandı | Başlığı otomatik algılar, no_header geri dönüşü |
| `get_word_vector()` | Tamamlandı | Türkçeye duyarlı normalleştirme, zarif OOV |
| `word_similarity()` | Tamamlandı | sklearn ile kosinüs, OOV için None |
| `cluster_words()` | Tamamlandı | L2 normalizasyonu, random_state=42, n_init=10, hata durumları için ValueError |
| `main.py` CLI | Tamamlandı | argparse, ortam değişkenleri, CSV + MD çıktı |
| `requirements.txt` | Tamamlandı | Minimal bağımlılıklar |
| `.gitignore` | Tamamlandı | data/*.vec*, outputs/ hariç tutar |
| `README.md` | Tamamlandı | Mermaid diyagramları, API referansı, kaynaklar |

| Ekstra | Durum |
|--------|-------|
| Kelime analojileri (25 dörtlü, 7 kategori) | Tamamlandı |
| t-SNE görselleştirme | Tamamlandı |
| AnlamVer kıyaslaması (500 çift, Spearman ρ) | Tamamlandı |
| Türkçe analoji kıyaslaması (7742 + 206 soru) | Tamamlandı |
| Genişletilmiş kümeleme (90 kelime, 5 kategori, ARI/NMI) | Tamamlandı |
| 5 model karşılaştırması (FastText vs BERT vs sentence-transformers) | Tamamlandı |
| Birden fazla sözlük boyutunda değerlendirme (200K vs 500K) | Tamamlandı |
| 16 dokümantasyon dosyası (EN + TR) | Tamamlandı |
