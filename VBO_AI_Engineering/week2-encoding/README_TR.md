# Hafta 2: Metin Kodlama ve Duygu Analizi

Bu proje, ham metni makine ogrenmesi modellerinin anlayabilecegi sayisal temsillere donusturme problemini ele alir ve bu temsiller uzerine bir duygu siniflandirici insa eder. Temel yontemlerle baslar (integer encoding, one-hot, TF-IDF) ve kademeli olarak modern yaklasimlara ilerler (transformer embedding, zero-shot siniflandirma, Turkce BERT fine-tuning).

---

## Neler Ogreneceksiniz

| Konu | Beklenen |
|------|----------|
| **Metinden sayiya** | ML modelleri vektor/matris girdisi gerektirir. Kelimeleri ve cumleleri cesitli kodlama yontemleriyle sayisal forma donustureceksiniz. |
| **Integer encoding** | Her kelimeye sozlukten benzersiz bir indeks atama: ornegin `"iyi" -> 42`. Basit, ama siralama anlamsiz. |
| **One-hot encoding** | Sozlukteki her kelime icin ikili bir boyut olusturma. Kelime varsa 1, yoksa 0. |
| **TF-IDF** | Kelimeleri, bir dokumanda ne kadar ayirt edici olduklarina gore agirliklandirma. Yaygin kelimeler dusuk puan alir. |
| **Duygu analizi** | Bir metnin duygusal tonunu tahmin eden siniflandirma gorevi (ornegin pozitif / negatif). |
| **Klasik ML modelleri** | Naive Bayes, Logistic Regression, SVM, XGBoost, LightGBM — temel odev icin derin ogrenme gerekmiyor. |
| **Transformer embedding** | Onceden egitilmis bir dil modeli kullanarak cumleleri yogun, anlam iceren vektorlere kodlama. |
| **Zero-shot siniflandirma** | Onceden egitilmis bir NLI modeli kullanarak hicbir egitim verisi olmadan duygu tahmini. |
| **Fine-tuning** | Onceden egitilmis bir Turkce BERT modelini bizim duygu analizi gorevimize uyarlama. |

---

## Temel Kavramlar

### 1. Metin Vektorlestirme

Bilgisayarlar ham metni dogrudan isleyemez. Herhangi bir modele metin beslemeden once, onu bir sayi dizisine veya vektore donusturmeniz gerekir. Bu adim genellikle **ozellik cikarimi** veya **metin vektorlestirme** olarak bilinir.

### 2. Integer Encoding (Tamsayi Kodlama)

Tum egitim metinlerinden bir **sozluk** olusturursunuz ve her benzersiz kelimeye bir tamsayi indeks atarsiniz (0'dan |V|-1'e kadar). Bir cumle bu indekslerin dizisine donusur.

**Onemli not:** Sayisal siralama keyfidir — indeks 42, indeks 7'den anlamli bir sekilde "buyuk" degildir. Bu yuzden integer encoding tek basina klasik ML modellerine girdi olarak nadiren kullanilir. Daha cok one-hot veya embedding tabanli temsillere gecis icin bir basamak gorevi gorur.

### 3. One-Hot ve Bag of Words

- **Etiketler icin:** 3 sinifta her ornek `[1,0,0]`, `[0,1,0]` veya `[0,0,1]` olur.
- **Metin icin (Bag of Words):** Sozluk kadar genis bir vektor olusturursunuz. Cumledeki her kelime icin ilgili pozisyon 1'e (veya frekans sayisina) ayarlanir. Bu **yuksek boyutlu, seyrek** vektorler uretir. scikit-learn'de `CountVectorizer(binary=True)` bu yaklasimi uygular.

### 4. TF-IDF

TF-IDF, ham kelime sayilarinin bir adim otesine gecer. Bircok dokumanda gecen kelimeleri ("bir", "ve", "bu" gibi) dusuk agirliklandirir, belirli bir dokumana ozgu kelimeleri ise yuksek agirliklandirir. scikit-learn'un `TfidfVectorizer`'i bunu otomatik yapar.

### 5. Duygu Siniflandirici Olusturma

Tipik is akisi:

1. Etiketli metin verisi hazirlayin (ornegin pozitif / negatif yorumlar).
2. Metni yukaridaki yontemlerden biriyle sayisal ozelliklere donusturun.
3. Egitim ve test setlerine bolin.
4. Bir siniflandirici egitin ve metriklerle degerlendirin (accuracy, F1, AUC-ROC).

---

## Modern Yaklasimlar (Odevin Otesinde)

Bu proje, temel odev gereksinimlerinin otesinde bes ek yaklasim uygular. Her biri veri gereksinimi, islem maliyeti ve performans arasinda farkli bir denge gosterir.

### 1. Transformer Embedding (`--transformer`)

CountVectorizer ve TF-IDF kelimeleri bagimsiz isler — "film harika" ile "harika film" arasindaki farki anlayamazlar. **Transformer embedding**'ler bu sorunu, tum cumleyi kelime sirasi, baglam ve anlamsal iliskileri yakalayan yogun bir vektore kodlayarak cozer.

Bu proje, alan ve dil uyumunun etkisini gostermek icin uc transformer modeli karsilastirir:

| Model | Boyut | Dil | Alan | AUC-ROC |
|-------|-------|-----|------|---------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Cok dilli (50+) | Genel | 0.9288 |
| `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` | 768 | Turkce | NLI / STS | 0.9441 |
| `Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0` | 768 | Turkce + Cok dilli | E-ticaret | **0.9568** |

Trendyol modeli en iyi sonucu verdi cunku veri setimiz urun yorumlarindan olusuyor — modelin fine-tune edildigi alanla ayni. Bu, **alan uyumunun model boyutundan daha onemli oldugunu** gosteriyor.

**Ne zaman kullanilmali:** Etiketli veriniz var ve sifirdan derin ogrenme modeli egitmeden onemli olcude daha iyi ozellikler istiyorsunuz.

### 2. Ollama ile Lokal Embedding (`--ollama`)

Ollama, `nomic-embed-text` embedding modelini tamamen yerel makinenizde calistirir. Hicbir veri harici bir API'ye gonderilmez — tam gizlilik saglayan tek secenektir.

Odun: performans. `nomic-embed-text` Ingilizce agirlikli bir modeldir ve Turkce anlayisi sinirlidir (Turkce veri setimizde AUC = 0.83). Ingilizce metin veya gizliligin oncelikli oldugu RAG uygulamalari icin saglam bir secimdir.

**Ne zaman kullanilmali:** Gizlilik kritik, Ingilizce metinle calisiyorsunuz veya internet erisimi olmadan lokal bir embedding cozumune ihtiyaciniz var.

### 3. Zero-Shot Siniflandirma (`--zero-shot`)

Zero-shot siniflandirma **hicbir egitim verisi gerektirmez**. Onceden egitilmis bir Dogal Dil Cikarimi (NLI) modelini kullanarak bir cumlenin verilen bir etiket hipoteziyle eslestip eslesmedegini degerlendirir (ornegin "Bu metin pozitif bir duygu ifade ediyor").

Bu projede `xlm-roberta-large-xnli` kullanilir — Turkce dahil 100'den fazla dili destekleyen cok dilli bir model.

**Ne zaman kullanilmali:** Etiketli veriniz yok, yeni bir alan kesfediyorsunuz veya veri toplama yatirimina girismeden once hizli bir temel cizgi gerekiyor.

### 4. BERTurk Fine-Tuning (`--finetune`)

Fine-tuning, onceden egitilmis bir Turkce BERT modelini (`dbmdz/bert-base-turkish-cased`) bizim duygu analizi gorevimize uyarlar. Model, buyuk bir korpus uzerindeki on-egitiminden Turkce dilbilgisi, kelime anlamlari ve baglami zaten anlar. Bir siniflandirma katmani ekler ve etiketli verimiz uzerinde birkac epoch egitiriz.

En guclu yaklasimdir ancak daha fazla islem gucu (GPU onerilir) ve etiketli veri gerektirir.

**Ne zaman kullanilmali:** Etiketli veriniz var ve mumkun olan en yuksek dogrulugu istiyorsunuz. Metin siniflandirma gorevleri icin endustri standardidir.

### Yaklasim Karsilastirmasi

| Yaklasim | Egitim Gerekli mi? | Veri Gerekli mi? | Guc |
|----------|--------------------|--------------------|-----|
| CountVectorizer + ML | Evet | Evet | Dusuk |
| TF-IDF + ML | Evet | Evet | Orta |
| Transformer Embedding + ML (genel) | Hayir (sadece encode) + Evet (ML) | Evet | Orta-Yuksek |
| Transformer Embedding + ML (alan-spesifik) | Hayir (sadece encode) + Evet (ML) | Evet | **Yuksek** |
| Ollama Lokal Embedding + ML | Hayir (sadece encode) + Evet (ML) | Evet | Dusuk (Turkce icin) |
| Zero-Shot | Hayir | Hayir | Orta-Yuksek |
| Fine-Tuning BERTurk | Evet (GPU onerilir) | Evet | En Yuksek |

---

## Kullanim

```bash
# Temel odev — sadece klasik ML
python src/sentiment_analysis.py

# Zeyrek morfolojik analiz ile
python src/sentiment_analysis.py --zeyrek

# Transformer embedding ekle (MiniLM + turkish-BERT-nli + Trendyol-ecomm)
python src/sentiment_analysis.py --transformer

# Lokal Ollama embedding ekle (gerekli: ollama pull nomic-embed-text)
python src/sentiment_analysis.py --ollama

# Zero-shot siniflandirma ekle (egitim gerekmiyor)
python src/sentiment_analysis.py --zero-shot

# BERTurk fine-tuning ekle
python src/sentiment_analysis.py --finetune

# Her seyi calistir
python src/sentiment_analysis.py --all

# Ozel CSV veri seti ile capraz dogrulama
python src/sentiment_analysis.py --data data/train.csv --test-data data/test.csv --transformer

# Tum embedding yontemleri ile tam karsilastirma
python src/sentiment_analysis.py --data data/turkish_sentiment_binary_5k.csv \
    --test-data data/external_test_1k.csv --transformer --ollama
```

### Cikti Dosyalari

Tum ciktilar `outputs/` klasorune kaydedilir:

| Dosya | Aciklama |
|-------|----------|
| `run_log.txt` | Her adimin zaman damgali tam logu |
| `comparison_results.csv` | Tum model sonuclari tablo formatinda |
| `analysis.md` | Sonuclarin otomatik olusturulan yorumu |
| `analysis_cross_dataset.md` | Capraz veri seti dogrulama sonuclari ve detayli yorum |

---

## Onerilen Uygulama Adimlari

1. Kucuk bir duygu veri seti secin veya olusturun (en az iki sinif: pozitif/negatif). Script, yedek olarak 40 yerlesik Turkce cumle icerir.
2. **Integer encoding:** Kavrami `CountVectorizer` veya manuel sozluk eslemesi ile gosterin.
3. **One-hot temsili:** Metin ozellikleri icin `CountVectorizer(binary=True)` kullanin.
4. **TF-IDF temsili:** Agirlikli ozellikler icin `TfidfVectorizer` kullanin.
5. **Model egitimi:** `MultinomialNB`, `LogisticRegression`, `LinearSVC`, `XGBClassifier`, `LGBMClassifier` egitin.
6. **Karsilastirma:** Tum vektorlestirici-model kombinasyonlarini F1 ve AUC-ROC ile degerlendirin.
7. *(Opsiyonel)* Daha derin bir karsilastirma icin transformer embedding, zero-shot veya fine-tuning ekleyin.

---

## Degerlendirme Metrikleri

Bu proje her model icin hem **F1 (macro)** hem de **AUC-ROC** raporlar:

- **F1** sabit bir siniflandirma esigine (genellikle 0.5) baglidir. Modelin o belirli esikte ne kadar iyi performans gosterdigini soyler.
- **AUC-ROC** esikten bagimsizdir. Modelin pozitif ornekleri negatif orneklerin uzerine siralama yetenegini tum olasi esikler boyunca olcer. Yuksek AUC ama dusuk F1, modelin faydali kaliplari ogrendigini ancak esik kalibrasyonuna ihtiyac duydugunu gosterir.

Her zaman ikisini de raporlayin. Model kalitesi hakkinda farkli hikayeler anlatirlar.

---

## Kaynaklar

### Resmi Dokumantasyon ve Ogreticiler

- [scikit-learn: Metin verisiyle calisma](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) — eksiksiz metin siniflandirma pipeline'i
- [scikit-learn: CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn: OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [scikit-learn: LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

### Duygu Analizi ve NLP Temelleri

- [NLTK Book — Bolum 6: Metin Siniflandirma](https://www.nltk.org/book/ch06.html) — klasik NLP perspektifi
- [Kaggle Learn: NLP Kursu](https://www.kaggle.com/learn/natural-language-processing) — kisa, pratik moduller
- Jurafsky & Martin, *Speech and Language Processing* — [online taslak](https://web.stanford.edu/~jurafsky/slp3/) — ozellikle metin siniflandirma ve logistic regression bolumleri

### Transformer Embedding

- [Sentence-Transformers Dokumantasyonu](https://www.sbert.net/) — onceden egitilmis modellerle cumle embedding'leri kullanimi
- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — transformer mimarisine gorsel rehber
- [Jay Alammar: The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — kelime vektorlerinden cumle vektorlerine
- [HuggingFace: paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — genel cok dilli sentence transformer
- [HuggingFace: bert-base-turkish-cased-mean-nli-stsb-tr](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr) — Turkce'ye ozel sentence transformer (NLI + STS)
- [HuggingFace: Trendyol E-Ticaret Embedding](https://huggingface.co/Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0) — Trendyol'un alan-spesifik e-ticaret embedding modeli (deneyimizdeki en iyi performans)

### Ollama ile Lokal Embedding

- [Ollama](https://ollama.com/) — LLM ve embedding modellerini lokalde calistirin
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text) — lokal kullanim icin 768 boyutlu embedding modeli
- [Ollama Embedding API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings) — `/api/embed` endpoint dokumantasyonu

### Zero-Shot Siniflandirma

- [HuggingFace: Zero-Shot Siniflandirma Rehberi](https://huggingface.co/tasks/zero-shot-classification) — kavram ve kullanim ornekleri
- [Yin et al., 2019 — Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — zero-shot metin siniflandirma temel makalesi
- [HuggingFace: xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli) — bu projede kullanilan cok dilli zero-shot model

### Fine-Tuning ve Transfer Ogrenme

- [HuggingFace: Metin Siniflandirma Rehberi](https://huggingface.co/docs/transformers/tasks/sequence_classification) — adim adim fine-tuning
- [BERTurk (dbmdz)](https://huggingface.co/dbmdz/bert-base-turkish-cased) — Turkce BERT model karti
- [Jay Alammar: The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — BERT'in gorsel aciklamasi
- [Devlin et al., 2019 — BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — orijinal BERT makalesi
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) — BERT ve transfer ogrenme iceren NLP kursu

### Genel NLP ve Derin Ogrenme

- [HuggingFace NLP Kursu (ucretsiz)](https://huggingface.co/learn/nlp-course) — baslangictan ileriye NLP
- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA) — Naive Bayes'in sezgisel aciklamasi
- [StatQuest: Word Embeddings](https://www.youtube.com/watch?v=viZrOnJclY0) — embedding'lere gorsel giris
- [Lilian Weng: Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — dikkat mekanizmalari uzerine kapsamli blog yazisi
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — sinir aglarinin nasil ogrendigine dair sezgi olusturma

### Turkce NLP Araclari

- [Zeyrek](https://github.com/obulat/zeyrek) — Turkce morfolojik analiz kutuphanesi
- [Turkce Duygu Veri Seti (Kaggle)](https://www.kaggle.com/datasets/winvoker/turkish-sentiment-analysis-dataset) — etiketli Turkce film yorumlari
- [Trendyol Acik Kaynak Modeller](https://huggingface.co/Trendyol) — Trendyol'un 19 acik kaynak modeli (LLM, embedding, goruntu)

### Metin Cikarimi (Opsiyonel)

- [textract (PyPI)](https://pypi.org/project/textract/) — PDF, DOCX ve diger formatlardan metin cikarimi
- [textract GitHub](https://github.com/deanmalmgren/textract) — desteklenen formatlar ve kurulum talimatlari

---

## Notlar

- **Turkce metin** ile calisiyorsaniz, temel on isleme (kucuk harfe cevirme, noktalama temizleme) sonuclari iyilestirebilir. Gelismis kok bulma/lemmatizasyon icin opsiyonel `--zeyrek` bayragi Zeyrek kutuphanesini kullanir.
- One-hot vektorleri buyuk sozluklerde cok **seyrek** olur. scikit-learn seyrek matrisleri verimli bir sekilde isler.
- Yerlesik ornek veri seti sadece 40 cumle icerir. Sonuclar gosterim amaclidir, guvenilir degildir. Anlamli karsilastirmalar icin 500'den fazla ornekli bir veri seti kullanin.
- Capraz veri seti dogrulama icin `--test-data` parametresini kullanin — egitim ve test verisi farkli kaynaklardan gelir, veri sizintisi riski ortadan kalkar.

---

*Bu proje, metin kodlama ve duygu analizi uzerine bir ders odevi kapsaminda olusturulmustur.*
