# Week 2: Metinleri Sayısallaştırma ve Basit Duygu Analizi

Bu haftanın ödevi, **ham metni makine öğrenmesinin anlayacağı sayısal temsillere** dönüştürmeyi ve bu temsiller üzerinde **basit bir duygu analizi (sentiment)** modeli kurmayı hedefler. İsteğe bağlı pratik olarak **dosyalardan metin çıkarma** için `textract` kütüphanesi önerilmiştir.

---

## Öğrenmeniz beklenen şeyler (özet)

| Konu | Ne öğrenmeniz isteniyor? |
|------|---------------------------|
| **Metin → sayı** | Model girdi olarak vektör/matris ister; kelime/cümle düzeyinde **encoding** ile bunu sağlarsınız. |
| **Integer encoding (tamsayı kodlama)** | Kelimeleri (veya karakterleri) sözlükteki bir **indeks** ile sayıya eşlersiniz: örn. `"güzel" → 42`. |
| **One-hot encoding** | Her kelime/kategori için **ayrı bir ikili boyut** açarsınız; ilgili kelime varsa o boyut 1, diğerleri 0. |
| **Duygu analizi** | Metnin **duygu etiketi** (ör. olumlu / olumsuz / nötr) tahmin eden bir **sınıflandırma** problemi. |
| **Basit model** | Derin öğrenme zorunlu değil; `scikit-learn` ile **Naive Bayes**, **lojistik regresyon**, **SVM** gibi klasik modeller tipik beklenti. |
| **textract (isteğe bağlı)** | PDF, DOC, DOCX vb. dosyalardan **düz metin çıkarma**; veri kaynağınız dosya ise iş akışına girer. |

---

## Kavramları açıklayarak: ne demek istiyorlar?

### 1. “Verilen metinleri … sayısallaştırma”

Bilgisayar metni doğrudan “anlamaz”; önce **sayı dizisi** veya **seyrek/yoğun vektör** haline getirirsiniz. Bu adıma genelde **özellik çıkarımı (feature extraction)** veya **metin vektörleştirme (vectorization)** denir.

### 2. Integer encoding (metin bağlamında)

- **Kelime düzeyi:** Tüm eğitim metinlerinden bir **kelime dağarcığı (vocabulary)** çıkarırsınız; her kelimeye `0 … |V|-1` arası bir tam sayı verirsiniz. Cümle, bu indekslerin dizisi olur (sabit uzunluk için **padding** veya **kesme** gerekir).
- **Dikkat:** Tamsayılar arasındaki büyüklük ilişkisi genelde **anlamsızdır** (“42”, “7”den “daha iyi” değildir). Bu yüzden birçok modelde ham indeks yerine **one-hot**, **TF-IDF** veya **embedding** kullanılır. Ödev özellikle **one-hot / integer** dediği için ikisini de denemeniz veya en az birini gerekçeyle seçmeniz beklenir.

### 3. One-hot encoding (metin / kelime çantası ile ilişkisi)

- **Kategorik etiket** için: 3 sınıf varsa her örnek `[1,0,0]`, `[0,1,0]`, `[0,0,1]` gibi temsil edilir.
- **Metin için (Bag of Words benzeri):** Kelime dağarcığı boyutunda bir vektör düşünün; cümledeki her kelime için ilgili konum **1**, yoksa **0** (veya sayım için frekans). Bu, pratikte **çok boyutlu ve seyrek** vektörler üretir; `scikit-learn` içinde `CountVectorizer` / `TfidfVectorizer` bu tür temsillerle çalışır.

### 4. “Basit bir duygu analizi modeli”

Tipik akış:

1. Etiketli cümle/metin verisi (ör. olumlu/olumsuz).
2. Metinleri yukarıdaki gibi sayısallaştırma.
3. Eğitim / doğrulama bölünmesi.
4. Bir sınıflandırıcı eğitimi ve test metrikleri (doğruluk, F1, karışıklık matrisi).

“Basit” genelde **klasik ML + vektörleştirme** demektir; BERT gibi transformatörler bu ödevin çekirdeği değildir (haftaya göre değişebilir).

### 5. `textract` pratiği (dosyadan metin)

Ödev metni “metinleri” diyebilir; kaynak **PDF veya Word** ise önce dosyadan UTF-8 metin çıkarmanız gerekir. Python’da [`textract`](https://pypi.org/project/textract/) çeşitli formatlardan metin çıkarmayı dener (sistemde ek bağımlılıklar gerekebilir). Alternatif olarak sadece `.txt` ile de ödev yapılabilir; `textract` isteğe bağlı bir **veri okuma** pratiğidir.

---

## Önerilen uygulama adımları (mini yol haritası)

1. Küçük bir duygu veri seti seçin veya `.txt`/CSV ile kendi örneklerinizi oluşturun (en az iki sınıf: olumlu/olumsuz).
2. **Integer temsil:** `sklearn.feature_extraction.text` ile kelime indeksleri veya `CountVectorizer` ile sayım vektörleri.
3. **One-hot benzeri temsil:** `CountVectorizer(binary=True)` veya çok sınıflı etiketlerde `OneHotEncoder` / `pd.get_dummies` (etiket tarafı için).
4. Model: `MultinomialNB`, `LogisticRegression` veya `LinearSVC`.
5. (İsteğe bağlı) Bir PDF’ten `textract` ile metin çekip aynı boru hattına verin.

---

## Kaynaklar

### Resmi dokümantasyon ve öğreticiler

- [scikit-learn: Working with text data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) — metin sınıflandırma boru hattı (vektörleştirme + model).
- [scikit-learn: `CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [scikit-learn: `TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) *(TF-IDF ödevde özellikle istenmese de pratikte çok kullanılır; one-hot/sayısallaştırma fikrini pekiştirir.)*
- [scikit-learn: `OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) — kategorik (ör. sınıf veya kategori) değişkenler için.
- [scikit-learn: `LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) *(hedef etiketleri sayıya çevirmek için; bazı modellerle birlikte kullanımına dikkat edin.)*

### Duygu analizi / NLP giriş

- [NLTK: Sentiment Analysis (kitap bölümü)](https://www.nltk.org/book/ch06.html) — klasik NLP perspektifi.
- [Kaggle Learn: NLP (course)](https://www.kaggle.com/learn/natural-language-processing) — kısa modüller (İngilizce).

### textract ve dosyadan metin

- [textract (PyPI)](https://pypi.org/project/textract/)
- [textract GitHub](https://github.com/deanmalmgren/textract) — desteklenen formatlar ve kurulum notları.

### Kavramsal okuma

- Jurafsky & Martin, *Speech and Language Processing* — [Logistic regression + NLP bölümleri (online taslak)](https://web.stanford.edu/~jurafsky/slp3/) — özellikle metin sınıflandırma ve lojistik regresyon bağlamı.

---

## Guncel Yaklasimlar (Ekstra)

Bu projede odev gereksinimlerinin otesinde 3 guncel yaklasim da uygulanmistir:

### 1. Transformer Embedding (`--transformer`)

CountVectorizer ve TF-IDF kelimeleri bagimsiz sayar; kelime sirasi ve anlam kaybolur.
Transformer embedding ise onceden egitilmis bir dil modelinin (orn. `paraphrase-multilingual-MiniLM-L12-v2`)
her cumleyi 384 boyutlu **yogun vektor** olarak kodlamasini saglar. Bu vektorler cumlenin
**butunsel anlamini** tasir ve ayni klasik ML modellerine (LogReg, SVM, XGBoost) girdi olarak verilir.
Egitim gerektirmez (sadece encode), ama vektorlestirme kalitesi cok daha yuksektir.

**Neden onemli:** Kelime cantasi (BoW) yaklasimlari "film harikaydı" ile "harikaydı film" arasinda
fark goremez; transformer ise cumle yapisini ve baglamsal anlami yakalar.

### 2. Zero-Shot Classification (`--zero-shot`)

Hicbir egitim verisi kullanmadan, onceden egitilmis bir NLI (Natural Language Inference) modeli ile
duygu tahmini yapar. Model her cumle icin "Bu cumle pozitif/negatif" hipotezlerini degerlendirir.
Etiketli veri yokken bile calisir — ozellikle yeni alan/dil icin hizli prototipleme icin idealdir.

**Neden onemli:** Gercek dunyada etiketli veri toplamak pahali ve yavasdir. Zero-shot ile
bir baseline olusturup "egitim yapmazsam ne elde ederim?" sorusunu cevaplayabilirsiniz.

### 3. Fine-Tuning BERTurk (`--finetune`)

`dbmdz/bert-base-turkish-cased` modelini kendi duygu verimizle ince ayar (fine-tune) yapar.
Model hem Turkce dil bilgisini (onceden ogrenilmis) hem de gorev-spesifik kaliplari (bizim verimizden)
birlestirir. Bu, duygu analizi icin en guclu yaklasimdir ama egitim icin GPU oneriler.

**Neden onemli:** Transfer ogrenme, az veriyle bile yuksek performans saglar.
Buyuk dil modeli genel Turkce'yi bilir; siz sadece "bu gorevde pozitif/negatif ne demek"
bilgisini ekliyorsunuz.

### Karsilastirma ozeti

| Yaklasim | Egitim Gerekli mi? | Veri Gerekli mi? | Guc |
|---|---|---|---|
| CountVectorizer + ML | Evet | Evet | Dusuk |
| TF-IDF + ML | Evet | Evet | Orta |
| Transformer Embedding + ML | Hayir (encode) + Evet (ML) | Evet | Yuksek |
| Zero-Shot | Hayir | Hayir | Orta-Yuksek |
| Fine-Tuning BERTurk | Evet (GPU oneriler) | Evet | En Yuksek |

---

## Guncel Yaklasimlar icin Kaynaklar

### Transformer Embedding

- [Sentence-Transformers dokumantasyonu](https://www.sbert.net/) — sentence embedding kullanimi ve onceden egitilmis modeller
- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Transformer mimarisini gorsel olarak anlatan klasik yazi
- [Jay Alammar: The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — kelime vektorlerinden cumle vektorlerine gecis
- [Hugging Face: paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — bu projede kullanilan model karti

### Zero-Shot Classification

- [Hugging Face: Zero-Shot Classification rehberi](https://huggingface.co/tasks/zero-shot-classification) — kavram ve kullanim ornekleri
- [Yin et al., 2019 — Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — zero-shot metin siniflandirma uzerine temel makale
- [Hugging Face: xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli) — bu projede kullanilan cok dilli zero-shot model

### Fine-Tuning (Transfer Ogrenme)

- [Hugging Face: Text Classification rehberi](https://huggingface.co/docs/transformers/tasks/sequence_classification) — fine-tuning adim adim
- [BERTurk (dbmdz)](https://huggingface.co/dbmdz/bert-base-turkish-cased) — Turkce BERT model karti ve kullanim
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) — NLP dersi, BERT ve transfer ogrenme bolumleri
- [Jay Alammar: The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — BERT mimarisini gorsel olarak anlatan yazi
- [Devlin et al., 2019 — BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — orijinal BERT makalesi

### Genel NLP / Derin Ogrenme

- [Hugging Face NLP Course (ucretsiz)](https://huggingface.co/learn/nlp-course) — baslangictan ileri seviyeye NLP kursu
- [StatQuest: Word Embedding & NLP videolari](https://www.youtube.com/watch?v=viZrOnJclY0) — sezgisel aciklamalar
- [Lilian Weng: Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — dikkat mekanizmasi uzerine kapsamli blog yazisi

---

## Notlar

- **Türkçe metin** kullanacaksanız basit ön işleme (küçük harf, noktalama) sonuçları iyileştirebilir; Türkçe için ileri seviye **stemming/lemmatization** ayrı kütüphane gerektirebilir, ödev kapsamı dışında bırakılabilir.
- Büyük kelime dağarcığında one-hot benzeri vektörler **seyrek (sparse)** olur; `scikit-learn` bunun için uygundur.

---

*Bu dosya, ders ödevi kapsamında öğrenme hedeflerini ve kaynakları derlemek için oluşturulmuştur.*
