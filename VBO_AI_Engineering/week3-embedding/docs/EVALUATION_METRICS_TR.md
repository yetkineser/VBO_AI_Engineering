# Kelime Gömmelerini Değerlendirme — Metrikler, Yöntemler & Kıyaslamalar

Kelime gömmeleri tek bir doğruluk sayısı üreten bir model değildir. Bir **temsil** biçimidir — kelimeleri vektör olarak kodlama yoludur. Değerlendirmek şu soruyu sormayı gerektirir: *"Bu temsil, dilbilgisel bilgiyi ne kadar iyi yakalıyor?"*

İki geniş değerlendirme ailesi vardır: **içsel (intrinsic)** (vektörleri doğrudan test et) ve **dışsal (extrinsic)** (bir alt göreve tak ve görev performansını ölç).

---

## 1. İçsel (Intrinsic) Değerlendirme

İçsel yöntemler, bir alt model eğitmeden gömme uzayının kendisini test eder. Hızlı, ucuz ve size hızlı bir teşhis verir.

### 1.1 Kelime Benzerliği (İnsan Değerlendirmeleriyle Korelasyon)

**Ne ölçer:** Kosinüs benzerlik skorları, insanların kelime ilişkiselliğini puanlamasıyla uyuşuyor mu?

**Nasıl çalışır:**
1. İnsan tarafından puanlanmış kelime çiftlerinden oluşan bir kıyaslama veri seti alın (ör. 1–10 ölçeği).
2. Gömmelerinizi kullanarak her çift için kosinüs benzerliğini hesaplayın.
3. İki sıralama arasındaki **Spearman sıra korelasyonunu** (ρ) ölçün.

**Neden Spearman, Pearson değil?** Biz *sıralama* ile ilgileniyoruz ("A-B, C-D'den daha mı benzer?"), ham sayılarla değil. Spearman doğrusallık varsaymadan monotonik ilişkileri yakalar.

| Metrik | Formül | Aralık | Yorum |
|--------|--------|--------|-------|
| Spearman ρ | İnsan skorları ile kosinüs benzerlikleri arasında sıra korelasyonu | [-1, +1] | +1 = tam uyum, 0 = korelasyon yok |
| Pearson r | Doğrusal korelasyon | [-1, +1] | Doğrusal ilişki varsayar (daha az uygun) |

**Kıyaslama veri setleri:**

| Veri Seti | Dil | # Çift | Ne ölçer |
|-----------|-----|--------|----------|
| [WordSim-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) | İngilizce | 353 | İlişkisellik + benzerlik |
| [SimLex-999](https://fh295.github.io/simlex.html) | İngilizce | 999 | Saf benzerlik (ilişkisellik değil) |
| [MEN](https://staff.fnwi.uva.nl/e.bruni/MEN) | İngilizce | 3000 | İlişkisellik |
| [RG-65](https://aclanthology.org/J91-1003/) | İngilizce | 65 | Klasik küçük kıyaslama |
| [AnlamVer](https://github.com/Wikipedia2Vec/AnlamVer) | Türkçe | 500 | Türkçe kelime benzerliği |

**Önemli ayrım:** *Benzerlik (similarity)* ile *ilişkisellik (relatedness)*. "Kahve" ve "çay" **benzer**dir (ikisi de içecek). "Kahve" ve "fincan" **ilişkili**dir (sık birlikte geçer) ama benzer değildir. SimLex-999 benzerliği ölçer; WordSim-353 ikisini karıştırır. Kıyaslamanızın hangisini test ettiğini bilin.

**Bağlantılar:**
- [Faruqui & Dyer, 2014 — Community Evaluation of Word Vectors](https://www.aclweb.org/anthology/W14-1508/) — birden fazla kıyaslamayı karşılaştırır
- [wordvectors.org](https://wordvectors.org/) — vektörlerinizi çevrimiçi değerlendirme aracı
- [Bakarov, 2018 — A Survey of Word Embeddings Evaluation Methods](https://arxiv.org/abs/1801.09536) — kapsamlı anket

---

### 1.2 Kelime Analojisi (Doğruluk)

**Ne ölçer:** Gömme uzayı "A, B'ye ne ise C de D'ye odur" gibi ilişkisel kalıpları yakalayabiliyor mu?

**Nasıl çalışır:**
1. Analoji dörtlüleri alın: (A, B, C, D).
2. `vec(B) - vec(A) + vec(C)` hesaplayın.
3. Sonuç vektörüne en yakın kelimeyi bulun (A, B, C hariç).
4. En yakın kelimenin D olup olmadığını kontrol edin.

**Metrikler:**

| Metrik | Açıklama |
|--------|----------|
| **Top-1 Doğruluk** | 1. sonucun beklenen kelime olduğu analojilerin yüzdesi |
| **Top-5 Doğruluk** | Beklenen kelimenin ilk 5'te göründüğü analojilerin yüzdesi |
| **MRR (Ortalama Ters Sıralama)** | Her beklenen kelime için 1/sıralama ortalaması. Doğru cevabı üst sıralara koyan modelleri ödüllendirir |

```
Top-1 Doğruluk = doğru_1_de / toplam_analoji
MRR = (1/N) * Σ (1 / doğru_cevabın_sırası)
```

**Kıyaslama veri setleri:**

| Veri Seti | # Analoji | Kategoriler |
|-----------|-----------|-------------|
| [Google Analogy](https://aclweb.org/aclwiki/Google_analogy_test_set_(State_of_the_art)) | 19.544 | Anlamsal (başkentler, para birimi, cinsiyet) + Sözdizimsel (zaman, çoğul) |
| [BATS](https://vecto.space/projects/BATS/) | 99.200 | 40 ilişki türü, Google'dan daha dengeli |
| [SemEval-2012 Task 2](https://aclanthology.org/S12-1047/) | 79 ilişki türü | Daha ince taneli ilişkisel benzerlik |

**Uyarılar:**
- Analoji doğruluğu **kırılgandır**. Küçük sözlük boşlukları (OOV) veya morfolojik varyasyon, gerçek gömme kalitesini yansıtmayan başarısızlıklara neden olabilir.
- Türkçe özellikle zorludur çünkü eklemeli morfoloji, "beklenen" cevabın farklı bir yüzey biçiminde görünebileceği anlamına gelir (ör. `yazdı` vs `yazdım`).
- Morfolojik olarak zengin diller için kesin Top-1 yerine **Top-5 doğruluk** veya **MRR** kullanmayı düşünün.

**Bağlantılar:**
- [Mikolov ve ark., 2013 — Linguistic Regularities in Continuous Space Word Representations](https://aclanthology.org/N13-1090/) — analoji görevini tanıttı
- [Levy & Goldberg, 2014 — Linguistic Regularities in Sparse and Explicit Word Representations](https://aclanthology.org/W14-1618/) — analojinin sihir olmadığını gösterir
- [Rogers ve ark., 2017 — Too Many Problems of Analogical Reasoning with Word Vectors](https://aclanthology.org/S17-1017/) — analoji değerlendirmesinin eleştirel analizi

---

### 1.3 Kelime Kategorilendirme / Kümeleme (Purity, NMI, ARI)

**Ne ölçer:** Kelime vektörlerini kümelediğinizde, ortaya çıkan kümeler bilinen anlamsal kategorilerle uyuşuyor mu?

**Nasıl çalışır:**
1. Bilinen kategori etiketleri olan bir kelime seti alın (ör. hayvanlar, araçlar, meyveler).
2. Kelime vektörlerini K-Means (veya başka bir algoritma) ile kümeleyin.
3. Tahmin edilen kümeleri gerçek etiketlerle kümeleme metrikleri kullanarak karşılaştırın.

**Metrikler:**

| Metrik | Aralık | Ne ölçer |
|--------|--------|----------|
| **Purity (Saflık)** | [0, 1] | Her küme tek bir sınıf tarafından domine edilir. Basit ama daha fazla kümeye doğru önyargılı. |
| **NMI (Normalleştirilmiş Karşılıklı Bilgi)** | [0, 1] | Tahmin edilen ve gerçek etiketler arasında karşılıklı bilgi, normalleştirilmiş. Farklı küme sayılarını ele alır. |
| **ARI (Düzeltilmiş Rand İndeksi)** | [-1, 1] | İki kümeleme arasındaki uyumu ölçer, şansa göre düzeltilmiş. 0 = rastgele, 1 = mükemmel. |
| **V-Measure** | [0, 1] | Homojenlik (her kümede tek sınıf) ile tamlığın (her sınıf tek kümede) harmonik ortalaması. |

```python
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score(gercek_etiketler, tahmin_etiketler)
ari = adjusted_rand_score(gercek_etiketler, tahmin_etiketler)
```

**Neden Purity yerine ARI?** Purity daha fazla kümeyle her zaman artar (her kelime kendi kümesiyse trivial olarak 1.0). ARI şansı düzeltir, bu da farklı `k` değerleri arasında karşılaştırılabilir kılar.

**Bağlantılar:**
- [sklearn — Kümeleme Değerlendirmesi](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) — tüm metriklerin mükemmel genel bakışı
- [Rosenberg & Hirschberg, 2007 — V-Measure](https://aclanthology.org/D07-1043/) — homojenlik + tamlık
- [Vinh ve ark., 2010 — Information Theoretic Measures for Clusterings Comparison](https://jmlr.org/papers/v11/vinh10a.html) — NMI varyantlarına derin dalış

---

### 1.4 Aykırı Değer Tespiti (OPP Skoru)

**Ne ölçer:** İlişkili kelimeler artı bir aykırı değer grubu verildiğinde, model aykırı değeri tespit edebilir mi?

**Nasıl çalışır:**
1. `{kedi, köpek, kuş, araba}` gibi bir set sunun.
2. Her kelimenin diğerlerine ortalama kosinüs benzerliğini hesaplayın.
3. En düşük ortalama benzerliğe sahip kelime tahmin edilen aykırı değerdir.
4. Bilinen aykırı değerle eşleşip eşleşmediğini kontrol edin.

**Metrik:** **OPP (Outlier Position Percentage)** — tüm test setlerinde aykırı değeri doğru tespit etme doğruluğu.

Bu, analojilerden daha basit, daha sezgisel bir test ve morfolojik varyasyona daha az duyarlıdır.

**Bağlantılar:**
- [Camacho-Collados & Navigli, 2016 — Find the Word Intruder](https://aclanthology.org/D16-1153/) — aykırı değer tespiti görevini resmileştirir
- [8-8-8 Dataset](https://github.com/Wikipedia2Vec/outlier-detection) — standart kıyaslama

---

## 2. Dışsal (Extrinsic) Değerlendirme

Dışsal yöntemler gömmeleri gerçek bir NLP görevine takar ve görev performansını ölçer. Pratik soruyu yanıtlarlar: *"Bu gömmeler sistemimi daha iyi yapıyor mu?"*

### 2.1 Metin Sınıflandırma (Accuracy, F1)

Kelime gömmelerini özellik olarak kullanın (ör. belge başına ortalama kelime vektörleri) ve bir sınıflandırıcı eğitin. TF-IDF temeli ile karşılaştırın.

| Metrik | Ne söyler |
|--------|-----------|
| **Accuracy** | Genel doğruluk |
| **F1 (makro)** | Hassasiyet ve duyarlılık arasındaki denge, sınıflar arasında ortalaması |
| **AUC-ROC** | Sıralama kalitesi, eşikten bağımsız |

Bu tam olarak Week 2'de yaptığımız şeydir — ve gömme kalitesinin en pratik ölçüsüdür.

### 2.2 Varlık İsmi Tanıma (F1)

Gömmeleri bir dizi etiketleyiciye (BiLSTM-CRF veya benzeri) girdi özellikleri olarak kullanın. Varlık düzeyinde F1 ölçün.

### 2.3 Anlamsal Metin Benzerliği (Pearson/Spearman)

Kelime vektörlerini ortalayarak cümle gömmeleri oluşturun, ardından insan cümle benzerlik değerlendirmeleriyle (STS Benchmark) korelasyon kurun.

**Bağlantılar:**
- [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) — standart cümle benzerliği kıyaslaması
- [Conneau & Kiela, 2018 — SentEval](https://arxiv.org/abs/1803.05449) — cümle temsillerini değerlendirme araç seti

---

## 3. Özet: Hangi Görev İçin Hangi Metrik?

```
┌──────────────────────────┬────────────────────────┬─────────────────────────┐
│ Ne ölçmek istiyorsunuz   │ Yöntem                 │ Ana metrik              │
├──────────────────────────┼────────────────────────┼─────────────────────────┤
│ Anlamsal benzerlik       │ Kelime benzerlik kıyas.│ Spearman ρ              │
│ İlişkisel kalıplar       │ Kelime analojisi       │ Top-5 Doğruluk / MRR   │
│ Kümeleme kalitesi        │ K-Means + altın etiket │ ARI / NMI               │
│ Aykırı değer tespiti     │ Davetsiz misafir testi │ OPP                     │
│ Alt görev yararlılığı    │ Metin sınıflandırma    │ F1 / AUC-ROC            │
│ Cümle anlama             │ STS kıyaslama          │ Spearman ρ              │
└──────────────────────────┴────────────────────────┴─────────────────────────┘
```

### Bu proje için öneri

Önceden eğitilmiş Türkçe gömmeler kullanan bir ödev ölçeğinde proje için en pratik değerlendirme yaklaşımı:

1. **Kelime benzerliği** — [AnlamVer](https://github.com/Wikipedia2Vec/AnlamVer) Türkçe veri setine karşı Spearman ρ hesaplayın (mevcutsa) veya insan puanlarıyla 20-30 Türkçe kelime çifti oluşturun.
2. **Analoji doğruluğu (Top-5 + MRR)** — Türkçe morfolojisi için Top-1'den daha affedici.
3. **Kümeleme ARI/NMI** — Zaten gerçek etiketleriniz var (hayvanlar/araçlar/meyveler), bu yüzden bu bedava.
4. **Niteliksel inceleme** — t-SNE grafiğine bakın. Kümeler görsel olarak ayrılmış mı? En yakın komşular mantıklı mı?

Tek bir sayının peşinden koşmayın. Birden fazla metrik kullanın ve aralarında tutarlılık arayın.

---

## 4. Yaygın Tuzaklar

| Tuzak | Neden önemli | Ne yapmalı |
|-------|-------------|------------|
| Yalnızca analojilerle değerlendirme | Analoji doğruluğu gürültülü ve literatürde abartılmış | Benzerlik + kümeleme + alt görev kullanın |
| OOV oranını yok saymak | Yüksek OOV, modelin cevap verme şansı bile bulamadığı anlamına gelir | Doğruluk yanında OOV oranını rapor edin |
| Sözlükler arasında karşılaştırma | 2M kelimelik bir model, 200K'lık bir modele karşı haksız avantaja sahiptir | Aynı `limit` kullanın veya sözlük boyutunu rapor edin |
| Türkçe için Top-1 doğruluk kullanma | Eklemeli morfoloji `yazdı` ≠ `yazdım` demektir | Top-5 veya MRR kullanın |
| Metni normalize etmemek | Aramada `"Kedi"` ≠ `"kedi"` | Değerlendirmeden önce daima küçük harf + noktalama temizliği |
| t-SNE'yi aşırı yorumlama | t-SNE deterministik değildir ve kümeler arası mesafeler anlamlı değildir | Yalnızca niteliksel içgörü için kullanın, metrik olarak değil |

---

## 5. İleri Okuma

### Anketler & Genel Bakışlar
- [Bakarov, 2018 — A Survey of Word Embeddings Evaluation Methods](https://arxiv.org/abs/1801.09536) — en kapsamlı anket (19 içsel ve 9 dışsal yöntemi kapsar)
- [Schnabel ve ark., 2015 — Evaluation Methods for Unsupervised Word Embeddings](https://aclanthology.org/D15-1036/) — değerlendirme yaklaşımlarının eleştirel karşılaştırması
- [Wang ve ark., 2019 — Evaluating Word Embedding Models](https://aclanthology.org/P19-1070/) — güncel meta-analiz

### Spesifik Yöntemler
- [Finkelstein ve ark., 2002 — Placing Search in Context](https://dl.acm.org/doi/10.1145/503104.503110) — WordSim-353 makalesi
- [Hill ve ark., 2015 — SimLex-999](https://aclanthology.org/J15-4004/) — neden benzerlik ≠ ilişkisellik
- [Levy ve ark., 2015 — Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://aclanthology.org/Q15-1016/) — hiper-parametreler algoritmalardan daha önemli
- [Gladkova ve ark., 2016 — Analogy-based Detection of Morphological and Semantic Relations](https://aclanthology.org/N16-2002/) — BATS veri seti

### Araçlar
- [wordvectors.org](https://wordvectors.org/) — vektörlerinizi standart kıyaslamalara karşı çevrimiçi değerlendirin
- [VecEval](https://github.com/AKGostar/VecEval) — gömme değerlendirmesi için Python çerçevesi
- [Gensim evaluate_word_pairs()](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.evaluate_word_pairs) — Gensim'de yerleşik değerlendirme
- [Gensim evaluate_word_analogies()](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.evaluate_word_analogies) — yerleşik analoji değerlendirmesi

### Türkçeye Özgü
- [AnlamVer — Türkçe Kelime Benzerliği Veri Seti](https://github.com/Wikipedia2Vec/AnlamVer)
- [Ercan & Yıldız, 2018 — AnlamVer: Intrinsic Evaluation of Word Embeddings for Turkish](https://dergipark.org.tr/en/pub/tbbmd/issue/40485/484526)
