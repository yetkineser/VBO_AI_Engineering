# Sonuç Analizi — Deneylerimiz Bize Ne Söylüyor

Bu belge, beş modelin Türkçe kelime gömme kıyaslamalarındaki karşılaştırma sonuçlarını yorumlar. **Her metriğin basit örneklerle ne anlama geldiğini**, **her modelin neden bu şekilde puan aldığını** ve **sonuçlardan ne öğrenebileceğimizi** açıklar.

---

## Deneyin Özeti

Beş gömme modelini dört Türkçe kıyaslamada test ettik:

| Model | Tip | Nasıl çalışır |
|-------|-----|--------------|
| FastText cc.tr.300 | Statik | Her kelimenin kelime eş-oluşumundan öğrenilmiş sabit bir vektörü var |
| BERTurk | Bağlamsal | Her kelime bağlamına göre farklı bir vektör alır |
| XLM-RoBERTa | Bağlamsal | BERT ile aynı ama 100+ dilde eğitilmiş |
| Turkish BERT-NLI-STS | Cümle-transformer | BERT'in benzer cümlelerin benzer vektörlere sahip olması için ince ayar yapılmış hali |
| Multilingual MiniLM | Cümle-transformer | Kompakt çok dilli cümle kodlayıcı |

---

## Metrikleri Anlamak (Basit Örneklerle)

### Spearman ρ (Spearman Sıra Korelasyonu)

**Ne ölçer:** Modelin benzerlik sıralaması insan sıralamasıyla eşleşiyor mu?

**Basit örnek:** İnsanlar tarafından puanlanmış (1–10 ölçeği) üç kelime çifti düşünün:

| Çift | İnsan puanı | Model kosinüs | İnsan sırası | Model sırası |
|------|-------------|--------------|-------------|-------------|
| kedi–köpek | 8.0 | 0.79 | 1. | 1. |
| elma–muz | 6.0 | 0.63 | 2. | 2. |
| kedi–araba | 2.0 | 0.37 | 3. | 3. |

Burada **sıralamalar mükemmel eşleşiyor**, yani Spearman ρ = 1.0. Ham sayılar önemli değil — yalnızca sıralama önemli. Model `elma–muz`'u `kedi–köpek`'in üstüne sıralasaydı, sıralamalar uyuşmazdı ve ρ düşerdi.

**Bizim sonuçlarımız:**

| Model | Spearman ρ | Yorum |
|-------|-----------|-------|
| FastText | **0.571** | İnsan sıralamasıyla %57 oranında uyuşuyor |
| Turkish NLI-STS | **0.514** | Yakın ikinci — benzerlik için ince ayar yapılmış |
| BERTurk | 0.356 | Zayıf — tek-kelime benzerliği için tasarlanmamış |
| MiniLM | 0.265 | Kötü — çok dilli seyreltme |
| XLM-RoBERTa | 0.014 | Rastgele — neredeyse korelasyon yok |

**Neden FastText burada kazanıyor:** Statik gömmeler, milyonlarca kelime eş-oluşumunda eğitilmiş her kelimeye kararlı bir vektör verir. "kedi ile köpek ne kadar benzer?" diye sorduğunuzda, modelin net bir cevabı var. BERT tipi modeller bağlama bağlı bir vektör üretir — yalnızca "kedi" kelimesini çevreleyen bir cümle olmadan verdiğinizde, vektör gürültülü ve güvenilmez olur.

**Bağlantılar:**
- [Vikipedi — Spearman Sıra Korelasyonu](https://tr.wikipedia.org/wiki/Spearman%27%C4%B1n_s%C4%B1ra_korelasyonu)
- [Simply Psychology — Spearman's Rank (görsel rehber)](https://www.simplypsychology.org/spearmans-rank.html)
- [Khan Academy — Korelasyon (video)](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/scatterplots-and-correlation/v/correlation-coefficient-intuition-examples)

---

### Top-1 ve Top-5 Doğruluk (Analoji)

**Ne ölçer:** `vec(B) - vec(A) + vec(C)` hesapladığımızda, beklenen cevap D en yakın kelimeler arasında mı?

**Basit örnek:** `erkek → kadın :: baba → ?` analojisi verildiğinde:

1. `vec(kadın) - vec(erkek) + vec(baba)` hesapla → sonuç vektörü
2. O vektöre en yakın 5 kelimeyi bul
3. **Top-1:** 1. en yakın kelime "anne" mi? → evetse, Top-1 doğru
4. **Top-5:** "anne" ilk 5'te herhangi bir yerde mi? → evetse, Top-5 doğru

Eğer ilk 5 sonuç `[babanın, anne, anneciğim, annesi, dede]` ise:
- Top-1 = ✗ (1. kelime "babanın", "anne" değil)
- Top-5 = ✓ ("anne" 2. pozisyonda)

**Türkçe için neden Top-5 Top-1'den daha önemli:** Türkçe eklemeli bir dildir — `yazdı`, `yazdım`, `yazdılar` hepsi "yazdı"nın geçerli biçimleridir. Model `yazdım` döndürüyorsa ama biz `yazdı` bekliyorsak, gerçekten yanlış değil. Top-5, yüzey biçimi farklı olsa bile doğru kavramı bulmaya puan verir.

**Bizim sonuçlarımız (anlamsal analojiler, 7742 soru):**

| Model | Top-1 | Top-5 | Ne anlama geliyor |
|-------|-------|-------|-------------------|
| FastText | **%35.8** | **%65.1** | Zamanın 2/3'ünde doğru kavramı buluyor |
| Turkish NLI-STS | %15.7 | %22.4 | Cümleler için eğitilmiş, kelime aritmetiği için değil |
| BERTurk | %9.9 | %18.1 | Bağlamsız vektörler güvenilmez |
| MiniLM | %7.8 | %14.6 | Çok dilli = seyreltilmiş |
| XLM-RoBERTa | %4.5 | %7.8 | Temelde rastgele |

**Sürpriz: Turkish NLI-STS sözdizimsel analojilere hükmediyor (%98.5 Top-5).** Neden? Sözdizimsel analojiler `verdi→verdiniz :: geldi→geldiniz` gibi morfolojik kalıpları test eder. Bu model Türkçe NLI verileriyle ince ayar yapılmış ve bu da morfolojik varyasyonu anlamayı gerektirir — yani Türkçe gramerini derinlemesine öğrenmiş.

**Bağlantılar:**
- [Mikolov ve ark., 2013 — Linguistic Regularities (orijinal analoji makalesi)](https://aclanthology.org/N13-1090/)
- [The Illustrated Word2Vec — Analoji bölümü](https://jalammar.github.io/illustrated-word2vec/#analogy)
- [Rogers ve ark., 2017 — Kelime Vektörleriyle Analoji Akıl Yürütme Sorunları](https://aclanthology.org/S17-1017/) — analoji testleri neden yanıltıcı olabilir

---

### MRR (Ortalama Ters Sıralama)

**Ne ölçer:** Ortalama olarak, doğru cevap kaçıncı sırada yer alıyor?

**Basit örnek:** Üç analoji sorusu:

| Soru | Doğru cevap pozisyonu | Ters Sıralama |
|------|----------------------|---------------|
| erkek→kadın :: baba→? | "anne" 1. sırada | 1/1 = 1.000 |
| iyi→kötü :: güzel→? | "çirkin" 3. sırada | 1/3 = 0.333 |
| türkiye→ankara :: fransa→? | "paris" ilk 5'te yok | 0.000 |

**MRR = (1.000 + 0.333 + 0.000) / 3 = 0.444**

MRR, 1. sırada olmasa bile doğru cevabı üst sıralara koyan modelleri ödüllendirir. MRR=0.5 olan bir model, doğru cevabı ortalama ~2. pozisyona koyar.

| MRR | Yorum |
|-----|-------|
| 1.0 | Her zaman 1. pozisyonda doğru |
| 0.5 | Doğru cevap ortalama ~2. pozisyonda |
| 0.33 | Doğru cevap ortalama ~3. pozisyonda |
| 0.0 | Doğru cevabı hiç bulamıyor |

**Bağlantılar:**
- [Vikipedi — Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
- [Croft ve ark. — Search Engines: Information Retrieval in Practice](https://ciir.cs.umass.edu/irbook/) — bilgi erişimi bağlamında MRR

---

### ARI (Düzeltilmiş Rand İndeksi) — Kümeleme Kalitesi

**Ne ölçer:** Kümeler gerçek kategorilerle eşleşiyor mu — şansa göre düzeltilmiş?

**Basit örnek:** 6 kelime ve 2 gerçek kategori olduğunu varsayın:

```
Gerçek etiketler:  [hayvan, hayvan, hayvan, meyve, meyve, meyve]
                    kedi    köpek   kuş     elma   muz    portakal
```

**Mükemmel kümeleme (ARI = 1.0):**
```
Tahmin:            [0, 0, 0, 1, 1, 1]   → Her küme = bir kategori
```

**Rastgele kümeleme (ARI ≈ 0.0):**
```
Tahmin:            [0, 1, 0, 1, 0, 1]   → Hayvanlar ve meyveler rastgele karışmış
```

**Kısmen doğru (ARI ≈ 0.5):**
```
Tahmin:            [0, 0, 1, 1, 1, 1]   → "kuş" yanlış yere konulmuş
```

Neden sadece doğru atamaları saymak yerine ARI? Çünkü her kelimeyi kendi kümesine koyma gibi trivial bir çözüm, basit metriklerde %100 alır. ARI şansı düzeltir, bu yüzden yalnızca anlamlı kümeleme yüksek skor alır.

**Bizim sonuçlarımız (90 kelime, 5 kategori):**

| Model | ARI | NMI | Purity | Derece |
|-------|-----|-----|--------|--------|
| FastText | **0.949** | 0.957 | 0.978 | Mükemmel — 88/90 kelime doğru |
| Turkish NLI-STS | 0.697 | 0.731 | 0.867 | İyi — biraz kategoriler arası karışıklık |
| BERTurk | 0.419 | 0.541 | 0.667 | Orta — kategoriler kısmen karışmış |
| MiniLM | 0.271 | 0.407 | 0.622 | Zayıf |
| XLM-RoBERTa | 0.020 | 0.113 | 0.333 | Rastgele — kümeleme yapısı yok |

**Bağlantılar:**
- [Vikipedi — Rand İndeksi](https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index)
- [scikit-learn — Kümeleme Değerlendirmesi](https://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-index) — görsel örnekler içerir
- [Vinh ve ark., 2010 — NMI karşılaştırması](https://jmlr.org/papers/v11/vinh10a.html)

---

## Büyük Resim: Neden Statik Gömmeler Kazandı

Bu sonuç sezgisel değil gibi görünüyor — BERT modelleri FastText'ten "daha iyi" değil mi? Cevap: **göreve bağlı**.

```
                    Kelime-düzeyi görevler     Cümle-düzeyi görevler
                    (benzerlik, analoji,       (NLI, duygu analizi,
                     kümeleme)                  parafraza tespiti)
                    ─────────────────         ────────────────────
FastText            ████████████████  ★       ████
BERTurk (ham)       ██████                    ████████████
Turkish NLI-STS     ███████████               ████████████████  ★
```

**Statik gömmeler (FastText, GloVe, Word2Vec)** her kelimeye milyarlarca token üzerindeki kelime eş-oluşum kalıplarından eğitilmiş sabit bir vektör atar. Bu vektör kararlı, genel amaçlı ve kelime düzeyinde anlambilimi iyi yakalar.

**Bağlamsal gömmeler (BERT, RoBERTa)** aynı kelime için çevreleyen cümleye bağlı olarak farklı bir vektör üretir. "nehir kıyısı"ndaki "kıyı" ile "kıyı şeridi"ndeki "kıyı" farklı vektörler alır. Bu cümle anlama için güçlüdür ama **bağlam olmadan tek bir kelime verdiğinizde işe yaramaz**. Model `[CLS] kedi [SEP]` görür ve çalışacak bir şeyi yoktur.

**Cümle-transformerlar (NLI-STS, MiniLM)** BERT'in üstüne bir havuzlama katmanı ekler ve benzerlik görevlerinde ince ayar yapar. Tek kelimeler için bile makul vektörler üretirler çünkü ince ayar süreci onlara bağlam uzunluğundan bağımsız olarak anlam kodlamayı öğretir.

### Ders

**Evrensel olarak "en iyi" gömme modeli yoktur.** Doğru seçim görevinize bağlıdır:

| Görev | En iyi model tipi | Neden |
|-------|-------------------|-------|
| Kelime benzerliği | Statik (FastText) | Kararlı, kelime düzeyinde vektörler |
| Kelime analojisi | Statik (FastText) | Vektör aritmetiği sabit vektörlerde çalışır |
| Kelime kümeleme | Statik (FastText) | Temiz, ayrılabilir kelime temsilleri |
| Cümle benzerliği | Cümle-transformer | Cümle düzeyinde karşılaştırma için ince ayar yapılmış |
| Metin sınıflandırma | Cümle-transformer veya ince ayarlı BERT | Cümle düzeyinde anlamı yakalar |
| Varlık ismi tanıma | Bağlamsal (BERT) | Bağlama bağlı kelime anlamı |

---

## Kategori Düzeyinde İçgörüler

### FastText'in üstün olduğu yerler

| Kategori | Top-5 | Neden |
|----------|-------|-------|
| şehir–bölge | %70.5 | Coğrafi isimler eğitim verisinde sık geçer |
| ülke–başkent | %71.4 | Klasik analoji başarı vakası |
| aile (akrabalık) | %70.0 | Cinsiyet/aile ilişkileri iyi kodlanır |
| eş anlamlılar | %65.8 | Benzer kelimeler → benzer bağlamlar |

### Turkish NLI-STS'in üstün olduğu yerler

| Kategori | Top-5 | Neden |
|----------|-------|-------|
| Sözdizimsel analojiler | **%98.5** | Türkçe NLI ile ince ayar → morfolojiyi derinden öğrenmiş |
| eş anlamlılar | **%95.7** | Eş anlamlı tespiti NLI'nin merkezinde |
| zıt anlamlılar | %72.3 | NLI çelişkiyi anlamayı gerektirir |

### Herkesin zorlandığı yerler

| Kategori | En iyi Top-5 | Neden |
|----------|-------------|-------|
| para-birimi | %27.8 (FastText) | Nadir kelimeler, sürekli değişen ilişkiler |
| capital-world | %41.9 (FastText) | Çok fazla benzer şehir/ülke ismi rekabet ediyor |

---

## Kapsam: Gizli Metrik

Kapsam, modelin gerçekten kaç test öğesini değerlendirebildiğidir (yani tüm kelimeler sözlükte mevcut).

| Model | AnlamVer | Anlamsal Analoji | Neden |
|-------|----------|-----------------|-------|
| FastText (200K) | %67.6 | %32.0 | Sabit sözlük — eklemeli biçimler OOV'ye neden olur |
| BERTurk | **%100** | **%100** | Alt-kelime tokenizasyonu herhangi bir dizgiyi ele alır |
| XLM-RoBERTa | **%100** | **%100** | Aynı |
| Turkish NLI-STS | **%100** | **%100** | Aynı |
| MiniLM | **%100** | **%100** | Aynı |

FastText'in anlamsal analojilerde %32 kapsamı, **soruların %68'ini deneyemediği** anlamına gelir. Kapsamı artırabilsek (daha fazla kelime yükleyerek veya alt-kelime geri dönüşüyle tam FastText modelini kullanarak), skorları muhtemelen önemli ölçüde değişirdi.

Bu önemli bir uyarıdır: **FastText'in yüksek doğruluğu kısmen yalnızca "kolay" soruları yanıtladığı içindir** — vektörleri olan yüksek-frekanslı kelime çiftleri. Transformer modelleri nadir ve karmaşık kelime biçimleri dahil her şeyi yanıtladı.

---

## Bu Sonuçları Ne İyileştirir?

| Yaklaşım | Beklenen iyileşme | Zorluk |
|----------|-------------------|--------|
| FastText'i `limit=500000` ile yükle | +%10 kapsam, karışık doğruluk etkisi | Düşük — sadece parametre değiştir |
| Tam FastText modeli kullan (KeyedVectors değil) | OOV → %0 alt-kelime geri dönüşü ile | Orta — `fasttext` kütüphanesi gerekir |
| BERTurk'ü kelime benzerliği için ince ayar yap | Spearman ρ → 0.6+ | Yüksek — eğitim verisi gerekir |
| Cümle düzeyinde kıyaslama kullan (STS-TR) | Transformer modelleri hükmedecek | Orta — farklı değerlendirme |
| Topluluk: kelimeler için FastText + cümleler için BERT | İki dünyanın en iyisi | Orta |

---

## İleri Okuma

### Statik vs Bağlamsal Gömmeler
- [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — bağlamsal gömmelerin neden farklı olduğu
- [Peters ve ark., 2018 — ELMo: Deep Contextualized Word Representations](https://arxiv.org/abs/1802.05365) — statik ile bağlamsal arasındaki köprü
- [Ethayarajh, 2019 — How Contextual are Contextualized Word Representations?](https://aclanthology.org/D19-1006/) — BERT vektörlerinin bağlamla ne kadar değiştiğini ölçer

### Gömme Değerlendirmesi
- [Bakarov, 2018 — Kelime Gömme Değerlendirme Yöntemleri Anketi](https://arxiv.org/abs/1801.09536) — 19 içsel + 9 dışsal yöntem
- [Schnabel ve ark., 2015 — Denetimsiz Kelime Gömmeleri için Değerlendirme Yöntemleri](https://aclanthology.org/D15-1036/) — eleştirel karşılaştırma
- [wordvectors.org](https://wordvectors.org/) — çevrimiçi değerlendirme aracı

### Türkçe NLP
- [AnlamVer Makalesi (Ercan & Yıldız, 2018)](https://aclanthology.org/C18-1323/) — kullandığımız Türkçe benzerlik kıyaslaması
- [BERTurk (dbmdz)](https://huggingface.co/dbmdz/bert-base-turkish-cased) — model kartı
- [Turkish BERT-NLI-STS](https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr) — model kartı
- [Türkçe Statik Kelime Gömmelerinin Kapsamlı Analizi](https://arxiv.org/abs/2405.07778) — 2024 anketi

### Metriklere Derin Dalışlar
- [Google ML Crash Course — Sınıflandırma Metrikleri](https://developers.google.com/machine-learning/crash-course/classification/accuracy) — doğruluk, hassasiyet, duyarlılık
- [scikit-learn — Kümeleme Metrikleri](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) — ARI, NMI, V-measure örneklerle
- [Towards Data Science — AUC-ROC'u Anlamak](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5) — görsel açıklama
- [StatQuest — Machine Learning Temelleri (oynatma listesi)](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) — tüm önemli metrikler üzerine sezgisel video serisi
