# Week 3 — Öğrenme Hedefleri & Çalışma Rehberi

Bu belge, Week 3 ödevinin arkasındaki **neden** sorusunu açıklar. Her bölüm, anlamanız beklenen bir kavramı, kısa bir açıklamayı ve daha derine inmek için derlenmiş bağlantıları (makaleler, yazılar, videolar) içerir.

---

## 1. Kelime gömmeleri neden var?

Klasik metin vektörleştirme (one-hot, Bag-of-Words, TF-IDF) her kelimeyi izole bir sembol olarak temsil eder. Bu seçimin iki sorunu vardır:

1. **Benzerlik kavramı yok.** "araba" ile "otomobil" birbirinden, "araba" ile "muz" kadar farklıdır.
2. **Yüksek boyutluluk.** 100K kelimelik bir sözlükle 100K boyutlu seyrek vektörler elde edersiniz; bunları modern sinir ağlarına özellik olarak vermek zordur.

Kelime gömmeleri bu iki sorunu, her kelime için **yoğun, düşük boyutlu** bir vektör öğrenerek çözer; öyle ki *benzer kelimelerin vektörleri de benzerdir*. Benzerlik bağlamdan öğrenilir: aynı tür cümlelerde geçen kelimeler birbirine yakın konumlanır.

**Çalışma bağlantıları**
- [Jay Alammar — The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — en net görsel giriş.
- [Chris McCormick — Word2Vec Tutorial Part 1](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) — somut sayılarla skip-gram.
- [StatQuest — Word Embedding and Word2Vec, Clearly Explained](https://www.youtube.com/watch?v=viZrOnJclY0) — 15 dakikalık sezgisel video.
- [Stanford CS224N Ders 1 — Word Vectors](https://www.youtube.com/watch?v=rmVRLeJRkl4) — standart haline gelmiş ders.

---

## 2. Word2Vec, GloVe ve FastText — aynı fikrin üç tadı

| Model | Yıl | Temel fikir | Güçlü yanı |
|---|---|---|---|
| **Word2Vec** | 2013 | Bağlamdan kelimeyi (CBOW) veya kelimeden bağlamı (skip-gram) tahmin et | Hızlı, ikonik temel yaklaşım |
| **GloVe** | 2014 | Küresel kelime eş-oluşum matrisini çarpanlarına ayır | Sadece yerel pencereleri değil, küresel istatistikleri de yakalar |
| **FastText** | 2016 | Her kelimeyi **karakter n-gramlarının** toplamı olarak temsil et | Nadir kelimeleri ve morfolojiyi ele alır — Türkçe için mükemmel |

Türkçe için FastText doğru tercihtir çünkü Türkçe **eklemeli** bir dildir — tek bir kök düzinelerce yüzey biçimi üretebilir (`kitap`, `kitabım`, `kitaplarımızda`). FastText, o kesin biçim eğitim sırasında görülmemiş olsa bile `kitaplarımızda` için alt-kelime parçalarından bir vektör oluşturabilir.

**Çalışma bağlantıları**
- [Mikolov ve ark., 2013 — Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781) — orijinal Word2Vec makalesi.
- [Pennington ve ark., 2014 — GloVe](https://nlp.stanford.edu/pubs/glove.pdf) — orijinal GloVe makalesi.
- [Bojanowski ve ark., 2017 — Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText makalesi.
- [Facebook FastText — 157 dil için önceden eğitilmiş vektörler](https://fasttext.cc/docs/en/crawl-vectors.html) — `cc.tr.300` buradan indirilir.
- [Gensim dokümanları — KeyedVectors](https://radimrehurek.com/gensim/models/keyedvectors.html) — kullanacağınız API.

---

## 3. Kosinüs benzerliği — gömmeler için mesafe metriği

Her kelime bir vektör olduğunda doğal soru şudur: *iki kelime ne kadar benzer?* Gömmeler için cevap neredeyse her zaman **kosinüs benzerliğidir**:

```
cos(u, v) = (u · v) / (||u|| * ||v||)
```

Kosinüs benzerliği vektör **büyüklüğünü** yok sayar ve yalnızca **yönle** ilgilenir. Bu önemlidir çünkü gömme büyüklükleri genellikle kelime sıklığından etkilenir (daha sık kelimeler daha büyük norma sahip olma eğilimindedir) ve "bir" kelimesinin "felsefe"den "daha güçlü" görünmesini istemezsiniz.

- `+1` — aynı yön (maksimum benzerlik)
- `0`  — dik (ilişkisiz)
- `-1` — zıt yön (teoride karşıt anlamlılar — pratikte nadiren görülür)

**Çalışma bağlantıları**
- [Vikipedi — Kosinüs benzerliği](https://tr.wikipedia.org/wiki/Kosin%C3%BCs_benzerli%C4%9Fi) — hızlı referans.
- [Machine Learning Mastery — How to Calculate Cosine Similarity](https://machinelearningmastery.com/cosine-similarity-for-nlp/) — Python örnekleri.
- [sklearn — cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) — referans uygulaması.
- [Gensim — KeyedVectors.similarity](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.similarity) — hızlı yol.

---

## 4. Gömme uzayında K-Means kümeleme

Kelimeler arasında benzerlik ölçme yöntemimiz olunca doğal soru şudur: *ilgili kelimelerin gruplarını otomatik olarak keşfedebilir miyiz?* K-Means bunu yapan en basit algoritmadır.

**Nasıl çalışır, tek paragrafta:** Rastgele `k` başlangıç küme merkezi seçin. Her noktayı en yakın merkezine atayın. Her merkezi, kendisine atanan noktaların ortalaması olarak yeniden hesaplayın. Hiçbir şey değişmeyene kadar tekrarlayın. Sonuç, her noktanın kendi merkezine diğer herhangi bir merkezden daha yakın olduğu `k` kümedir.

Bilmeniz gereken temel parametreler:
- `n_clusters=k` — kaç grup olacağı.
- `random_state=42` — rastgele başlangıcı sabitler, tekrarlanabilirlik için.
- `n_init=10` — K-Means'i farklı rastgele tohumlardan 10 kez çalıştırır ve en iyisini tutar. Ödev bunu ister çünkü tek çalışma kötü bir yerel minimuma takılabilir.

Bir uyarı: K-Means **Öklid mesafesini** kullanır, kosinüs değil. Gömme kümelemesi için ya (a) vektörlerinizi önce L2 normuyla normalize edin — böylece Öklid mesafesi kosinüs mesafesine eşdeğer olur — ya da (b) ham vektörler üzerinde `sklearn.cluster.KMeans` kullanın ve sonuçların kosinüs tabanlı kümelemeye yakın ama birebir aynı olmayacağını kabul edin.

**Çalışma bağlantıları**
- [StatQuest — K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA) — 8 dakikalık görsel açıklama.
- [sklearn — KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — kullanacağınız API.
- [sklearn — Kümeleme Kullanıcı Kılavuzu](https://scikit-learn.org/stable/modules/clustering.html#k-means) — K-Means'in ne zaman çalışıp ne zaman çalışmadığının tartışması.
- [Google Developers — K-Means Avantajlar ve Dezavantajlar](https://developers.google.com/machine-learning/clustering/algorithm/advantages-disadvantages) — ödünleşmeleri bilin.

---

## 5. Pratik mühendislik dersleri

Ödev küçük, ama gerçek dünya NLP çalışmasında önemli olan çeşitli mühendislik reflekslerini çalıştırır:

1. **Büyük dosyaların git'te yeri yoktur.** Türkçe FastText modeli ~4.5 GB'dır. Onu `.gitignore`'a ekleyin ve indirme adımını README'de belgeleyin.
2. **Tembel yükleme önemlidir.** 2 milyon kelime vektörünü yüklemek dakikalar ve çok fazla RAM alır. Geliştirme sırasında `load_word2vec_format()` fonksiyonunun `limit=` parametresini kullanarak yalnızca en sık geçen N kelimeyi yükleyin.
3. **Tekrarlanabilirlik.** Rastgelelik kullanan herhangi bir algoritma için daima `random_state` ayarlayın. Ödev açıkça `random_state=42` ister.
4. **OOV bir istisna değil, varsayılan durumdur.** Aradığınız herhangi bir kelimenin sözlükte olmayabileceğini varsayın ve programın çökmesine izin vermek yerine `None` döndürün veya net bir hata fırlatın.
5. **Aramadan önce normalize edin.** `"Kedi"`, `"KEDİ"` ve `"kedi."` hepsi aynı vektöre eşlenmelidir. Küçük harfe çevirme ve noktalama temizliği asgari gereksinimdir.

**Çalışma bağlantıları**
- [Gensim — bellek-verimli yükleme](https://radimrehurek.com/gensim/models/keyedvectors.html#how-to-obtain-word-vectors) — `limit` ve `binary` parametreleri.
- [The Twelve-Factor App — III. Config](https://12factor.net/config) — model yollarının neden kodda değil, ortam değişkenlerinde olması gerektiği.
- [Hugging Face NLP Course — Tokenization](https://huggingface.co/learn/nlp-course/chapter6/1) — metin normalizasyonu üzerine daha derin arka plan.

---

## 6. Türkçeye özgü NLP notları

Türkçenin kelime gömmeleri için ilginç bir test durumu olmasını sağlayan birkaç özelliği vardır:

- **Eklemeli morfoloji.** Ekler zaman, kişi, durum, iyelik vb. kodlamak için üst üste binebilir. Bu, yüzey sözlüğünü patlatır.
- **Ünlü uyumu.** Eklerdeki ünlüler köke uyum sağlamak için değişir. Bu, FastText gibi alt-kelime modelleri için önemlidir.
- **Görece serbest kelime sırası.** Bağlam pencereleri yine çalışır, ama bir kelimenin "komşuları" İngilizce'dekinden daha değişken olabilir.
- **Özel karakterler.** `ı`, `İ`, `ş`, `ğ`, `ö`, `ü`, `ç` — tokenizasyon ve normalizasyonunuzun bunları doğru koruduğundan emin olun.

**Çalışma bağlantıları**
- [Zemberek — Türkçe NLP araç seti](https://github.com/ahmetaa/zemberek-nlp) — lemmatizasyon, morfoloji vb.
- [Zeyrek — Zemberek morfolojisinin Python portu](https://github.com/obulat/zeyrek) — daha hafif bir alternatif.
- [Türkçe NLP Kaynakları (awesome list)](https://github.com/topics/turkish-nlp) — GitHub toplayıcısı.
- [Stemming and Lemmatization for Turkish (blog)](https://towardsdatascience.com/text-preprocessing-for-turkish-4f1abb72a9d8) — pratik rehber.

---

## 7. Bu ödevi "anlamak" neye benzer?

Ödevi bitirdikten sonra, hiçbir şeye bakmadan aşağıdakileri yanıtlayabilmelisiniz:

- Bir kelime gömme vektörü neyi temsil eder? Sayılar nereden gelir?
- Gömmeler için Öklid mesafesi yerine neden kosinüs benzerliği tercih edilir?
- Word2Vec, GloVe ve FastText arasındaki fark birer cümleyle nedir?
- FastText neden Türkçeyi Word2Vec'ten daha iyi ele alır?
- K-Means üzerinde `random_state` ve `n_init` değerlerini neden ayarlıyoruz?
- "Açıkça benzer" iki kelimenin modelinizde düşük kosinüs benzerliğine sahip olduğu durumu nasıl hata ayıklarsınız?

Bunlardan herhangi biri sallantılı hissettiriyorsa, yukarıdaki ilgili bölümü yeniden ziyaret edin.

---

## 8. Ödevin ötesine geçmek

Erken bitirir ve daha ileri gitmek isterseniz, çok şey öğreten üç yön:

1. **Analoji görevleri** — `kral - erkek + kadın` hesaplayın ve en yakın kelimeyi bulun. Klasik bir duman testi.
2. **Görselleştirme** — kümelenmiş kelimelerinize PCA veya t-SNE uygulayın ve 2D bir dağılım grafiği çizin. Kümeleri görmek kavramı çok daha somut hale getirir.
3. **Modelleri karşılaştırın** — hem FastText'i hem de GloVe'u yükleyin, her ikisinde de aynı benzerlik ve kümeleme görevlerini çalıştırın ve sezginize hangisinin daha çok uyduğunu rapor edin.

Bunlar bu klasördeki `EXTRA_SUGGESTIONS_TR.md` dosyasında ayrıntılı olarak ele alınmıştır.
