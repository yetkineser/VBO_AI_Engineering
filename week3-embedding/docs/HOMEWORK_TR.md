# Week 3 Ödevi — Türkçe Kelime Gömmeleri (Word Embeddings)

## Amaç

Önceden eğitilmiş bir **Türkçe kelime gömme modelini** (FastText veya GloVe) yükleyen, kelime vektörlerini alan, iki kelime arasındaki anlamsal benzerliği ölçen ve ilgili kelimeleri **K-Means kümeleme** ile gruplayan küçük bir Python projesi yazın.

Bu ödevin amacı, kelime gömmelerinin arkasındaki teorik kavramları (yoğun vektörler, kosinüs benzerliği, anlamsal uzayda kümeleme) çalışan ve tekrarlanabilir bir boru hattına dönüştürmektir.

---

## Arka Plan

Kelime gömmeleri, her kelimeyi sabit uzunlukta yoğun bir vektöre eşler; öyle ki anlam olarak benzer kelimeler vektör uzayında birbirine yakın konumlanır. Tek-sıcak (one-hot) veya sayım tabanlı vektörlerin aksine, gömmeler **anlamsal ilişkileri** yakalar — örneğin `vector("kral") - vector("erkek") + vector("kadın")` yaklaşık olarak `vector("kraliçe")` vektörüne eşittir.

Bu ödevde **önceden eğitilmiş** gömmeler kullanacağız; yani modeli büyük bir Türkçe külliyat üzerinde (Wikipedia + Common Crawl) başkası eğitmiş durumda. Biz sadece vektörleri **tüketeceğiz**.

Önceden eğitilmiş Türkçe kelime vektörleri için iki popüler kaynak:

| Kaynak | Dosya | Boyut | Sözlük |
|---|---|---|---|
| Facebook FastText (`cc.tr.300`) | `cc.tr.300.vec.gz` | ~4.5 GB | ~2M kelime |
| Stanford GloVe (topluluk Türkçe yapımları) | `glove.tr.300.txt` | ~1 GB | ~400K kelime |

İki dosya da **word2vec metin biçimini** takip eder: ilk satır `<kelime_sayısı> <boyut>` içerir ve sonraki her satır `<kelime> <v1> <v2> ... <vN>` şeklindedir. Gensim'in `KeyedVectors.load_word2vec_format()` fonksiyonu bunu doğrudan okur.

---

## Gereksinimler

### 1. Önceden eğitilmiş bir model yükleyin

FastText veya GloVe vektörlerini yükleyip `gensim.models.KeyedVectors` nesnesi döndüren bir fonksiyon yazın.

```python
def load_fasttext_model(path: str) -> KeyedVectors: ...
def load_glove_model(path: str) -> KeyedVectors: ...
```

### 2. Kelime vektörünü alın

Yüklenmiş bir model ve bir kelime verildiğinde, kelimenin 300 boyutlu vektörünü döndürün. Sözlükte olmayan (OOV) durumu şık biçimde ele alın — `None` döndürün veya açık bir hata fırlatın; çökmesin.

```python
def get_word_vector(model, word: str) -> np.ndarray | None: ...
```

### 3. Kelime benzerliği (kosinüs)

İki kelime arasındaki kosinüs benzerliğini hesaplayın. `[-1, 1]` aralığında bir float döndürün. OOV kelimeleri ele alın.

```python
def word_similarity(model, word1: str, word2: str) -> float: ...
```

### 4. K-Means kümeleme

Bir kelime listesini K-Means ile `k` kümeye ayırın (`random_state=42`, `n_init=10`). Her kelimeyi küme etiketine eşleyen bir sözlük veya DataFrame döndürün.

```python
def cluster_words(model, words: list[str], k: int = 3) -> dict[str, int]: ...
```

### 5. Komut satırı arayüzü

Aşağıdakileri yapan bir `main.py` betiği yazın:

1. Modeli yapılandırılabilir bir yoldan yükler.
2. Birkaç kelime vektörünü yazdırır (tüm 300 boyutu değil, sadece başını).
3. Birkaç kelime çifti için benzerlik hesaplar.
4. Örnek bir kelime listesini `k=3` ile kümeler ve grupları yazdırır.
5. Sonuçları `outputs/` altına CSV veya Markdown olarak kaydeder.

### Örnek çıktı

```
kedi ↔ köpek    : 0.78
araba ↔ otobüs : 0.71
elma ↔ muz     : 0.64
kedi ↔ araba   : 0.12

Küme 0: kedi, köpek, kuş, balık        (hayvanlar)
Küme 1: araba, otobüs, uçak, tren     (araçlar)
Küme 2: elma, muz, portakal, çilek    (meyveler)
```

---

## Teslim Edilecekler

- `src/embedding_utils.py` — yukarıdaki dört fonksiyon.
- `src/main.py` — CLI giriş noktası.
- `requirements.txt` — bağımlılıklar.
- `README.md` — modelin nasıl indirileceği, kurulum ve çalıştırma.
- `outputs/` — benzerlik tablosu ve küme atamaları.

---

## Kısıtlar & İpuçları

- **Gömme dosyasını git'e commit etmeyin** — çok büyük. `data/*.vec*` ve `data/*.gz` kalıplarını `.gitignore`'a ekleyin.
- **Bir kez indirin, tekrar kullanın.** Tam FastText dosyasını yüklemek dakikalar sürer. Yalnızca en sık geçen N kelimeyi `limit=200000` ile yüklemeyi düşünün.
- **Tam FastText model nesnesi yerine `KeyedVectors`** kullanın — çok daha hızlıdır ve çok daha az bellek kullanır.
- **Kosinüs benzerliği** = L2 ile normalize edilmiş vektörlerin nokta çarpımıdır. sklearn'ün `cosine_similarity` fonksiyonu çalışır; gensim'in `model.similarity(w1, w2)` fonksiyonu da çalışır ve genelde daha hızlıdır.
- **OOV önemli.** Türkçe eklemeli bir dildir; "git" mevcut olsa bile "gidiyorum" sözlükte olmayabilir. FastText'in alt-kelime desteği bunu yardımcı olur (sadece KeyedVectors değil, tam modeli yüklerseniz), ama bu ödevde KeyedVectors seviyesinde kalıyoruz.
- Arama öncesi metni normalize edin (küçük harf, noktalama temizliği).

---

## Değerlendirme

Gönderiminiz şu kriterlere göre değerlendirilecek:

1. **Doğruluk** — dört fonksiyon belirtildiği gibi çalışıyor mu?
2. **Sağlamlık** — OOV kelimeleri ve boş girdiyi ele alıyor mu?
3. **Tekrarlanabilirlik** — klon, kurulum, çalıştırma sürprizsiz mi?
4. **Kod kalitesi** — tip ipuçları, docstring'ler, anlamlı bir CLI.
5. **Dokümantasyon** — *ne*, *nasıl* ve *neden* sorularını açıklayan bir README.

Başarılar!
