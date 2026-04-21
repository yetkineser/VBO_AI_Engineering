# Hafta 4: Doküman Yükleme Pipeline'ı — MongoDB + Elasticsearch + Vektör Arama

Bir klasördeki dosyaları üç farklı sisteme kaydet: metadata MongoDB'ye, tam metin Elasticsearch'e, yoğun vektör gömmeleri (dense embeddings) semantik arama için. Bu, her RAG uygulamasının ihtiyaç duyduğu depolama katmanı.

---

## Neden Üç Sistem?

| Sistem | Ne saklar | Ne yapar iyi | Ne yapamaz |
|--------|-----------|--------------|------------|
| **MongoDB** | Yapılandırılmış metadata | Yazar, boyut, tarih, etiket ile filtreleme | Metin içinde arama yapamaz |
| **Elasticsearch** | Tam metin (ters indeks) | Tam kelime ve cümle eşleştirme | Anlamca benzer içeriği bulamaz |
| **Vektör DB** | Yoğun gömmeler | Anlam bazlı içerik bulma | Tam cümle eşleştirme yapamaz |

Üretim RAG sistemi **üçünü birlikte** kullanır: metadata ile filtrele, anahtar kelime ile sırala, semantik benzerlik ile tekrar sırala.

---

## Hızlı Başlangıç

### 1. Veritabanlarını başlat (Docker)

```bash
docker run -d --name mongo -p 27017:27017 mongo:latest

docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.17.0
```

### 2. Python bağımlılıklarını kur

```bash
cd VBO_AI_Engineering/week4-vectorization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Pipeline'ı çalıştır

```bash
# Tüm kitapları üç sisteme yükle
python run.py --ingest

# Arama karşılaştırma demosunu çalıştır
python run.py --search "Yapay zeka uygulaması nasıl kurulur?"

# Tüm demo sorgularını çalıştır ve rapor üret
python run.py --demo
```

---

## Proje Yapısı

```
week4-vectorization/
├── run.py                 ← Ana giriş noktası
├── config.py              ← Yollar, model adı, bağlantı ayarları
├── requirements.txt
├── README.md / README_TR.md
├── GUIDE.md               ← Adım adım ödev rehberi
├── LEARNING.md / LEARNING_TR.md  ← Ne öğrenmeli + okuma linkleri
├── src/
│   ├── file_parser.py     ← PDF/EPUB/MD'den metin + metadata çıkarma
│   ├── mongo_store.py     ← MongoDB metadata işlemleri
│   ├── elastic_store.py   ← Elasticsearch metin indeksleme + arama
│   ├── vector_store.py    ← Embedding üretme + kNN arama
│   └── search_demo.py     ← 3 sistemin yan yana karşılaştırması
└── outputs/
    └── search_comparison.md
```

## Veri Kaynağı

Pipeline `~/Desktop/week_4_researchs/` klasöründen okur — AI, veri bilimi, algoritmalar ve kariyer konularında 78 kitap ve makale. Formatlar: PDF, EPUB, AZW3.

---

## Sonraki Adım

Bu pipeline RAG'ın **erişim** (retrieval) yarısı. Hafta 5–6'da bunu bir LLM'e bağlayarak dokümanlarına dayalı **cevap üretme** (generation) öğreneceksin.
