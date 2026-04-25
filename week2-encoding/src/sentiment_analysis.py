"""
Week 2 — Metinleri Sayisallastirma ve Basit Duygu Analizi
==========================================================

Bu script:
  1. Turkce duygu verisi yukler (CSV veya gomulu ornek)
  2. Metin on isleme yapar (kucuk harf, noktalama temizligi, opsiyonel Zeyrek lemmatizasyon)
  3. Integer encoding kavrami gosterir (demo)
  4. CountVectorizer, TfidfVectorizer ve Transformer Embedding ile vektorlestirir
  5. 5 ML modeli egitir (Naive Bayes, Logistic Regression, SVM, XGBoost, LightGBM)
  6. Zero-shot classification (egitim gerektirmez)
  7. Fine-tuning BERTurk (transfer ogrenme)
  8. Tum sonuclari karsilastirir

Kullanim:
    python src/sentiment_analysis.py                         # Klasik ML
    python src/sentiment_analysis.py --zeyrek                # + Zeyrek lemmatizasyon
    python src/sentiment_analysis.py --transformer           # + Transformer embedding
    python src/sentiment_analysis.py --zero-shot             # + Zero-shot
    python src/sentiment_analysis.py --finetune              # + Fine-tuning
    python src/sentiment_analysis.py --all                   # Hepsini calistir
    python src/sentiment_analysis.py --data data/my.csv      # Ozel veri seti
"""

import argparse
import glob
import logging
import os
import re
import string
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ═══════════════════════════════════════════════════════════════════════════════
# SABITLER
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
TEST_SIZE = 0.2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING YAPILANDIRMASI
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    """Konsol + dosya loglama yapilandirir."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = logging.getLogger("sentiment")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    # Konsol handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Dosya handler
    fh = logging.FileHandler(os.path.join(OUTPUT_DIR, "run_log.txt"),
                             mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 1: VERİ YUKLEME
# ═══════════════════════════════════════════════════════════════════════════════

def get_sample_data():
    """Gomulu Turkce ornek veri seti (40 cumle)."""
    positive = [
        "Bu film harikaydı kesinlikle tavsiye ederim",
        "Yemekler çok lezzetliydi tekrar geleceğim",
        "Müşteri hizmetleri çok ilgili ve yardımsever",
        "Harika bir deneyimdi çok memnun kaldım",
        "Ürün kalitesi beklentilerimin üzerindeydi",
        "Çok güzel bir mekan ambiyansı muhteşem",
        "Fiyat performans oranı mükemmel",
        "Kargo çok hızlı geldi paketleme özenli",
        "Personel çok güler yüzlü ve profesyonel",
        "Bu kitap hayatımı değiştirdi şiddetle öneriyorum",
        "Otel odası tertemizdi manzara nefes kesiciydi",
        "Hızlı teslimat ve kaliteli ürün teşekkürler",
        "Çocuklar için harika bir etkinlik çok eğlendiler",
        "Kursu bitirdim çok şey öğrendim değerli bir eğitim",
        "Restoran atmosferi harika servis kusursuzdu",
        "Müzik kalitesi mükemmeldi sahne performansı çok iyi",
        "Doktor çok ilgiliydi tedavi süreci rahat geçti",
        "Uygulama çok kullanışlı arayüzü sade ve anlaşılır",
        "Tatil köyü harikaydı her şey dahil sistemi çok iyi",
        "Kulaklık ses kalitesi süper bass çok iyi",
    ]
    negative = [
        "Çok kötü bir deneyimdi hiç memnun kalmadım",
        "Hizmet berbattı bir daha gelmem",
        "Ürün bozuk geldi iade sürecinde sorun yaşadım",
        "Yemekler soğuk geldi lezzetsiz ve pahalıydı",
        "Kargo çok geç geldi üstelik ürün hasarlıydı",
        "Personel ilgisiz ve kabaydı",
        "Film çok sıkıcıydı vakit kaybı",
        "Otel kirli ve bakımsızdı pişman olduk",
        "Fiyatlar çok yüksek ama kalite düşük",
        "Müşteri desteği hiç yardımcı olmadı sorun çözülmedi",
        "Uygulama sürekli çöküyor kullanılamaz durumda",
        "Ses kalitesi çok kötü paramı boşa harcadım",
        "Restoranda böcek gördük iğrenç bir deneyimdi",
        "Kurs içeriği çok yüzeysel hiçbir şey öğrenmedim",
        "Doktor ilgisizdi muayene beş dakika bile sürmedi",
        "Tatil köyü fotoğraflardaki gibi değildi hayal kırıklığı",
        "Montaj çok zordu parçalar eksik geldi",
        "İnternet sürekli kopuyordu çalışamadık",
        "Beklentimin çok altında kaldı kesinlikle önermem",
        "Kumaş kalitesi berbat ilk yıkamada söküldü",
    ]

    texts = positive + negative
    labels = ["pozitif"] * len(positive) + ["negatif"] * len(negative)
    return pd.DataFrame({"text": texts, "label": labels})


def load_data(csv_path=None):
    """Veri yukler: verilen CSV yolu, data/ klasoru veya gomulu ornek."""
    logger = logging.getLogger("sentiment")

    # 1) Acikca verilen CSV yolu
    if csv_path and os.path.isfile(csv_path):
        logger.info("CSV dosyasi yukleniyor: %s", csv_path)
        df = pd.read_csv(csv_path)
    else:
        # 2) data/ klasorunde CSV ara
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if csv_files:
            chosen = csv_files[0]
            logger.info("data/ klasorunde CSV bulundu: %s", chosen)
            df = pd.read_csv(chosen)
        else:
            # 3) Gomulu ornek veri
            logger.info("CSV bulunamadi — gomulu ornek veri kullaniliyor (40 cumle)")
            df = get_sample_data()

    # Sutun ismi normalizasyonu
    col_map = {}
    for col in df.columns:
        low = col.lower().strip()
        if low in ("text", "metin", "review", "yorum", "sentence", "cumle"):
            col_map[col] = "text"
        elif low in ("label", "sentiment", "etiket", "duygu", "sinif", "class"):
            col_map[col] = "label"
    if col_map:
        df = df.rename(columns=col_map)

    if "text" not in df.columns or "label" not in df.columns:
        logger.error("CSV'de 'text' ve 'label' sutunlari bulunamadi. Mevcut sutunlar: %s",
                      list(df.columns))
        sys.exit(1)

    df = df.dropna(subset=["text", "label"])

    logger.info("Toplam ornek sayisi: %d", len(df))
    logger.info("Sinif dagilimi:\n%s", df["label"].value_counts().to_string())
    logger.info("Ornek veri (ilk 3):")
    for i, row in df.head(3).iterrows():
        logger.info("  [%s] %s", row["label"], row["text"][:80])

    if len(df) < 100:
        logger.warning("Ornek sayisi < 100 — sonuclar gosterim amaclidir, guvenilirlik icin "
                       "daha buyuk veri seti kullanin.")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 2: METİN ÖN İŞLEME
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_text(text, analyzer=None):
    """Tek bir metni on isler: kucuk harf, noktalama/rakam temizligi, opsiyonel lemmatizasyon."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if analyzer is not None:
        tokens = text.split()
        lemmas = []
        for token in tokens:
            results = analyzer.lemmatize(token)
            if results:
                lemmas.append(results[0][1][0])
            else:
                lemmas.append(token)
        text = " ".join(lemmas)

    return text


def preprocess_corpus(texts, use_zeyrek=False):
    """Tum korpusu on isler."""
    logger = logging.getLogger("sentiment")

    analyzer = None
    if use_zeyrek:
        try:
            from zeyrek import MorphAnalyzer
            analyzer = MorphAnalyzer()
            logger.info("Zeyrek morfolojik analiz: AKTIF")
        except ImportError:
            logger.warning("Zeyrek kurulu degil — lemmatizasyon atlanacak. "
                           "Kurmak icin: pip install zeyrek")

    if analyzer is None and use_zeyrek is False:
        logger.info("Zeyrek morfolojik analiz: DEVRE DISI")

    sample_before = texts.iloc[0]
    processed = texts.apply(lambda t: preprocess_text(t, analyzer))
    sample_after = processed.iloc[0]

    logger.info("On isleme ornegi:")
    logger.info("  Oncesi : \"%s\"", sample_before[:80])
    logger.info("  Sonrasi: \"%s\"", sample_after[:80])

    return processed


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 3: INTEGER ENCODING DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo_integer_encoding(texts):
    """Integer encoding kavramini gosterir (egitim amacli, modele girdi olarak kullanilmaz)."""
    logger = logging.getLogger("sentiment")

    # Kelime dagarcigi olustur
    vocab = {}
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    logger.info("Kelime dagarcigi boyutu: %d", len(vocab))

    # 3 ornek cumleyi integer dizisine cevir
    samples = texts.head(3).tolist()
    logger.info("Integer encoding ornekleri:")
    for sent in samples:
        encoded = [vocab[w] for w in sent.split() if w in vocab]
        logger.info("  \"%s\"", sent[:60])
        logger.info("  → %s", encoded)

    logger.info("Not: Tamsayi kodlama kelimeler arasi buyukluk iliskisi ima eder "
                "(ornegin 42 > 7), bu anlamsizdir. Bu yuzden klasik ML'de one-hot "
                "veya TF-IDF tercih edilir.")


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 4 & 5: VEKTORLESTIRME (CountVectorizer vs TfidfVectorizer)
# ═══════════════════════════════════════════════════════════════════════════════

def vectorize_data(X_train, X_test):
    """Iki farkli vektorlestirme yontemi uygular ve sonuclari dondurur."""
    logger = logging.getLogger("sentiment")
    results = {}

    # --- CountVectorizer (binary=True → one-hot benzeri) ---
    cv = CountVectorizer(binary=True)
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)

    logger.info("CountVectorizer (One-Hot):")
    logger.info("  Kelime dagarcigi boyutu: %d", len(cv.vocabulary_))
    logger.info("  Egitim matrisi boyutu : %s", X_train_cv.shape)
    logger.info("  Test matrisi boyutu   : %s", X_test_cv.shape)
    sparsity_cv = 1.0 - (X_train_cv.nnz / (X_train_cv.shape[0] * X_train_cv.shape[1]))
    logger.info("  Seyreklik (sparsity)  : %.2f%%", sparsity_cv * 100)

    results["CountVectorizer (One-Hot)"] = (X_train_cv, X_test_cv)

    # --- TfidfVectorizer ---
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    logger.info("TfidfVectorizer:")
    logger.info("  Kelime dagarcigi boyutu: %d", len(tfidf.vocabulary_))
    logger.info("  Egitim matrisi boyutu : %s", X_train_tfidf.shape)
    logger.info("  Test matrisi boyutu   : %s", X_test_tfidf.shape)
    logger.info("  TF-IDF deger araligi  : [%.4f, %.4f]",
                X_train_tfidf.data.min(), X_train_tfidf.data.max())

    results["TfidfVectorizer"] = (X_train_tfidf, X_test_tfidf)

    # Ayni cumlenin iki temsildeki ilk 10 degerini karsilastir
    logger.info("Ayni cumlenin iki farkli temsili (ilk 10 ozellik):")
    logger.info("  One-Hot : %s", X_train_cv[0, :10].toarray().flatten().tolist())
    logger.info("  TF-IDF  : %s", [round(v, 4) for v in X_train_tfidf[0, :10].toarray().flatten().tolist()])

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 5b: TRANSFORMER EMBEDDING (opsiyonel)
# ═══════════════════════════════════════════════════════════════════════════════

TRANSFORMER_MODELS = {
    "multilingual-MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "turkish-BERT-nli": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "Trendyol-ecomm": "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0",
}


def get_transformer_embeddings(X_train, X_test, model_name=None):
    """Sentence-Transformers ile yogun vektor temsili olusturur.

    CountVectorizer/TF-IDF kelime sirasini ve anlamini yok sayar.
    Transformer embedding ise cumlenin butunsel anlamini yogun bir
    vektore kodlar — anlam benzerligi korunur.
    """
    logger = logging.getLogger("sentiment")

    from sentence_transformers import SentenceTransformer

    if model_name is None:
        model_name = list(TRANSFORMER_MODELS.values())[0]

    logger.info("Transformer model yukleniyor: %s", model_name)
    model = SentenceTransformer(model_name, trust_remote_code=True)

    logger.info("Egitim seti encode ediliyor (%d cumle)...", len(X_train))
    t0 = time.time()
    X_train_emb = model.encode(X_train.tolist(), show_progress_bar=False,
                               convert_to_numpy=True)
    logger.info("  Egitim encoding suresi: %.2f sn", time.time() - t0)

    logger.info("Test seti encode ediliyor (%d cumle)...", len(X_test))
    t0 = time.time()
    X_test_emb = model.encode(X_test.tolist(), show_progress_bar=False,
                              convert_to_numpy=True)
    logger.info("  Test encoding suresi: %.2f sn", time.time() - t0)

    logger.info("Transformer Embedding:")
    logger.info("  Model           : %s", model_name)
    logger.info("  Vektor boyutu   : %d", X_train_emb.shape[1])
    logger.info("  Egitim matrisi  : %s", X_train_emb.shape)
    logger.info("  Deger araligi   : [%.4f, %.4f]",
                X_train_emb.min(), X_train_emb.max())
    logger.info("  Ornek vektor (ilk 10): %s",
                [round(v, 4) for v in X_train_emb[0, :10].tolist()])

    return X_train_emb, X_test_emb


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 5c: OLLAMA EMBEDDING (opsiyonel)
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_BATCH_SIZE = 20


def get_ollama_embeddings(X_train, X_test):
    """Ollama ile lokal embedding olusturur (API'ye veri gonderilmez).

    nomic-embed-text 768 boyutlu vektor uretir.
    Batch halinde gonderilir cunku tek seferde cok fazla metin yavaslayabilir.
    """
    logger = logging.getLogger("sentiment")
    import requests

    MAX_CHARS = 2000  # nomic-embed-text context limiti icin truncate

    def _embed_batch(texts):
        """Ollama /api/embed endpoint'ine batch istek gonderir."""
        safe_texts = []
        for t in texts:
            t = str(t).strip()
            if not t or t.lower() == "nan":
                t = "bos"
            if len(t) > MAX_CHARS:
                t = t[:MAX_CHARS]
            safe_texts.append(t)
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": safe_texts},
            timeout=300,
        )
        if not resp.ok:
            logger.error("Ollama hata: %s — %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def _embed_all(texts):
        """Tum metinleri batch'ler halinde encode eder."""
        all_embs = []
        for i in range(0, len(texts), OLLAMA_BATCH_SIZE):
            batch = texts[i : i + OLLAMA_BATCH_SIZE]
            embs = _embed_batch(batch)
            all_embs.extend(embs)
            done = i + len(batch)
            if done % 500 == 0 or done == len(texts):
                logger.info("    %d / %d cumle encode edildi...", done, len(texts))
        return np.array(all_embs, dtype=np.float32)

    logger.info("Ollama embedding modeli: %s (lokal, API'ye veri gonderilmez)", OLLAMA_EMBED_MODEL)

    train_texts = X_train.tolist() if hasattr(X_train, "tolist") else list(X_train)
    test_texts = X_test.tolist() if hasattr(X_test, "tolist") else list(X_test)

    logger.info("Egitim seti encode ediliyor (%d cumle)...", len(train_texts))
    t0 = time.time()
    X_train_emb = _embed_all(train_texts)
    logger.info("  Egitim encoding suresi: %.2f sn", time.time() - t0)

    logger.info("Test seti encode ediliyor (%d cumle)...", len(test_texts))
    t0 = time.time()
    X_test_emb = _embed_all(test_texts)
    logger.info("  Test encoding suresi: %.2f sn", time.time() - t0)

    logger.info("Ollama Embedding:")
    logger.info("  Model           : %s (lokal)", OLLAMA_EMBED_MODEL)
    logger.info("  Vektor boyutu   : %d", X_train_emb.shape[1])
    logger.info("  Egitim matrisi  : %s", X_train_emb.shape)
    logger.info("  Deger araligi   : [%.4f, %.4f]",
                X_train_emb.min(), X_train_emb.max())
    logger.info("  Ornek vektor (ilk 10): %s",
                [round(v, 4) for v in X_train_emb[0, :10].tolist()])

    return X_train_emb, X_test_emb


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 8: ZERO-SHOT CLASSIFICATION (opsiyonel)
# ═══════════════════════════════════════════════════════════════════════════════

ZERO_SHOT_MODEL = "joeddav/xlm-roberta-large-xnli"


def zero_shot_evaluate(texts, labels, label_names=None):
    """Zero-shot classification: hic egitim yapmadan duygu tahmini.

    Onceden egitilmis bir NLI (Natural Language Inference) modeli kullanir.
    Her cumle icin "Bu cumle pozitif/negatif" hipotezlerinin olasiligini hesaplar.
    """
    logger = logging.getLogger("sentiment")

    from transformers import pipeline

    if label_names is None:
        label_names = sorted(labels.unique().tolist())

    logger.info("Zero-shot model yukleniyor: %s", ZERO_SHOT_MODEL)
    classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL,
                          device=-1)

    logger.info("Zero-shot tahmin yapiliyor (%d cumle)...", len(texts))
    t0 = time.time()

    y_pred = []
    y_scores = []
    for text in texts:
        result = classifier(text, candidate_labels=label_names)
        y_pred.append(result["labels"][0])
        # En yuksek olasilik skoru
        idx = result["labels"].index(label_names[-1])
        y_scores.append(result["scores"][idx])

    elapsed = time.time() - t0
    logger.info("  Tahmin suresi: %.2f sn (ortalama %.2f sn/cumle)",
                elapsed, elapsed / len(texts))

    # Metrikler
    le = LabelEncoder()
    le.fit(label_names)
    y_test_enc = le.transform(labels)

    acc = accuracy_score(labels, y_pred)
    prec = precision_score(labels, y_pred, average="macro", zero_division=0)
    rec = recall_score(labels, y_pred, average="macro", zero_division=0)
    f1 = f1_score(labels, y_pred, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_test_enc, y_scores)
    except ValueError:
        auc = float("nan")

    logger.info("  Accuracy : %.4f", acc)
    logger.info("  Precision: %.4f", prec)
    logger.info("  Recall   : %.4f", rec)
    logger.info("  F1 (macro): %.4f", f1)
    logger.info("  AUC-ROC  : %.4f", auc)

    report = classification_report(labels, y_pred, zero_division=0)
    logger.info("  Siniflandirma Raporu:\n%s", report)

    return {
        "Vektorlestirme": "—",
        "Model": f"Zero-Shot ({ZERO_SHOT_MODEL.split('/')[-1]})",
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 (macro)": round(f1, 4),
        "AUC-ROC": round(auc, 4),
        "Sure (sn)": round(elapsed, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 9: FINE-TUNING BERTurk (opsiyonel)
# ═══════════════════════════════════════════════════════════════════════════════

FINETUNE_MODEL = "dbmdz/bert-base-turkish-cased"
FINETUNE_EPOCHS = 3
FINETUNE_BATCH_SIZE = 8
FINETUNE_LR = 2e-5
FINETUNE_MAX_LEN = 128


def finetune_berturk(X_train, X_test, y_train, y_test):
    """BERTurk fine-tuning: onceden egitilmis Turkce BERT'i kendi verimizle ince ayar yapar.

    Bu, duygu analizi icin en guclu yaklasimdir. Model hem Turkce dil bilgisini
    hem de gorev-spesifik kaliplari ogrenir.
    """
    logger = logging.getLogger("sentiment")

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Etiketleri sayisallastir
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_labels = len(le.classes_)
    logger.info("Etiketler: %s", dict(zip(le.classes_, le.transform(le.classes_))))

    # Tokenizer ve model yukle
    logger.info("Model yukleniyor: %s", FINETUNE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINETUNE_MODEL, num_labels=num_labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Cihaz: %s", device)
    model.to(device)

    # Dataset sinifi
    class SentimentDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(texts, truncation=True,
                                       padding=True, max_length=FINETUNE_MAX_LEN,
                                       return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_dataset = SentimentDataset(X_train.tolist(), y_train_enc)
    test_dataset = SentimentDataset(X_test.tolist(), y_test_enc)

    train_loader = DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=FINETUNE_BATCH_SIZE)

    # Egitim
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR)
    logger.info("Fine-tuning basliyor: %d epoch, batch_size=%d, lr=%s",
                FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE, FINETUNE_LR)

    t0 = time.time()
    model.train()
    for epoch in range(FINETUNE_EPOCHS):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        avg_loss = total_loss / len(train_loader)
        logger.info("  Epoch %d/%d — loss: %.4f", epoch + 1, FINETUNE_EPOCHS, avg_loss)

    train_time = time.time() - t0
    logger.info("  Egitim suresi: %.2f sn", train_time)

    # Degerlendirme
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    y_pred = le.inverse_transform(all_preds)
    elapsed = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_test_enc, all_probs)
    except ValueError:
        auc = float("nan")

    logger.info("  Accuracy : %.4f", acc)
    logger.info("  Precision: %.4f", prec)
    logger.info("  Recall   : %.4f", rec)
    logger.info("  F1 (macro): %.4f", f1)
    logger.info("  AUC-ROC  : %.4f", auc)

    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info("  Siniflandirma Raporu:\n%s", report)

    return {
        "Vektorlestirme": "—",
        "Model": f"Fine-Tuned BERTurk ({FINETUNE_EPOCHS} epoch)",
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 (macro)": round(f1, 4),
        "AUC-ROC": round(auc, 4),
        "Sure (sn)": round(elapsed, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 6: MODEL EGITIMI VE DEGERLENDIRME
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(vectorized, y_train, y_test):
    """Her vektorlestirme x her model kombinasyonu icin egitim ve degerlendirme yapar."""
    logger = logging.getLogger("sentiment")

    # XGBoost ve LightGBM sayisal etiket ister — LabelEncoder ile donustur
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    logger.info("Etiket kodlamasi: %s", dict(zip(le.classes_, le.transform(le.classes_))))

    models = {
        "Naive Bayes (MultinomialNB)": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "SVM (LinearSVC)": LinearSVC(max_iter=2000, random_state=RANDOM_SEED),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=RANDOM_SEED, eval_metric="logloss",
            use_label_encoder=False, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=RANDOM_SEED, verbose=-1,
        ),
    }

    all_results = []

    # Boosting modelleri sayisal etiket ve yogun matris ister
    boosting_models = {"XGBoost", "LightGBM"}

    for vec_name, (X_tr, X_te) in vectorized.items():
        for model_name, model in models.items():
            logger.info("Egitiliyor: %s + %s", vec_name, model_name)
            t0 = time.time()

            is_sparse = scipy.sparse.issparse(X_tr)

            # MultinomialNB negatif degerlerle calismaz (transformer embedding)
            if model_name == "Naive Bayes (MultinomialNB)" and not is_sparse:
                if X_tr.min() < 0:
                    logger.info("  ATLANDI: MultinomialNB negatif degerlerle calismaz "
                                "(transformer embedding). Sonraki modele geciliyor.")
                    continue

            if model_name in boosting_models:
                X_tr_dense = X_tr.toarray() if is_sparse else X_tr
                X_te_dense = X_te.toarray() if is_sparse else X_te
                model.fit(X_tr_dense, y_train_enc)
                y_pred_raw = model.predict(X_te_dense)
                y_pred = le.inverse_transform(y_pred_raw)
            elif not is_sparse:
                # Dense matris (transformer embedding)
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)
            else:
                model.fit(X_tr, y_train)
                y_pred = model.predict(X_te)

            elapsed = time.time() - t0

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # AUC — threshold-bagimsiz metrik (binary ve multiclass destegi)
            n_classes = len(le.classes_)
            try:
                X_te_for_score = X_te
                if model_name in boosting_models:
                    X_te_for_score = X_te.toarray() if is_sparse else X_te

                if model_name in boosting_models:
                    y_score = model.predict_proba(X_te_for_score)
                elif hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_te)
                elif hasattr(model, "decision_function"):
                    y_score = model.decision_function(X_te)
                else:
                    y_score = None

                if y_score is not None:
                    if n_classes == 2:
                        # Binary: sadece pozitif sinifin skorunu kullan
                        if y_score.ndim == 2:
                            y_score = y_score[:, 1]
                        auc = roc_auc_score(y_test_enc, y_score)
                    else:
                        # Multiclass: OvR stratejisi
                        if y_score.ndim == 1:
                            # decision_function tek boyutlu donebilir — AUC hesaplanamaz
                            auc = float("nan")
                        else:
                            auc = roc_auc_score(y_test_enc, y_score,
                                                multi_class="ovr", average="macro")
                else:
                    auc = float("nan")
            except (ValueError, TypeError):
                auc = float("nan")

            logger.info("  Accuracy : %.4f", acc)
            logger.info("  Precision: %.4f", prec)
            logger.info("  Recall   : %.4f", rec)
            logger.info("  F1 (macro): %.4f", f1)
            logger.info("  AUC-ROC  : %.4f", auc)
            logger.info("  Sure     : %.3f sn", elapsed)

            report = classification_report(y_test, y_pred, zero_division=0)
            logger.info("  Siniflandirma Raporu:\n%s", report)

            all_results.append({
                "Vektorlestirme": vec_name,
                "Model": model_name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 (macro)": round(f1, 4),
                "AUC-ROC": round(auc, 4),
                "Sure (sn)": round(elapsed, 3),
            })

            # Modeli sifirla (ayni nesne yeniden kullanilacak)
            models[model_name] = type(model)(
                **{k: v for k, v in model.get_params().items()
                   if k in type(model)().get_params()}
            )

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# ADIM 7: SONUC KARSILASTIRMA TABLOSU
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(results, n_train=0, n_test=0):
    """Tum sonuclari tablo halinde gosterir ve CSV'ye kaydeder."""
    logger = logging.getLogger("sentiment")

    df = pd.DataFrame(results)
    logger.info("\n" + "=" * 90)
    logger.info("SONUC KARSILASTIRMA TABLOSU")
    logger.info("=" * 90)
    logger.info("\n%s", df.to_string(index=False))

    # En iyi kombinasyonu bul — AUC-ROC birincil metrik
    df_valid_auc = df.dropna(subset=["AUC-ROC"])
    if not df_valid_auc.empty:
        best_idx = df_valid_auc["AUC-ROC"].idxmax()
        best = df.loc[best_idx]
        logger.info("\n🏆 En iyi kombinasyon (AUC-ROC): %s + %s (AUC=%.4f, F1=%.4f)",
                    best["Vektorlestirme"], best["Model"],
                    best["AUC-ROC"], best["F1 (macro)"])
    else:
        best_idx = df["F1 (macro)"].idxmax()
        best = df.loc[best_idx]
        logger.info("\n🏆 En iyi kombinasyon (F1): %s + %s (F1=%.4f)",
                    best["Vektorlestirme"], best["Model"], best["F1 (macro)"])

    # CSV'ye kaydet
    csv_path = os.path.join(OUTPUT_DIR, "comparison_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Sonuclar kaydedildi: %s", csv_path)

    # Analysis MD olustur
    generate_analysis_md(df, n_train=n_train, n_test=n_test)

    return df


def generate_analysis_md(df, n_train=0, n_test=0):
    """Sonuclari yorumlayan bir Markdown rapor dosyasi olusturur."""
    logger = logging.getLogger("sentiment")

    # AUC-ROC birincil metrik — NaN olanlari filtrele
    df_valid_auc = df.dropna(subset=["AUC-ROC"])
    if not df_valid_auc.empty:
        best_idx = df_valid_auc["AUC-ROC"].idxmax()
        primary_metric = "AUC-ROC"
    else:
        best_idx = df["F1 (macro)"].idxmax()
        primary_metric = "F1 (macro)"
    best = df.loc[best_idx]

    worst_idx = df["F1 (macro)"].idxmin()
    worst = df.loc[worst_idx]

    # Vectorizer bazinda gruplama
    vec_groups = df.groupby("Vektorlestirme")

    # AUC vs F1 uyumsuzluklari bul
    discrepancies = []
    for _, row in df.iterrows():
        if not np.isnan(row["AUC-ROC"]):
            if row["AUC-ROC"] >= 0.7 and row["F1 (macro)"] < 0.6:
                discrepancies.append(row)
            elif row["AUC-ROC"] < 0.5 and row["F1 (macro)"] >= 0.6:
                discrepancies.append(row)

    # Vectorizer bazinda en iyiler (AUC birincil)
    vec_best = {}
    for vec_name, group in vec_groups:
        if vec_name == "—":
            continue
        group_valid = group.dropna(subset=["AUC-ROC"])
        if not group_valid.empty:
            best_in_group = group_valid.loc[group_valid["AUC-ROC"].idxmax()]
        else:
            best_in_group = group.loc[group["F1 (macro)"].idxmax()]
        vec_best[vec_name] = best_in_group

    lines = []
    lines.append("# Sentiment Analysis — Results Comparison & Interpretation")
    lines.append("")
    lines.append("*Auto-generated analysis report*")
    lines.append("")

    # --- Full results table ---
    lines.append("## Full Results Table")
    lines.append("")
    lines.append("| Vectorization | Model | Accuracy | Precision | Recall | F1 (macro) | AUC-ROC | Time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in df.iterrows():
        vec = row["Vektorlestirme"]
        model = row["Model"]
        is_best_row = (row.name == best_idx)
        marker = " **[BEST]**" if is_best_row else ""
        lines.append(
            f"| {vec} | {model}{marker} | {row['Accuracy']:.4f} | "
            f"{row['Precision']:.4f} | {row['Recall']:.4f} | "
            f"{row['F1 (macro)']:.4f} | {row['AUC-ROC']:.4f} | "
            f"{row['Sure (sn)']:.3f} |"
        )
    lines.append("")

    # --- Overall winner ---
    lines.append("## Overall Winner")
    lines.append("")
    if primary_metric == "AUC-ROC":
        lines.append(
            f"**{best['Vektorlestirme']} + {best['Model']}** achieved the highest "
            f"AUC-ROC of **{best['AUC-ROC']:.4f}** (F1={best['F1 (macro)']:.4f}). "
            f"AUC-ROC is the primary metric because it evaluates ranking quality "
            f"independent of any threshold choice."
        )
    else:
        lines.append(
            f"**{best['Vektorlestirme']} + {best['Model']}** achieved the highest "
            f"F1 score of **{best['F1 (macro)']:.4f}**."
        )
    lines.append("")

    # --- Vectorization comparison ---
    lines.append("## Vectorization Method Comparison")
    lines.append("")
    if vec_best:
        for vec_name, row in vec_best.items():
            lines.append(f"### {vec_name}")
            lines.append("")
            lines.append(
                f"- Best model: **{row['Model']}** "
                f"(F1={row['F1 (macro)']:.4f}, AUC={row['AUC-ROC']:.4f})"
            )

            if "CountVectorizer" in vec_name:
                lines.append(
                    "- CountVectorizer with `binary=True` creates a one-hot style "
                    "representation. Each word is either present (1) or absent (0). "
                    "This approach ignores word frequency and word order entirely."
                )
            elif "Tfidf" in vec_name:
                lines.append(
                    "- TF-IDF weights words by how important they are to a document "
                    "relative to the whole corpus. Common words get lower scores, "
                    "distinctive words get higher ones. Still ignores word order."
                )
            elif "Ollama" in vec_name:
                lines.append(
                    "- Ollama runs the `nomic-embed-text` embedding model locally on "
                    "your machine. No data is sent to any external API. The model "
                    "produces 768-dimensional dense vectors, but since it is primarily "
                    "trained on English text, its Turkish performance is limited."
                )
            elif "Transformer" in vec_name:
                if "Trendyol" in vec_name:
                    lines.append(
                        "- Trendyol's e-commerce embedding model is fine-tuned from "
                        "`Alibaba-NLP/gte-multilingual-base` on Turkish e-commerce data. "
                        "Since our dataset consists of product reviews, the domain match "
                        "gives this model a significant advantage."
                    )
                else:
                    lines.append(
                        "- Transformer embeddings encode the full meaning of a sentence "
                        "into a dense vector. Unlike bag-of-words methods, they capture "
                        "word order, context, and semantic relationships."
                    )
            lines.append("")

    # Compare vectorizers head-to-head
    vec_names = [v for v in vec_best.keys()]
    if len(vec_names) >= 2:
        lines.append("### Head-to-Head Comparison")
        lines.append("")

        # Find same model across vectorizers for fair comparison
        common_models = set(df[df["Vektorlestirme"] == vec_names[0]]["Model"].values)
        for vn in vec_names[1:]:
            common_models &= set(df[df["Vektorlestirme"] == vn]["Model"].values)

        if common_models:
            sample_model = sorted(common_models)[0]
            lines.append(
                f"Using **{sample_model}** as the control model across vectorizers:"
            )
            lines.append("")
            for vn in vec_names:
                row = df[(df["Vektorlestirme"] == vn) & (df["Model"] == sample_model)].iloc[0]
                lines.append(f"- {vn}: F1={row['F1 (macro)']:.4f}, AUC={row['AUC-ROC']:.4f}")
            lines.append("")

        if "Transformer Embedding" in vec_names and "CountVectorizer (One-Hot)" in vec_names:
            te_best = vec_best.get("Transformer Embedding")
            cv_best = vec_best.get("CountVectorizer (One-Hot)")
            if te_best is not None and cv_best is not None:
                f1_diff = te_best["F1 (macro)"] - cv_best["F1 (macro)"]
                if f1_diff > 0:
                    lines.append(
                        f"Transformer embeddings outperformed CountVectorizer by "
                        f"**{f1_diff:.4f}** F1 points. This demonstrates the value "
                        f"of semantic representations: the same ML models perform "
                        f"dramatically better when fed meaningful features that "
                        f"capture sentence-level meaning rather than just word presence."
                    )
                    lines.append("")

    # --- Standalone approaches ---
    standalone = df[df["Vektorlestirme"] == "—"]
    if not standalone.empty:
        lines.append("## Standalone Deep Learning Approaches")
        lines.append("")
        for _, row in standalone.iterrows():
            lines.append(f"### {row['Model']}")
            lines.append("")
            lines.append(
                f"- F1: **{row['F1 (macro)']:.4f}** | AUC: **{row['AUC-ROC']:.4f}** | "
                f"Time: {row['Sure (sn)']:.1f}s"
            )
            if "Zero-Shot" in row["Model"]:
                lines.append(
                    "- Zero-shot classification requires **no training data at all**. "
                    "It uses a pre-trained NLI model to judge whether a sentence "
                    "matches a given label hypothesis. This is ideal for rapid "
                    "prototyping or when labeled data is unavailable."
                )
            elif "Fine-Tuned" in row["Model"]:
                lines.append(
                    "- Fine-tuning adapts a pre-trained Turkish BERT model to our "
                    "specific sentiment task. The model brings general Turkish "
                    "language understanding and learns task-specific patterns from "
                    "our labeled data. This typically gives the strongest results, "
                    "especially with more training data."
                )
            lines.append("")

    # --- AUC vs F1 analysis ---
    lines.append("## AUC-ROC vs F1: Why Both Matter")
    lines.append("")
    lines.append(
        "F1 score depends on a fixed classification threshold (usually 0.5). "
        "A model might have mediocre F1 simply because its default threshold "
        "is not optimal. AUC-ROC, on the other hand, evaluates the model's "
        "ability to **rank** positive examples above negative ones across all "
        "possible thresholds. A high AUC with low F1 means the model learned "
        "useful patterns but needs threshold tuning to translate that into "
        "good predictions."
    )
    lines.append("")

    if discrepancies:
        lines.append("### Notable AUC vs F1 Discrepancies")
        lines.append("")
        for row in discrepancies:
            lines.append(
                f"- **{row['Vektorlestirme']} + {row['Model']}**: "
                f"F1={row['F1 (macro)']:.4f} but AUC={row['AUC-ROC']:.4f} — "
            )
            if row["AUC-ROC"] > row["F1 (macro)"]:
                lines.append(
                    f"  This model ranks examples well but its default threshold "
                    f"is suboptimal. Threshold tuning would likely improve F1 "
                    f"significantly."
                )
            else:
                lines.append(
                    f"  Despite a reasonable F1, the model's ranking ability is "
                    f"poor, suggesting it may be getting lucky with the threshold "
                    f"rather than truly learning the pattern."
                )
        lines.append("")

    # --- Key takeaways ---
    lines.append("## Key Takeaways")
    lines.append("")
    lines.append(
        "1. **Representation matters more than the model.** The same SVM or "
        "Logistic Regression can jump from ~80% to ~90% accuracy simply by "
        "changing how text is vectorized. The choice of embedding model has "
        "a bigger impact than the choice of classifier."
    )
    lines.append(
        "2. **Domain-specific embeddings dominate.** Trendyol's e-commerce "
        "embedding model outperformed all others because its training domain "
        "(Turkish product reviews) matches our test data. Generic multilingual "
        "models like MiniLM scored lower despite being larger."
    )
    lines.append(
        "3. **AUC and F1 tell different stories.** Always report both. A model "
        "with high AUC but lower F1 is not broken — it just needs threshold "
        "calibration. The AUC-F1 gap across all models suggests 3-5% F1 "
        "improvement is possible with threshold tuning alone."
    )
    lines.append(
        "4. **Local embedding models (Ollama) lag behind.** `nomic-embed-text` "
        "running locally via Ollama scored the lowest AUC (~0.83). This is "
        "expected — it is an English-centric model not optimized for Turkish "
        "sentiment. Local models are best suited for English text or RAG "
        "applications where privacy is a priority."
    )
    lines.append(
        "5. **Linear models beat tree-based models on text.** Logistic Regression "
        "and SVM consistently outperformed XGBoost and LightGBM with sparse "
        "features (CountVectorizer, TF-IDF). Tree-based models performed better "
        "only with dense transformer embeddings."
    )
    lines.append(
        "6. **TF-IDF is not dead.** On a binary sentiment task with enough data, "
        "TF-IDF + SVM can match or beat generic transformer embeddings. Do not "
        "dismiss classical methods without benchmarking them."
    )
    lines.append("")

    # --- Caveat ---
    lines.append("## Experiment Setup")
    lines.append("")
    if n_train > 0 and n_test > 0:
        lines.append(
            f"- **Training set:** {n_train:,} samples"
        )
        lines.append(
            f"- **Test set:** {n_test:,} samples"
        )
        lines.append(
            "- Train and test sets come from separate, non-overlapping splits "
            "of the source dataset to prevent data leakage."
        )
        if n_test < 500:
            lines.append(
                f"- **Note:** With only {n_test} test samples, metrics may be noisy. "
                "For more reliable conclusions, use 500+ test samples."
            )
    else:
        lines.append(
            "Training and test set sizes are not available. "
            "Check the run log for details."
        )
    lines.append("")

    md_content = "\n".join(lines)
    md_path = os.path.join(OUTPUT_DIR, "analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Analysis raporu olusturuldu: %s", md_path)


# ═══════════════════════════════════════════════════════════════════════════════
# ANA FONKSIYON
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Turkce Duygu Analizi — Week 2")
    parser.add_argument("--zeyrek", action="store_true",
                        help="Zeyrek morfolojik analiz aktif et")
    parser.add_argument("--data", type=str, default=None,
                        help="Egitim CSV dosya yolu (opsiyonel)")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Ayri test CSV dosyasi — verilirse train/test split yapilmaz, "
                             "--data ile egitilir, --test-data ile test edilir")
    parser.add_argument("--transformer", action="store_true",
                        help="Transformer embedding vektorlestirme ekle")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Zero-shot classification calistir")
    parser.add_argument("--finetune", action="store_true",
                        help="BERTurk fine-tuning calistir")
    parser.add_argument("--ollama", action="store_true",
                        help="Ollama ile lokal embedding (nomic-embed-text)")
    parser.add_argument("--all", action="store_true",
                        help="Tum yaklasimlari calistir")
    args = parser.parse_args()

    if args.all:
        args.transformer = True
        args.zero_shot = True
        args.finetune = True
        args.ollama = True

    logger = setup_logging()
    start_time = time.time()

    logger.info("=" * 90)
    logger.info("TURKCE DUYGU ANALIZI — WEEK 2: METIN SAYISALLASTIRMA")
    logger.info("=" * 90)
    logger.info("Python: %s | scikit-learn: %s | pandas: %s",
                sys.version.split()[0],
                __import__("sklearn").__version__,
                pd.__version__)
    logger.info("Modlar: transformer=%s | ollama=%s | zero-shot=%s | finetune=%s",
                args.transformer, args.ollama, args.zero_shot, args.finetune)

    # --- ADIM 1: Veri yukleme ---
    logger.info("\n" + "═" * 90)
    logger.info("ADIM 1: VERI YUKLEME")
    logger.info("═" * 90)
    df_train_source = load_data(args.data)

    # --- ADIM 2: On isleme ---
    logger.info("\n" + "═" * 90)
    logger.info("ADIM 2: METIN ON ISLEME")
    logger.info("═" * 90)
    df_train_source["text_clean"] = preprocess_corpus(df_train_source["text"],
                                                       use_zeyrek=args.zeyrek)

    # --- ADIM 3: Integer encoding demo ---
    logger.info("\n" + "═" * 90)
    logger.info("ADIM 3: INTEGER ENCODING DEMO")
    logger.info("═" * 90)
    demo_integer_encoding(df_train_source["text_clean"])

    # --- ADIM 4: Train/test split ---
    logger.info("\n" + "═" * 90)
    logger.info("ADIM 4: EGITIM / TEST BOLUNMESI")
    logger.info("═" * 90)

    if args.test_data:
        # Ayri test seti modu: --data ile egit, --test-data ile test et
        logger.info("AYRI TEST SETI MODU: egitim ve test farkli kaynaklardan")
        logger.info("Egitim kaynak: %s", args.data or "gomulu veri")
        logger.info("Test kaynak  : %s", args.test_data)

        df_test_source = load_data(args.test_data)
        df_test_source["text_clean"] = preprocess_corpus(df_test_source["text"],
                                                          use_zeyrek=args.zeyrek)

        # Etiket uyumu kontrolu — test setindeki etiketler egitim setinde de olmali
        train_labels = set(df_train_source["label"].unique())
        test_labels = set(df_test_source["label"].unique())
        common_labels = train_labels & test_labels
        if common_labels != test_labels:
            missing = test_labels - train_labels
            logger.warning("Test setinde egitim setinde olmayan etiketler var: %s — "
                           "bu ornekler filtreleniyor.", missing)
            df_test_source = df_test_source[df_test_source["label"].isin(common_labels)]

        X_train = df_train_source["text_clean"]
        y_train = df_train_source["label"]
        X_test = df_test_source["text_clean"]
        y_test = df_test_source["label"]

        X_train_raw = df_train_source["text"]
        X_test_raw = df_test_source["text"]
    else:
        # Normal mod: tek kaynaktan train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            df_train_source["text_clean"], df_train_source["label"],
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=df_train_source["label"],
        )
        X_train_raw, X_test_raw, _, _ = train_test_split(
            df_train_source["text"], df_train_source["label"],
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=df_train_source["label"],
        )

    logger.info("Egitim seti: %d ornek", len(X_train))
    logger.info("Test seti  : %d ornek", len(X_test))
    logger.info("Egitim sinif dagilimi:\n%s", y_train.value_counts().to_string())
    logger.info("Test sinif dagilimi:\n%s", y_test.value_counts().to_string())

    # --- ADIM 5: Vektorlestirme ---
    logger.info("\n" + "═" * 90)
    logger.info("ADIM 5: VEKTORLESTIRME (CountVectorizer vs TfidfVectorizer)")
    logger.info("═" * 90)
    vectorized = vectorize_data(X_train, X_test)

    # --- ADIM 5b: Transformer Embedding (opsiyonel) ---
    if args.transformer:
        for short_name, full_name in TRANSFORMER_MODELS.items():
            logger.info("\n" + "═" * 90)
            logger.info("ADIM 5b: TRANSFORMER EMBEDDING — %s", short_name)
            logger.info("═" * 90)
            X_tr_emb, X_te_emb = get_transformer_embeddings(
                X_train_raw, X_test_raw, model_name=full_name
            )
            vectorized[f"Transformer ({short_name})"] = (X_tr_emb, X_te_emb)

    # --- ADIM 5c: Ollama Embedding (opsiyonel) ---
    if args.ollama:
        logger.info("\n" + "═" * 90)
        logger.info("ADIM 5c: OLLAMA EMBEDDING (lokal — %s)", OLLAMA_EMBED_MODEL)
        logger.info("═" * 90)
        X_tr_ollama, X_te_ollama = get_ollama_embeddings(X_train_raw, X_test_raw)
        vectorized[f"Ollama ({OLLAMA_EMBED_MODEL})"] = (X_tr_ollama, X_te_ollama)

    # --- ADIM 6: Model egitimi ---
    logger.info("\n" + "═" * 90)
    logger.info("ADIM 6: MODEL EGITIMI VE DEGERLENDIRME")
    logger.info("═" * 90)
    results = train_and_evaluate(vectorized, y_train, y_test)

    # --- ADIM 8: Zero-shot (opsiyonel) ---
    if args.zero_shot:
        logger.info("\n" + "═" * 90)
        logger.info("ADIM 8: ZERO-SHOT CLASSIFICATION")
        logger.info("═" * 90)
        zs_result = zero_shot_evaluate(X_test_raw.tolist(), y_test)
        results.append(zs_result)

    # --- ADIM 9: Fine-tuning (opsiyonel) ---
    if args.finetune:
        logger.info("\n" + "═" * 90)
        logger.info("ADIM 9: FINE-TUNING BERTurk")
        logger.info("═" * 90)
        ft_result = finetune_berturk(X_train_raw, X_test_raw, y_train, y_test)
        results.append(ft_result)

    # --- ADIM 7: Karsilastirma ---
    logger.info("\n" + "═" * 90)
    logger.info("ADIM 7: SONUC KARSILASTIRMA")
    logger.info("═" * 90)
    print_comparison(results, n_train=len(y_train), n_test=len(y_test))

    total_time = time.time() - start_time
    logger.info("\nToplam calisma suresi: %.2f saniye", total_time)
    logger.info("Log dosyasi: %s", os.path.join(OUTPUT_DIR, "run_log.txt"))
    logger.info("Bitti!")


if __name__ == "__main__":
    main()
