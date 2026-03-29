# Research Monitor — Daily AI/ML Digest

Günlük olarak arXiv, GitHub, Medium/blog RSS feed'leri ve Hacker News'i tarayarak ilgi alanlarına göre filtrelenmiş bir araştırma raporu üreten otomasyon sistemi.

## Takip Edilen Konular

| Konu | Açıklama |
|------|----------|
| Autoresearch | Automated ML research, experiment automation, NAS |
| AI Engineering | LLM ops, RAG, agents, vector DBs, prompt engineering |
| Push/Allocation Optimization | Push notification optimization, bandit algorithms, resource allocation |
| Fraud & Anomaly Detection | Fraud detection, real-time anomaly detection |
| NLP & Text Classification | Text classification, sentiment, transformers, fine-tuning |
| Recommendation Systems | Recommendation, ranking, learning to rank, e-commerce |
| LLM Optimization | Fine-tuning, quantization, LoRA, inference optimization |

## Kaynaklar

- **arXiv API** — cs.AI, cs.LG, cs.CL, cs.IR, cs.CV, stat.ML kategorileri
- **GitHub Search API** — Trending repos, yeni projeler (opsiyonel token ile 5000 req/hr)
- **RSS Feeds** — Towards Data Science, Towards AI, Medium (ML/AI/LLM), Dev.to, Analytics Vidhya, Sebastian Raschka, Lilian Weng, Jay Alammar, Chip Huyen, Eugene Yan, Simon Willison, deeplearning.ai
- **Hacker News API** — Top stories filtered by AI/ML relevance

## Kurulum

```bash
cd research-monitor

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Dependencies
pip install -r requirements.txt

# (Opsiyonel) GitHub token
cp .env.example .env
# .env dosyasına GITHUB_TOKEN= satırını ekle
```

## Kullanım

### Tek seferlik çalıştırma
```bash
python run.py
```

### Parametreler
```bash
# Son 3 günün arXiv makalelerini, 14 günün GitHub repolarını tara
python run.py --arxiv-days 3 --github-days 14

# Sürekli çalışan scheduler (her gün 08:00 UTC'de)
python run.py --schedule --schedule-time 08:00
```

### Cron ile otomatik çalıştırma (macOS/Linux)
```bash
chmod +x setup_cron.sh
./setup_cron.sh
```

## Çıktılar

Her çalıştırmada `reports/` klasörüne iki dosya üretir:

- `reports/digest-YYYY-MM-DD.md` — Markdown rapor
- `reports/digest-YYYY-MM-DD.html` — Tarayıcıda açılabilir dark-theme HTML rapor

Rapor içeriği:
1. arXiv makaleleri (başlık, yazarlar, abstract, PDF linki, eşleşen keywords)
2. GitHub trending repolar (star sayısı, dil, açıklama)
3. Blog yazıları (Medium, TDS, Dev.to, kişisel bloglar)
4. Hacker News hikayeleri (score, yorum sayısı)

## Yapı

```
research-monitor/
├── run.py              ← Entry point
├── src/
│   ├── config.py       ← Tüm ayarlar, konular, keywords, RSS feed listesi
│   ├── arxiv_fetcher.py
│   ├── github_fetcher.py
│   ├── rss_fetcher.py
│   ├── hn_fetcher.py
│   ├── report.py       ← Markdown + HTML report generator
│   └── monitor.py      ← Orchestrator
├── reports/            ← Günlük raporlar (gitignored)
├── data/               ← history.jsonl (gitignored)
├── requirements.txt
├── setup_cron.sh       ← Cron job installer
├── .env.example
└── README.md
```

## Konu/Keyword Ekleme

`src/config.py` dosyasında `TOPICS` dict'ine yeni konu ekleyerek veya mevcut keyword'leri güncelleyerek sistemi özelleştirebilirsin. Yeni RSS feed'leri de `RSS_FEEDS` dict'ine eklenir.

## Gelecek İyileştirmeler

- [ ] E-posta ile rapor gönderme (SMTP)
- [ ] Slack/Discord webhook ile bildirim
- [ ] Semantic similarity ile daha akıllı filtreleme (embeddings)
- [ ] Geçmiş raporlarla karşılaştırma (yeni vs tekrar)
- [ ] Docker container + docker-compose
- [ ] Web dashboard (FastAPI + Streamlit)
