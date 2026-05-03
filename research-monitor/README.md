# Research Monitor — Daily AI/ML Digest

An automation system that scans arXiv, GitHub, Medium/blog RSS feeds, and Hacker News every day, then produces a research report filtered by your topics of interest.

## Tracked Topics

| Topic | Description |
|------|----------|
| Autoresearch | Automated ML research, experiment automation, NAS |
| AI Engineering | LLM ops, RAG, agents, vector DBs, prompt engineering |
| Push/Allocation Optimization | Push notification optimization, bandit algorithms, resource allocation |
| Fraud & Anomaly Detection | Fraud detection, real-time anomaly detection |
| NLP & Text Classification | Text classification, sentiment, transformers, fine-tuning |
| Recommendation Systems | Recommendation, ranking, learning to rank, e-commerce |
| LLM Optimization | Fine-tuning, quantization, LoRA, inference optimization |

## Sources

- **arXiv API** — cs.AI, cs.LG, cs.CL, cs.IR, cs.CV, stat.ML categories
- **GitHub Search API** — trending repos, new projects (optional token gives 5000 req/hr)
- **RSS Feeds** — Towards Data Science, Towards AI, Medium (ML/AI/LLM), Dev.to, Analytics Vidhya, Sebastian Raschka, Lilian Weng, Jay Alammar, Chip Huyen, Eugene Yan, Simon Willison, deeplearning.ai
- **Hacker News API** — top stories filtered by AI/ML relevance

## Setup

```bash
cd research-monitor

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Dependencies
pip install -r requirements.txt

# (Optional) GitHub token
cp .env.example .env
# Add a GITHUB_TOKEN= line in the .env file
```

## Usage

### One-off run
```bash
python run.py
```

### Parameters
```bash
# Scan the last 3 days of arXiv papers and 14 days of GitHub repos
python run.py --arxiv-days 3 --github-days 14

# Long-running scheduler (every day at 08:00 UTC)
python run.py --schedule --schedule-time 08:00
```

### Automatic runs via cron (macOS/Linux)
```bash
chmod +x setup_cron.sh
./setup_cron.sh
```

## Outputs

Every run produces two files under `reports/`:

- `reports/digest-YYYY-MM-DD.md` — Markdown report
- `reports/digest-YYYY-MM-DD.html` — dark-theme HTML report (opens in any browser)

Report contents:
1. arXiv papers (title, authors, abstract, PDF link, matched keywords)
2. GitHub trending repos (stars, language, description)
3. Blog posts (Medium, TDS, Dev.to, personal blogs)
4. Hacker News stories (score, comment count)

## Layout

```
research-monitor/
├── run.py              ← Entry point
├── src/
│   ├── config.py       ← All settings: topics, keywords, RSS feed list
│   ├── arxiv_fetcher.py
│   ├── github_fetcher.py
│   ├── rss_fetcher.py
│   ├── hn_fetcher.py
│   ├── report.py       ← Markdown + HTML report generator
│   └── monitor.py      ← Orchestrator
├── reports/            ← Daily reports (gitignored)
├── data/               ← history.jsonl (gitignored)
├── requirements.txt
├── setup_cron.sh       ← Cron job installer
├── .env.example
└── README.md
```

## Adding Topics or Keywords

You can customize the system by editing the `TOPICS` dict in `src/config.py` — add a new topic or update the keywords of an existing one. New RSS feeds go into the `RSS_FEEDS` dict in the same file.

## Planned Improvements

- [ ] Email delivery (SMTP)
- [ ] Slack / Discord webhook notifications
- [ ] Smarter filtering with semantic similarity (embeddings)
- [ ] Diff against previous reports (new vs. repeated items)
- [ ] Docker container + docker-compose
- [ ] Web dashboard (FastAPI + Streamlit)
