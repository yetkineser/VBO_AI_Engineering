"""
Central configuration for Research Monitor.
All topics, keywords, and settings live here.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # GitHub personal access token (optional, increases rate limit from 60 to 5000/hr)
    github_token: str = Field(default="", alias="GITHUB_TOKEN")

    # Report output directory
    reports_dir: str = "reports"
    data_dir: str = "data"

    # Max results per source
    arxiv_max_results: int = 30
    github_max_repos: int = 20
    rss_max_per_feed: int = 15
    hn_max_stories: int = 20

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


TOPICS = {
    "autoresearch": {
        "description": "Automated ML research / experiment automation",
        "arxiv_queries": [
            "automated machine learning research",
            "autonomous ML experimentation",
            "neural architecture search",
            "hyperparameter optimization automation",
            "LLM-driven research agents",
        ],
        "github_keywords": ["autoresearch", "automated-ml-research", "automl", "neural-architecture-search"],
        "rss_keywords": ["autoresearch", "automated ML", "automl", "neural architecture search"],
    },
    "ai_engineering": {
        "description": "Production AI systems, LLM ops, RAG, agents",
        "arxiv_queries": [
            "retrieval augmented generation",
            "LLM agents",
            "large language model deployment",
            "AI engineering systems",
            "prompt engineering optimization",
            "vector database retrieval",
            "agentic AI systems",
        ],
        "github_keywords": [
            "rag", "llm-agents", "langchain", "langgraph",
            "vector-database", "ai-engineering", "llm-ops",
            "agentic-ai", "crewai", "autogen",
        ],
        "rss_keywords": [
            "RAG", "retrieval augmented generation", "LLM agent",
            "AI engineering", "vector database", "LangChain",
            "prompt engineering", "agentic AI",
        ],
    },
    "push_allocation_optimization": {
        "description": "Push notification optimization, resource allocation, bandit algorithms",
        "arxiv_queries": [
            "push notification optimization",
            "notification delivery optimization",
            "multi-armed bandit notification",
            "user engagement optimization",
            "resource allocation optimization machine learning",
            "budget allocation optimization",
        ],
        "github_keywords": [
            "push-notification-optimization", "notification-system",
            "multi-armed-bandit", "resource-allocation",
            "budget-optimization",
        ],
        "rss_keywords": [
            "push notification", "notification optimization",
            "multi-armed bandit", "resource allocation",
            "budget allocation",
        ],
    },
    "fraud_anomaly_detection": {
        "description": "Fraud detection, anomaly detection in production",
        "arxiv_queries": [
            "fraud detection deep learning",
            "real-time anomaly detection",
            "graph neural network fraud",
            "transaction fraud detection",
        ],
        "github_keywords": [
            "fraud-detection", "anomaly-detection",
            "real-time-anomaly", "graph-fraud-detection",
        ],
        "rss_keywords": [
            "fraud detection", "anomaly detection",
            "real-time fraud", "transaction fraud",
        ],
    },
    "nlp_text_classification": {
        "description": "NLP, text classification, sentiment, transformers",
        "arxiv_queries": [
            "text classification transformers",
            "sentiment analysis large language models",
            "fine-tuning BERT text classification",
            "multilingual NLP",
        ],
        "github_keywords": [
            "text-classification", "sentiment-analysis",
            "nlp-transformers", "fine-tuning-bert",
        ],
        "rss_keywords": [
            "text classification", "sentiment analysis",
            "NLP transformer", "fine-tuning BERT",
        ],
    },
    "recommendation_systems": {
        "description": "Recommendation, ranking, sorting algorithms for e-commerce",
        "arxiv_queries": [
            "recommendation system deep learning",
            "e-commerce ranking algorithm",
            "learning to rank",
            "two-tower model recommendation",
        ],
        "github_keywords": [
            "recommendation-system", "learning-to-rank",
            "two-tower-model", "collaborative-filtering",
        ],
        "rss_keywords": [
            "recommendation system", "ranking algorithm",
            "learning to rank", "collaborative filtering",
        ],
    },
    "llm_optimization": {
        "description": "LLM fine-tuning, quantization, inference optimization",
        "arxiv_queries": [
            "LLM fine-tuning efficiency",
            "model quantization inference",
            "LoRA parameter efficient fine-tuning",
            "speculative decoding",
            "KV cache optimization",
        ],
        "github_keywords": [
            "llm-fine-tuning", "lora", "qlora",
            "model-quantization", "vllm", "speculative-decoding",
        ],
        "rss_keywords": [
            "LLM fine-tuning", "LoRA", "quantization",
            "inference optimization", "speculative decoding",
        ],
    },
}

RSS_FEEDS = {
    "Towards Data Science": "https://towardsdatascience.com/feed",
    "Towards AI": "https://pub.towardsai.net/feed",
    "ML @ Medium": "https://medium.com/feed/tag/machine-learning",
    "AI @ Medium": "https://medium.com/feed/tag/artificial-intelligence",
    "LLM @ Medium": "https://medium.com/feed/tag/llm",
    "Dev.to AI": "https://dev.to/feed/tag/ai",
    "Dev.to ML": "https://dev.to/feed/tag/machinelearning",
    "Analytics Vidhya": "https://www.analyticsvidhya.com/feed/",
    "Sebastian Raschka": "https://magazine.sebastianraschka.com/feed",
    "The Batch (deeplearning.ai)": "https://www.deeplearning.ai/the-batch/feed/",
    "Lil'Log (Lilian Weng)": "https://lilianweng.github.io/index.xml",
    "Jay Alammar": "https://jalammar.github.io/feed.xml",
    "Chip Huyen": "https://huyenchip.com/feed.xml",
    "Eugene Yan": "https://eugeneyan.com/rss/",
    "Simon Willison": "https://simonwillison.net/atom/everything/",
}

ARXIV_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CL", "cs.IR", "cs.CV",
    "stat.ML",
]


settings = Settings()
