"""Embedding utilities: load pre-trained Turkish word vectors, measure
similarity, and cluster words.

All four functions required by the Week 3 homework live here so they can be
imported cleanly from `main.py` or reused from other scripts/notebooks.

Design notes
------------
- We always return a `gensim.models.KeyedVectors` object (not a full FastText
  model) because it is faster to load and uses far less memory.
- OOV (out-of-vocabulary) handling is explicit: `get_word_vector` returns
  `None`, and `word_similarity` returns `float('nan')` rather than crashing.
- Text is normalised (lowercased, stripped) before lookup so that
  "Kedi", "KEDİ ", and "kedi" all hit the same vector.
"""

from __future__ import annotations

import logging
import string
from pathlib import Path
from typing import Iterable

import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Text normalisation                                                          #
# --------------------------------------------------------------------------- #

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalise_word(word: str) -> str:
    """Lowercase + strip whitespace + remove ASCII punctuation.

    Turkish-specific lowercase:
      'İ' -> 'i'  (dotted capital I -> dotted lowercase i)
      'I' -> 'ı'  (undotted capital I -> undotted lowercase i)

    We handle this explicitly because Python's casefold() produces
    'i\\u0307' for 'İ', which won't match FastText/GloVe vocabularies.
    """
    if word is None:
        return ""
    w = word.strip().translate(_PUNCT_TABLE)
    # Turkish-aware lowercase: handle İ and I before generic lowering
    w = w.replace("İ", "i").replace("I", "ı")
    return w.lower()


# --------------------------------------------------------------------------- #
# 1. Model loading                                                            #
# --------------------------------------------------------------------------- #


def load_fasttext_model(path: str | Path, limit: int | None = 200_000) -> KeyedVectors:
    """Load pre-trained FastText Turkish vectors from a word2vec text file.

    Parameters
    ----------
    path:
        Path to `cc.tr.300.vec` (uncompressed) or `cc.tr.300.vec.gz`.
    limit:
        Load only the top-N most frequent words. Defaults to 200K to keep
        memory under ~600 MB. Set to `None` to load the full ~2M vocabulary.

    Returns
    -------
    KeyedVectors
        A read-only vector store indexed by word.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FastText file not found at {path}. "
            "See README.md for the download instructions."
        )

    logger.info("Loading FastText vectors from %s (limit=%s)", path, limit)
    kv = KeyedVectors.load_word2vec_format(
        str(path), binary=False, limit=limit, encoding="utf-8"
    )
    logger.info("Loaded %d vectors, dim=%d", len(kv), kv.vector_size)
    return kv


def load_glove_model(path: str | Path, limit: int | None = 200_000) -> KeyedVectors:
    """Load pre-trained GloVe Turkish vectors.

    GloVe files usually omit the `<vocab_size> <dim>` header that word2vec
    format requires. We probe the first line: if it looks like a header we
    load directly; otherwise we prepend a synthetic header on the fly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"GloVe file not found at {path}. "
            "See README.md for the download instructions."
        )

    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip().split()

    has_header = len(first) == 2 and first[0].isdigit() and first[1].isdigit()

    if has_header:
        logger.info("Loading GloVe vectors (word2vec header detected)")
        return KeyedVectors.load_word2vec_format(
            str(path), binary=False, limit=limit, encoding="utf-8"
        )

    # No header -> use no_header=True (gensim >=4.0 supports this flag)
    logger.info("Loading GloVe vectors (no header, gensim no_header=True)")
    return KeyedVectors.load_word2vec_format(
        str(path),
        binary=False,
        limit=limit,
        encoding="utf-8",
        no_header=True,
    )


# --------------------------------------------------------------------------- #
# 2. Word vector lookup                                                       #
# --------------------------------------------------------------------------- #


def get_word_vector(model: KeyedVectors, word: str) -> np.ndarray | None:
    """Return the vector for `word`, or `None` if the word is OOV.

    The lookup is forgiving: we try the raw word first, then a normalised
    version. This lets callers pass user input directly.
    """
    if not word:
        return None

    if word in model:
        return model[word]

    norm = normalise_word(word)
    if norm and norm in model:
        return model[norm]

    logger.debug("OOV word: %r (normalised: %r)", word, norm)
    return None


# --------------------------------------------------------------------------- #
# 3. Cosine similarity                                                        #
# --------------------------------------------------------------------------- #


def word_similarity(model: KeyedVectors, word1: str, word2: str) -> float:
    """Cosine similarity between two words in `[-1, 1]`.

    Returns `float('nan')` when either word is OOV, so that callers can
    filter on `math.isnan()` without catching exceptions.
    """
    v1 = get_word_vector(model, word1)
    v2 = get_word_vector(model, word2)
    if v1 is None or v2 is None:
        return float("nan")

    # sklearn.cosine_similarity expects 2D arrays
    sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
    return float(sim)


# --------------------------------------------------------------------------- #
# 4. K-Means clustering                                                       #
# --------------------------------------------------------------------------- #


def cluster_words(
    model: KeyedVectors,
    words: Iterable[str],
    k: int = 3,
) -> dict[str, int]:
    """Cluster `words` into `k` groups with K-Means.

    Uses `random_state=42` and `n_init=10` (as required by the homework).
    Vectors are L2-normalised before clustering, which makes Euclidean
    distance equivalent to cosine distance — closer to what we actually
    want in embedding space.

    OOV words are silently skipped (a debug log entry is emitted). If no
    words survive, we return an empty dict.
    """
    words = list(words)
    if not words:
        return {}

    known_words: list[str] = []
    vectors: list[np.ndarray] = []
    for w in words:
        v = get_word_vector(model, w)
        if v is None:
            logger.debug("Skipping OOV word: %r", w)
            continue
        known_words.append(w)
        vectors.append(v)

    if not known_words:
        logger.warning("All %d input words were OOV — returning empty result", len(words))
        return {}

    X = np.vstack(vectors).astype(np.float32)

    # L2-normalise so Euclidean distance == cosine distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    # Guard against k > number of valid words
    k_effective = min(k, len(known_words))
    if k_effective < k:
        logger.warning(
            "Only %d valid words for k=%d — reducing k to %d",
            len(known_words),
            k,
            k_effective,
        )

    km = KMeans(n_clusters=k_effective, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    return {w: int(lbl) for w, lbl in zip(known_words, labels)}
