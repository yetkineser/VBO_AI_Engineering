# Data Directory

This directory holds pre-trained word embedding files. They are **not committed** to git because they are too large (up to ~4.5 GB).

## Download Instructions

### FastText Turkish (recommended)

```bash
# Download the compressed file (~1.2 GB download, ~4.5 GB uncompressed)
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.vec.gz

# Decompress (optional — gensim can read .gz directly, but it's slower)
gunzip cc.tr.300.vec.gz

# Move to this directory
mv cc.tr.300.vec data/
```

Alternatively, visit: https://fasttext.cc/docs/en/crawl-vectors.html and download the Turkish `.vec` file.

### GloVe Turkish (alternative)

Community-trained Turkish GloVe vectors can be found on various GitHub repos and Kaggle datasets. Look for files named `glove.tr.300.txt` or similar.

## After Download

Your `data/` directory should look like:

```
data/
├── README.md          (this file)
└── cc.tr.300.vec      (or cc.tr.300.vec.gz)
```

Then run:
```bash
python src/main.py --model data/cc.tr.300.vec
```
