# Week 3 — Word Embedding Results

*Generated: 2026-04-10 11:51*
*Model: fasttext — data/cc.tr.300.vec (limit=200000, vocab=200000)*

---

## Cosine Similarity

| Word 1 | Word 2 | Similarity |
|--------|--------|------------|
| kedi | köpek | 0.7857 |
| araba | otobüs | 0.5893 |
| elma | muz | 0.6328 |
| kedi | araba | 0.3681 |
| güzel | çirkin | 0.5695 |
| iyi | kötü | 0.7199 |
| büyük | küçük | 0.6939 |
| hızlı | yavaş | 0.6676 |
| okul | üniversite | 0.6412 |
| doktor | hastane | 0.5650 |
| bilgisayar | telefon | 0.5530 |
| kitap | dergi | 0.6098 |

---

## K-Means Clustering

- **Cluster 0:** araba, otobüs, uçak, tren, bisiklet, gemi, kamyon, motosiklet
- **Cluster 1:** elma, muz, portakal, çilek, üzüm, karpuz, kiraz, armut
- **Cluster 2:** kedi, köpek, kuş, balık, at, tavuk, inek, aslan

---

## Analogies

| A | B | C | Expected | Result | Score |
|---|---|---|----------|--------|-------|
| erkek | kadın | kral | kraliçe | kralın | 0.6981 |
| erkek | kadın | baba | anne | anne | 0.7786 |
| erkek | kadın | oğul | kız | baba | 0.6341 |
| erkek | kadın | amca | teyze | amcanın | 0.7492 |
| erkek | kadın | dede | nine | dedeler | 0.6507 |
| türkiye | ankara | almanya | berlin | ankarada | 0.6051 |
| türkiye | ankara | fransa | paris | belçika | 0.5872 |
| türkiye | ankara | japonya | tokyo | sincan | 0.5563 |
| türkiye | ankara | rusya | moskova | moskova | 0.6943 |
| türkiye | ankara | ingiltere | londra | ankarada | 0.5718 |
| iyi | kötü | güzel | çirkin | çirkin | 0.5629 |
| sıcak | soğuk | büyük | küçük | küçük | 0.5852 |
| hızlı | yavaş | uzun | kısa | uzunca | 0.7399 |
| zengin | fakir | güçlü | zayıf | güçsüz | 0.7250 |
| gitmek | geldi | yazmak | yazdı | yazdım | 0.5723 |
| okumak | okudu | yazmak | yazdı | yazdı | 0.6526 |
| doktor | hastane | öğretmen | okul | okul | 0.6329 |
| doktor | hastane | asker | kışla | askeri | 0.5947 |
| fransa | fransızca | almanya | almanca | almanca | 0.6503 |
| fransa | fransızca | japonya | japonca | japonca | 0.6546 |
| fransa | fransızca | türkiye | türkçe | ingilizce | 0.5413 |
| türkiye | lira | amerika | dolar | liraya | 0.7121 |
| türkiye | lira | japonya | yen | liraya | 0.7040 |
| köpek | havlamak | kedi | miyavlamak | OOV | N/A (OOV) |
| iyi | daha | çok | fazla | hayli | 0.6454 |
