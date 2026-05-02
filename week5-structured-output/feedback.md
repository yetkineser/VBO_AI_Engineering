# Week 5 — Feedback ve Düzeltme

## Hocadan Gelen Geri Bildirim (Alican Payaslı, 2026-04-29)

> Ellerine sağlık Yetkin, her hafta olduğu gibi yine dört dörtlük bir çalışma. İstenilenin de ötesinde teslimler. Sadece şurada küçük bir eksik var:
>
> **Error Handling:**
> - Try/except yok, bir satırın LLM çağrısında hata olursa tüm pipeline çöker ve geri kalan ticketlar işlenmez. Satır bazında try/except + opsiyonel retry iyi olur. (-1p)
>
> Teşekkür ederiz

## Geri Bildirim Üzerine Yapılan Düzeltme

### Sorun
Orijinal `main.py`'da `agent.invoke(...)` çağrısı `for` döngüsünün içinde **çıplak** çağrılıyordu. Tek bir satırda hata olursa (rate limit, timeout, network drop, Pydantic validation fail) tüm pipeline çöker ve geri kalan ticket'lar işlenmezdi. 50 satırlık bir batch'in 47'si işlendikten sonra çökmek = baştan başlamak.

### Çözüm: Satır Bazında try/except + Tenacity ile Retry

İki katmanlı hata yönetimi eklendi:

1. **Inner katman (transient hatalar için retry)**: `tenacity` ile 3 deneme, exponential backoff (2s → 4s → 8s).
2. **Outer katman (her satır için try/except)**: 3 retry de başarısız olursa, hata `errors.jsonl`'e yazılır ve döngü diğer satırla devam eder.

### Akıllı Retry: ValidationError'ları retry'lama
`pydantic.ValidationError` model çıktısı şemaya uymadığı için olur — aynı prompt'u tekrar göndermek aynı hatayı verir. `retry_if_not_exception_type(ValidationError)` ile bu durumda direkt fail edilir, üç defa boşa beklenmez.

### Yeni Output Dosyası: `errors.jsonl`
Her hatalı satır için:
```json
{"customer_id": "CUST-XXX", "error_type": "TimeoutError", "error_message": "..."}
```

Stdout'a düşen progress örneği:
```
[3/8] FAIL CUST-003: TransientLLMError — Connection timeout
  → logged to errors.jsonl, continuing.
...
Done. 7/8 successful, 1 errors.
```

## Ders

**Production-style pipeline'larda her external API çağrısı satır bazında try/except + retry ile sarılmalı.** Tek bir transient hata batch'in ortasında pipeline'ı çökertirse, kısmi sonuç kaybolur ve baştan başlamak gerekir. Satır-bazlı izolasyon + retry ile failure radius azalır, başarılı satırlar korunur, sadece gerçekten kalıcı hatalı satırlar işlenmez.

Bu düzeltme [main.py](main.py)'a uygulandı; `pyproject.toml`'a `tenacity>=8.0` eklendi.
