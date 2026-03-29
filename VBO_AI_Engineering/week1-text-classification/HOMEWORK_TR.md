# Hafta 1 Ödevi: GPU ile Hızlandırılmış Metin Sınıflandırma (PyTorch)

## Amaç
Bu ödevin amacı, **PyTorch** ve **Hugging Face Transformers** kullanarak yüksek performanslı bir metin sınıflandırma pipeline'ı oluşturmaktır. Metin verilerini verimli bir şekilde kategorize eden bir modeli eğitmek ve değerlendirmek için **GPU hızlandırmasından (CUDA)** yararlanacaksınız.

---

## Görevler

### Görev 1: Ortam Kurulumu ve GPU Yapılandırması (15 puan)
1. `torch`, `transformers` ve `datasets` kütüphaneleri ile bir Python ortamı kurun.
2. `torch.cuda.is_available()` kullanarak GPU'nun kullanılabilir olduğunu doğrulayın.
3. Eğitim döngüsünü, hem **modelin** hem de **tensor'ların** `cuda` cihazına taşınacak şekilde yapılandırın.

### Görev 2: Veri Ön İşleme ve Tokenizasyon (25 puan)
1. Standart bir metin sınıflandırma veri setini yükleyin (örneğin duygu analizi için *IMDb* veya konu sınıflandırma için *AG News*).
2. Ham metni girdi ID'lerine (input IDs) ve dikkat maskelerine (attention masks) dönüştürmek için bir tokenizer uygulayın (örneğin `BertTokenizer` veya `AutoTokenizer`).
3. Eğitim ve doğrulama için uygun batch boyutlarıyla PyTorch `DataLoader` nesneleri oluşturun.

### Görev 3: Model Mimarisi ve Eğitim (40 puan)
1. Dizi sınıflandırma (sequence classification) için önceden eğitilmiş bir Transformer modelini başlatın (örneğin `distilbert-base-uncased`).
2. Aşağıdakileri içeren bir eğitim döngüsü tanımlayın:
    * İleri geçiş (forward pass)
    * Kayıp hesaplama (CrossEntropy)
    * Geri yayılım (backpropagation)
    * Optimizer adımı (AdamW)
3. En az 3 epoch boyunca eğitim kaybını (training loss) ve doğrulama doğruluğunu (validation accuracy) kaydedin.

### Görev 4: Değerlendirme ve Çıkarım (20 puan)
1. Modeli test seti üzerinde değerlendirin ve bir **Sınıflandırma Raporu** (Precision, Recall, F1-skoru) oluşturun.
2. Özel bir metin girdisi alan, bunu GPU'ya taşıyan ve tahmin edilen kategoriyi döndüren bir fonksiyon yazın.

---

## Teslim Edilecekler
- `requirements.txt`: Gerekli kütüphanelerin listesi.
- `train.py` veya `notebook.ipynb`: Kurulum, eğitim ve değerlendirmeyi içeren tam kod.
- `training_log.txt`: Epoch'lar boyunca kaybın azaldığını ve nihai doğruluğu gösteren çıktı.
- `inference_examples.pdf`: Modelin yeni metin girdileri için kategori tahminleri yaptığı 3-5 örnek.

---

## Puanlama
- **GPU Doğrulama ve Cihaz Yönetimi**: 15 puan
- **Veri Pipeline'ı ve Tokenizasyon**: 25 puan
- **Eğitim Döngüsü Uygulaması**: 40 puan
- **Değerlendirme ve Çıkarım Mantığı**: 20 puan
- **Toplam: 100 puan**
