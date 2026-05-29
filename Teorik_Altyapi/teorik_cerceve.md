# Teorik Çerçeve

## 1. Duygu Analizi (Sentiment Analysis)

Duygu analizi, metin verilerinden duygusal tonu (pozitif, negatif, nötr) otomatik olarak çıkarma işlemidir. Doğal dil işleme (NLP) alanının önemli bir alt dalı olup, müşteri geri bildirimi analizi, sosyal medya izleme ve pazar araştırmasında yaygın olarak kullanılır.

### Yaklaşımlar

1. **Kural Tabanlı:** Önceden tanımlanmış kelime listeleri ve dilbilgisi kurallarını kullanır
2. **Makine Öğrenmesi Tabanlı:** Etiketli verilerden öğrenilen istatistiksel modeller
3. **Derin Öğrenme Tabanlı:** Sinir ağı mimarileri (RNN, LSTM, Transformer)
4. **Transfer Öğrenme:** Önceden eğitilmiş dil modelleri (BERT, GPT)

## 2. TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF, bir kelimenin bir dokümandaki önemini ölçen istatistiksel bir ölçüttür.

- **TF (Terim Frekansı):** Bir kelimenin dokümanda kaç kez geçtiğini ölçer
- **IDF (Ters Doküman Frekansı):** Kelimenin tüm dokümanlardaki nadirliği

`TF-IDF = TF x IDF`

Yüksek TF-IDF skoru, kelimenin o dokümana özgü ve önemli olduğunu gösterir.

## 3. BERT (Bidirectional Encoder Representations from Transformers)

BERT, Google tarafından geliştirilen çift yönlü transformer tabanlı bir dil modelidir. Geleneksel modellerin aksine, bir kelimenin anlamını hem sol hem sağ bağlamını dikkate alarak anlar.

### Özellikler
- **Çift Yönlü:** Metni her iki yönde işler
- **Ön Eğitim + İnce Ayar:** Büyük veri setlerinde ön eğitim, belirli görevler için ince ayar
- **Transfer Öğrenme:** Bir görevde öğrenilenler başka görevlere aktarılabilir

### Kullanılan Model
`nlptown/bert-base-multilingual-uncased-sentiment` modeli 6 dilde 100K+ ürün yorumu ile ince ayarlanmıştır.

## 4. Konu Modelleme (Topic Modeling)

### LDA (Latent Dirichlet Allocation)

LDA, bir doküman koleksiyonundaki gizli konuları keşfeden olasılıksal bir modeldir:

- Her doküman birden fazla konunun bir karışımıdır
- Her konu birden fazla kelimenin bir karışımıdır
- Bayes çıkarımı ile konuları otomatik olarak belirler

## 5. LIME (Local Interpretable Model-agnostic Explanations)

LIME, herhangi bir sınıflandırma modelinin bireysel tahminlerini açıklayan bir yorumlanabilirlik yöntemidir:

1. Açıklanacak örneğin civarında yapay örnekler oluşturur
2. Her yapay örneği orijinal modelle tahmin eder
3. Yerel olarak doğrusal bir vekil model eğitir
4. Vekil modelin katsayıları, her özelliklerin katkısını gösterir

## 6. Klasik Makine Öğrenmesi Modelleri

### Logistic Regression
Doğrusal sınıflandırma modeli. Metin sınıflandırmada basit ama etkili.

### Naive Bayes (MultinomialNB)
Bayes teoremine dayalı olasılıksal sınıflandırıcı. Metin verileri için özellikle uygun.

### SVM (Support Vector Machine)
Sınıflar arasındaki en geniş marjini bulan sınıflandırıcı. Yüksek boyutlu metin verilerinde başarılı.

### Random Forest
Birden fazla karar ağacının toplu oylaması ile sınıflandırma. Aşırı uyuma karşı dayanıklı.

### XGBoost
Gradyan artırma (gradient boosting) tabanlı topluluk yöntemi. Yüksek performanslı ve esnek.

## 7. Değerlendirme Metrikleri

| Metrik | Formül | Açıklama |
|--------|--------|----------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Genel doğruluk |
| Precision | TP/(TP+FP) | Pozitif tahmin doğruluğu |
| Recall | TP/(TP+FN) | Gerçek pozitifleri yakalama |
| F1-Score | 2*(P*R)/(P+R) | Precision-Recall dengesi |

### Cross-Validation (Çapraz Doğrulama)
Veri setini K katlamaya bölüp, her seferinde farklı bir katlama test seti olarak kullanma yöntemi. Modelin genelleme yeteneğini daha güvenilir ölçer.

## Kaynaklar

1. Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.
2. Blei, D. M. et al. (2003). Latent Dirichlet Allocation. JMLR.
3. Ribeiro, M. T. et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.
4. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
