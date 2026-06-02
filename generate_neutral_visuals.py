import pandas as pd
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading data...")
df = pd.read_csv('data/processed/reviews_cleaned.csv')
df = df.dropna(subset=['yorum'])

# 1. WordCloud
neutral_text = " ".join(df[df['label'] == 'neutral']['yorum'].astype(str))
print("Generating WordCloud...")
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Wistia').generate(neutral_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('visuals/wordcloud_neutral.png', bbox_inches='tight')
plt.close()

# 2. TF-IDF
print("Calculating TF-IDF...")
vectorizer = TfidfVectorizer(max_features=20)
X = vectorizer.fit_transform(df[df['label'] == 'neutral']['yorum'].astype(str))
scores = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
neutral_tfidf = [{"word": word, "score": score} for word, score in sorted_scores]

# Update JSON
print("Updating JSON...")
with open('data/processed/tfidf_top_words.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

data['neutral'] = neutral_tfidf

with open('data/processed/tfidf_top_words.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Done!")
