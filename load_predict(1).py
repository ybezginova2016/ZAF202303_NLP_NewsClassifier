"""load_predict.ipynb

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier

       """**DIRECT LINK TO DATASET**

https://www.kaggle.com/datasets/balatmak/newsgroup20bbcnews

df = pd.read_csv('/content/News-text.csv')
df.head()

print(f"Shape : {df.shape}")

**Data Preprocessing**

df.info()

df.isnull().sum()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
sns.countplot(x=df.category, color='blue')
plt.title('text class distribution', fontsize=16)
plt.ylabel('Class Counts', fontsize=16)
plt.xlabel('Class Label', fontsize=16)
plt.xticks(rotation='vertical')

from gensim import utils
import gensim.parsing.preprocessing as gsp

filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text
          ]

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s

**Using word Cloud**

%matplotlib inline

from wordcloud import WordCloud

def plot_word_cloud(text):
    wordcloud_instance = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords=None,
                min_font_size = 10).generate(text) 
             
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud_instance) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()

texts = ''
for index, item in df.iterrows():
    texts = texts + ' ' + clean_text(item['text'])
    
plot_word_cloud(texts)

**Using Tf-IDF Vectorizer**

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator

class Text2TfIdfTransformer(BaseEstimator):

    def __init__(self):
        self._model = TfidfVectorizer()
        pass

    def fit(self, df_x, df_y=None):
        df_x = df_x.apply(lambda x : clean_text(x))
        self._model.fit(df_x)
        return self

    def transform(self, df_x):
        return self._model.transform(df_x)

tfidf_transformer = Text2TfIdfTransformer()
tfidf_vectors = tfidf_transformer.fit(df_x).transform(df_x)

tfidf_vectors.shape

print(tfidf_vectors)

**Using LabelEncoder**

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(list(set(list(df['category']))))

list(le.classes_)
['business', 'entertainment', 'politics', 'sport', 'tech']

le.transform(list(set(list(df['category']))))

df['category'] = le.transform(df['category'])

from sklearn.model_selection import train_test_split
#Split test and training data set
X_train, X_test, y_train, y_test = train_test_split(df['text'].values.astype('U'),df['category'].values.astype('int32'), test_size=0.10, random_state=0)
classes  = df['category'].unique()

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

**Using Grid search**

vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,2), max_features=50000,max_df=0.5,use_idf=True, norm='l2') 
counts = vectorizer.fit_transform(X_train)
vocab = vectorizer.vocabulary_
classifier = SGDClassifier(alpha=1e-05,max_iter=50,penalty='elasticnet')
targets = y_train
classifier = classifier.fit(counts, targets)
example_counts = vectorizer.transform(X_test)
predictions = classifier.predict(example_counts)

import pickle
pickle.dump(classifier,open("news_classifier.pkl","wb"))
pickle.dump(vocab,open("vocab_news_classifier.pkl","wb"))

ls
news_classifier.pkl  News-text.csv  sample_data/  vocab_news_classifier.pkl

scores = cross_val_score(classifier, example_counts, y_test, cv=5)

score = scores.mean()
print(round(score,3))

vec = open("news_classifier.pkl", 'rb')
loaded_model = pickle.load(vec)
vcb = open("vocab_news_classifier.pkl", 'rb')
loaded_vocab = pickle.load(vcb)

test = clean_text(df.iloc[2,1])

examples = [test]

from sklearn.feature_extraction.text import TfidfTransformer

count_vect = TfidfVectorizer(analyzer='word',ngram_range=(1,2), max_features=50000,max_df=0.5,use_idf=True, norm='l2',vocabulary=loaded_vocab)
tfidf_transformer = TfidfTransformer()
x_count = count_vect.fit_transform(examples)
predicted = loaded_model.predict(x_count)
result_category = predicted[0]

test= clean_text(newTest)
examples = [test]
count_vect = TfidfVectorizer(analyzer='word',ngram_range=(1,2), max_features=50000,max_df=0.5,use_idf=True, norm='l2',vocabulary=loaded_vocab)
tfidf_transformer = TfidfTransformer()
x_count = count_vect.fit_transform(examples)
predicted = loaded_model.predict(x_count)
result_category = predicted[0]
result_category

le.inverse_transform([3])

final_pred = le.inverse_transform([result_category])
print(final_pred)
