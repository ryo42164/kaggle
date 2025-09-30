import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


train = pd.read_csv('../ml_data/jigsaw/train.csv')
test = pd.read_csv('../ml_data/jigsaw/test.csv')
sample_sub = pd.read_csv('../ml_data/jigsaw/sample_submission.csv')




char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3,5),
    min_df=2,
    max_features=5000,
    sublinear_tf=True
)

X = char_vectorizer.fit_transform(train['body'].astype(str))
y = train['rule_violation']
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)


model = LogisticRegression(max_iter=1000,class_weight="balanced")
model.fit(X_train,y_train)

y_pred = model.predict(X_valid)
print('char-gram accuracy',accuracy_score(y_valid,y_pred))