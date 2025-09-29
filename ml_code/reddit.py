import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv('../ml_data/jigsaw/train.csv')
test = pd.read_csv('../ml_data/jigsaw/test.csv')
sample_sub = pd.read_csv('../ml_data/jigsaw/sample_submission.csv')




vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(train['body'])
y = train['rule_violation']
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=1)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_valid)
print('accuracy',accuracy_score(y_valid,y_pred))