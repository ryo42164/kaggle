import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack


train = pd.read_csv('../ml_data/jigsaw/train.csv')
test = pd.read_csv('../ml_data/jigsaw/test.csv')
sample_sub = pd.read_csv('../ml_data/jigsaw/sample_submission.csv')

text_col='body'
target='rule_violation'



char_vec = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3,5),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
    lowercase=True
)
word_vec = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95,
    max_features=30000,
    sublinear_tf=True,
    lowercase=True
)


X_text_train,X_text_valid,y_train,y_valid = train_test_split(train[text_col],train[target],test_size=0.2,random_state=1,stratify=train[target])
Xw_tr = word_vec.fit_transform(X_text_train)
Xw_val = word_vec.transform(X_text_valid)
Xc_tr = char_vec.fit_transform(X_text_train)
Xc_val = char_vec.transform(X_text_valid)
X_train = hstack([Xw_tr,Xc_tr])
X_valid = hstack([Xw_val,Xc_val])

def get_score(C):
    clf = LogisticRegression(
        solver='saga',
        max_iter=1000,
        C=C,
        class_weight='balanced',
        n_jobs=-1
    )
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_valid)
    return accuracy_score(y_pred,y_valid)

results={}
for c in [0.25,1,2,3,4]:
    results[c] = get_score(c)

print(f'best accuracy {max(results.values())} C {max(results,key=results.get)}')