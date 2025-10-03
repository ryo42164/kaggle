import pandas as pd 
from sklearn.model_selection import StratifiedKFold,cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import make_scorer, accuracy_score, f1_score


train = pd.read_csv('../ml_data/jigsaw/train.csv')
test = pd.read_csv('../ml_data/jigsaw/test.csv')
sample_sub = pd.read_csv('../ml_data/jigsaw/sample_submission.csv')

text_col='body'
target='rule_violation'

X_text = train[text_col].astype(str).fillna('')
y = train[target]

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

features = FeatureUnion([
    ('word',word_vec),
    ('char',char_vec)
])

scoring = {
    'acc' : make_scorer(accuracy_score),
    'f1m' : make_scorer(f1_score,average='macro')
}

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=1)

def eval_model(model,name):
    pipe = Pipeline([
        ('feats',features),
        ('clf',model)
    ])
    res = cross_validate(pipe,X_text,y,cv = cv,scoring=scoring,n_jobs=-1,return_train_score=False)
    print(f"[{name}]  ACC {res['test_acc'].mean():.4f} ± {res['test_acc'].std():.4f} | "
          f"F1-macro {res['test_f1m'].mean():.4f} ± {res['test_f1m'].std():.4f}")

eval_model(LogisticRegression(solver="saga", max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1), "LogReg")
eval_model(LinearSVC(C=1.0), "LinearSVC")                 # predict_probaは不要、精度は強い
eval_model(RidgeClassifier(alpha=1.0), "RidgeClassifier") # 速くて堅い
eval_model(ComplementNB(alpha=0.5), "ComplementNB")  