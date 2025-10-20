from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedGroupKFold
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
from scipy import sparse

# dataダウンロード
train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

# sbert　メモリを小さくなるように工夫した
model = SentenceTransformer('/kaggle/input/sentence-transformer-all-minilm-l6-v2/pytorch/default/1/all-MiniLM-L6-v2')

def batch_encode_texts(texts, batch_size=2000,inner_bs = 16):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        emb = model.encode(chunk, batch_size = inner_bs,show_progress_bar=False)
        all_embs.append(emb)
    return np.vstack(all_embs)

def calc_sbert_pair(df, col1, col2):
    emb1 = batch_encode_texts(df[col1].fillna('').tolist()).astype('float16')
    emb2 = batch_encode_texts(df[col2].fillna('').tolist()).astype('float16')
    sim = np.diag(cosine_similarity(emb1, emb2))
    del emb1, emb2
    gc.collect()
    return sim

train['sim_body_rule'] = calc_sbert_pair(train, 'body', 'rule')
train['sim_body_pos1'] = calc_sbert_pair(train, 'body', 'positive_example_1')
train['sim_body_pos2'] = calc_sbert_pair(train, 'body', 'positive_example_2')
train['sim_body_neg1'] = calc_sbert_pair(train, 'body', 'negative_example_1')
train['sim_body_neg2'] = calc_sbert_pair(train, 'body', 'negative_example_2')

test['sim_body_rule'] = calc_sbert_pair(test, 'body', 'rule')
test['sim_body_pos1'] = calc_sbert_pair(test, 'body', 'positive_example_1')
test['sim_body_pos2'] = calc_sbert_pair(test, 'body', 'positive_example_2')
test['sim_body_neg1'] = calc_sbert_pair(test, 'body', 'negative_example_1')
test['sim_body_neg2'] = calc_sbert_pair(test, 'body', 'negative_example_2')

# tfidf ここもメモリのためにmax_featuresを少し小さくした
train['text'] = train['body'] + train['rule']
test['text'] = test['body'] + test['rule']

char_vec = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3,5),
    min_df=2,
    max_features=30000,
    sublinear_tf=True,
    lowercase=True
)

word_vec = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95,
    max_features=20000,
    sublinear_tf=True,
    lowercase=True,
    stop_words='english'
)

features = FeatureUnion([
    ('char', char_vec),
    ('word', word_vec)
])

# fold内で使う変数、関数の準備
def diag_cos(a, b):
    num = a.multiply(b).sum(axis=1)
    na = np.sqrt(a.multiply(a).sum(axis=1)).A1
    nb = np.sqrt(b.multiply(b).sum(axis=1)).A1
    den = (na * nb)
    return np.divide(
        np.asarray(num).ravel(),
        np.asarray(den).ravel(),
        out=np.zeros(a.shape[0]),
        where=(np.asarray(den).ravel() != 0)
    )

oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
skf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=1)
fold_scores = []

sim_vec = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1,3),
    min_df=2,
    max_df=0.95,
    max_features=20000,
    sublinear_tf=True,
    lowercase=True,
    stop_words='english'
)

sim_vec.fit(pd.concat([
    train['body'].fillna(''),
    train['positive_example_1'].fillna(''),
    train['positive_example_2'].fillna(''),
    train['negative_example_1'].fillna(''),
    train['negative_example_2'].fillna('')
], axis=0))

# fold を回す
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train, groups=train['rule']), 1):
    trn, val = train.iloc[trn_idx], train.iloc[val_idx]

    # --- Smoothed Target Encoding ---
    global_mean = trn['rule_violation'].mean()
    subrule_stats = trn.groupby(['subreddit', 'rule'])['rule_violation'].agg(['mean', 'count']).reset_index()
    m = 20
    subrule_stats['smoothed_te'] = (
        (subrule_stats['mean'] * subrule_stats['count'] + m * global_mean) /
        (subrule_stats['count'] + m)
    )

    def map_smoothed_te(df, stats, global_mean):
        vals = pd.merge(
            df[['subreddit', 'rule']],
            stats[['subreddit', 'rule', 'smoothed_te']],
            on=['subreddit', 'rule'],
            how='left',
            validate='m:1'
        )[["smoothed_te"]].reset_index(drop=True)
        return vals['smoothed_te'].fillna(global_mean).values.reshape(-1, 1)

    te_tr = map_smoothed_te(trn, subrule_stats, global_mean)
    te_val = map_smoothed_te(val, subrule_stats, global_mean)
    te_te = map_smoothed_te(test, subrule_stats, global_mean)

    # --- TF-IDF and add example ---
    X_tr_vec = features.fit_transform(trn['text'])
    X_val_vec = features.transform(val['text'])
    X_test_vec = features.transform(test['text'])

    def sim_feats(df):
        b = sim_vec.transform(df['body'].fillna(''))
        p1 = sim_vec.transform(df['positive_example_1'].fillna(''))
        p2 = sim_vec.transform(df['positive_example_2'].fillna(''))
        n1 = sim_vec.transform(df['negative_example_1'].fillna(''))
        n2 = sim_vec.transform(df['negative_example_2'].fillna(''))
        return np.c_[
            diag_cos(b, p1),
            diag_cos(b, p2),
            diag_cos(b, n1),
            diag_cos(b, n2)
        ]

    sim_tr = sim_feats(trn)
    sim_val = sim_feats(val)
    sim_te = sim_feats(test)

    num_tr = np.c_[
        te_tr,
        sim_tr,
        trn[['sim_body_rule', 'sim_body_pos1', 'sim_body_pos2', 'sim_body_neg1', 'sim_body_neg2']].values
    ]
    num_val = np.c_[
        te_val,
        sim_val,
        val[['sim_body_rule', 'sim_body_pos1', 'sim_body_pos2', 'sim_body_neg1', 'sim_body_neg2']].values
    ]
    num_te = np.c_[
        te_te,
        sim_te,
        test[['sim_body_rule', 'sim_body_pos1', 'sim_body_pos2', 'sim_body_neg1', 'sim_body_neg2']].values
    ]

    scaler = StandardScaler()
    num_tr_s = sparse.csr_matrix(scaler.fit_transform(num_tr))
    num_val_s = sparse.csr_matrix(scaler.transform(num_val))
    num_te_s = sparse.csr_matrix(scaler.transform(num_te))

    X_tr_final = hstack([X_tr_vec, num_tr_s])
    X_val_final = hstack([X_val_vec, num_val_s])
    X_test_final = hstack([X_test_vec, num_te_s])

    lr = LogisticRegression(
        solver='saga',
        max_iter=1500,
        C=0.5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=fold
    )

    lr.fit(hstack([X_tr_vec, num_tr_s]), trn['rule_violation'])
    pred_lr_val = lr.predict_proba(hstack([X_val_vec, num_val_s]))[:, 1]
    pred_lr_test = lr.predict_proba(hstack([X_test_vec, num_te_s]))[:, 1]

    lgbm = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        random_state=fold,
        n_jobs=-1
    )

    lgbm.fit(
        num_tr_s,
        trn['rule_violation'],
        eval_set=[(num_val_s, val['rule_violation'])],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    pred_lgb_val = lgbm.predict_proba(num_val_s)[:, 1]
    pred_lgb_test = lgbm.predict_proba(num_te_s)[:, 1]

    w = 0.6
    oof[val_idx] = w * pred_lr_val + (1 - w) * pred_lgb_val
    test_pred += w * pred_lr_test + (1 - w) * pred_lgb_test / skf.n_splits

    fold_auc = roc_auc_score(val['rule_violation'], oof[val_idx])
    print(f"[fold{fold}] AUC(val): {fold_auc:.6f}")

    del X_tr_vec, X_val_vec, X_test_vec, lr, lgbm
    gc.collect()
