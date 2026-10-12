"""Train + evaluate quantum and classical models, export results for the website.
Usage: python train.py [--backend numpy|qiskit] [--n-train 600] [--n-test 2000] [--maxiter 300]"""
import argparse, json, time, numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import qmodel as Q

ap = argparse.ArgumentParser()
ap.add_argument('--backend', default='numpy'); ap.add_argument('--n-train', type=int, default=600)
ap.add_argument('--n-test', type=int, default=2000); ap.add_argument('--maxiter', type=int, default=300)
ap.add_argument('--seed', type=int, default=7); a = ap.parse_args()
rng = np.random.default_rng(a.seed)

df = Q.load('data/quantum_stock_prediction_dataset.csv')
df = df[df.y >= 0]                                        # binary task: UP vs DOWN (SIDEWAYS = neutral band at inference)
tr_all, te_all = df[df.date < '2023-01-01'], df[df.date >= '2023-01-01']   # chronological split, no shuffling across time
tr = tr_all.iloc[rng.choice(len(tr_all), a.n_train, replace=False)]
te = te_all.iloc[rng.choice(len(te_all), a.n_test, replace=False)].sort_values('date')

# 12 engineered features -> 4 (one per qubit) using ANOVA F-score on the training window only
sel = SelectKBest(f_classif, k=Q.N_QUBITS).fit(tr_all[Q.FEATS], tr_all.y)
feats = [f for f, m in zip(Q.FEATS, sel.get_support()) if m]
mean, std = tr_all[feats].mean().values, tr_all[feats].std().values
ang = lambda d: Q.to_angles(d[feats].values, mean, std)
Xtr, Xte, ytr, yte = ang(tr), ang(te), tr.y.values, te.y.values
print('qubit features:', feats, '| backend:', a.backend)

def score(name, p, extra=None):
    r = dict(model=name, accuracy=accuracy_score(yte, p > .5), auc=roc_auc_score(yte, p), f1=f1_score(yte, p > .5))
    r.update(extra or {}); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}); return r

res = [score('Majority class', np.full(len(yte), ytr.mean() > .5) * 1.0 + 0)]
res[0]['auc'] = 0.5
res.append(score('Logistic regression (4 feats)', LogisticRegression().fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
res.append(score('Classical RBF-SVM (4 feats)', SVC(probability=True, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150).fit(tr_all[Q.FEATS], tr_all.y)
res.append(score('Gradient boosting (12 feats, full train)', gb.predict_proba(te[Q.FEATS])[:, 1]))

t = time.time(); qk = Q.QuantumKernelSVM(a.backend).fit(Xtr, ytr)
res.append(score('Quantum kernel SVM', qk.predict_proba(Xte), dict(quantum=True)))
vqc = Q.VQC(a.backend).fit(Xtr, ytr, a.maxiter); pv = vqc.predict_proba(Xte)
res.append(score('Variational quantum classifier (VQC)', pv, dict(quantum=True))); print('quantum time %.1fs' % (time.time() - t))

# save models to models/ directory
import os, pickle, joblib
from pathlib import Path
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
joblib.dump(LogisticRegression().fit(Xtr, ytr), models_dir / 'logistic_regression.joblib')
joblib.dump(SVC(probability=True, random_state=0).fit(Xtr, ytr), models_dir / 'classical_svm_rbf.joblib')
joblib.dump(gb, models_dir / 'gradient_boosting.joblib')
with open(models_dir / 'quantum_kernel_svm.pkl', 'wb') as f:
    pickle.dump(qk, f)
with open(models_dir / 'vqc_model.pkl', 'wb') as f:
    pickle.dump(vqc, f)
np.save(models_dir / 'vqc_weights.npy', vqc.w)
json.dump({
    'selected_features': feats,
    'all_features': Q.FEATS,
    'mean': mean.tolist(),
    'std': std.tolist(),
    'n_qubits': Q.N_QUBITS,
    'n_layers': Q.N_LAYERS,
    'fm_reps': Q.FM_REPS,
    'n_theta': Q.N_THETA
}, open(models_dir / 'preprocessor_config.json', 'w'), indent=2)
print('Saved all models into models/')

# export for the website
samples = te.iloc[np.linspace(0, len(te) - 1, 120).astype(int)]
hist = vqc.history[::max(1, len(vqc.history) // 120)]
out = dict(backend=a.backend, n_train=a.n_train, n_test=a.n_test, train_range=[tr_all.date.min(), tr_all.date.max()],
           test_range=[te_all.date.min(), te_all.date.max()], metrics=res, loss_history=hist,
           model=dict(n=Q.N_QUBITS, layers=Q.N_LAYERS, fm_reps=Q.FM_REPS, w=vqc.w.tolist(), features=feats,
                      mean=mean.tolist(), std=std.tolist(),
                      lo=tr_all[feats].quantile(.02).tolist(), hi=tr_all[feats].quantile(.98).tolist()),
           samples=[dict(date=r.date, ticker=r.ticker, trend=r.target_trend, x=[float(r[f]) for f in feats], p=float(p))
                    for (_, r), p in zip(samples.iterrows(), vqc.predict_proba(ang(samples)))])
json.dump(out, open('results/metrics.json', 'w'), indent=1)
open('website/data.js', 'w').write('const DATA = ' + json.dumps(out) + ';')
