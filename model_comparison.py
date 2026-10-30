"""Model Comparison Report: 3 supervised models + unsupervised analysis on Breast Cancer Wisconsin data."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve,
                             silhouette_score, adjusted_rand_score)

SEED = 42
# ---------------- 1. Data ----------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = (data.target == 0).astype(int)            # 1 = malignant (positive class), 0 = benign
print(f"Samples={len(y)}, features={X.shape[1]}, malignant={y.sum()} ({y.mean():.1%}), benign={(y==0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
print(f"Train={len(y_train)}, Test={len(y_test)} (test positives={y_test.sum()})")

# ---------------- 2. Supervised models ----------------
# Pipelines keep scaling inside each CV fold (no leakage). Small grids tuned on TRAIN only.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
candidates = {
    "Logistic Regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
                            {"logisticregression__C": [0.01, 0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED),
                      {"n_estimators": [200, 400], "max_depth": [None, 5, 10]}),
    "SVM (RBF)": (make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
                  {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01]}),
}
best, cv_rows, test_rows, proba, pred = {}, [], [], {}, {}
for name, (est, grid) in candidates.items():
    gs = GridSearchCV(est, grid, cv=cv, scoring="f1", n_jobs=1).fit(X_train, y_train)
    best[name] = gs.best_estimator_
    print(f"{name}: best params={gs.best_params_}")
    scores = cross_validate(best[name], X_train, y_train, cv=cv,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    cv_rows.append({"Model": name, **{m: f"{scores['test_'+m].mean():.3f} ± {scores['test_'+m].std():.3f}"
                                      for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}})
    p = best[name].predict(X_test); pr = best[name].predict_proba(X_test)[:, 1]
    pred[name], proba[name] = p, pr
    test_rows.append({"Model": name, "Accuracy": accuracy_score(y_test, p), "Precision": precision_score(y_test, p),
                      "Recall": recall_score(y_test, p), "F1": f1_score(y_test, p), "ROC-AUC": roc_auc_score(y_test, pr),
                      "FN": int(confusion_matrix(y_test, p)[1, 0]), "FP": int(confusion_matrix(y_test, p)[0, 1])})
cv_df = pd.DataFrame(cv_rows).set_index("Model"); test_df = pd.DataFrame(test_rows).set_index("Model").round(3)
print("\n5-fold CV on training set (mean ± std):\n", cv_df.to_string())
print("\nHeld-out test set:\n", test_df.to_string())
cv_df.to_csv("cv_results.csv"); test_df.to_csv("test_results.csv")

# Confusion matrices
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, name in zip(ax, best):
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred[name]), display_labels=["Benign", "Malignant"]).plot(ax=a, colorbar=False, cmap="Blues")
    a.set_title(name)
plt.tight_layout(); plt.savefig("confusion_matrices.png", dpi=150); plt.close()
# ROC
plt.figure(figsize=(6, 5))
for name in best:
    f, t, _ = roc_curve(y_test, proba[name]); plt.plot(f, t, label=f"{name} (AUC={roc_auc_score(y_test, proba[name]):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test set)")
plt.legend(); plt.tight_layout(); plt.savefig("roc_curves.png", dpi=150); plt.close()

# ---------------- 3. Unsupervised: K-Means + PCA (labels never used for fitting) ----------------
scaler = StandardScaler().fit(X_train)
Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)
print("\nSilhouette by k (train):")
sil = {}
for k in range(2, 8):
    sil[k] = silhouette_score(Xtr_s, KMeans(k, n_init=20, random_state=SEED).fit_predict(Xtr_s)); print(f"  k={k}: {sil[k]:.3f}")
km = KMeans(2, n_init=20, random_state=SEED).fit(Xtr_s)
ctr, cte = km.labels_, km.predict(Xte_s)
# Align cluster id to "malignant" by majority for readability
mal_cluster = pd.crosstab(ctr, y_train.values).idxmax()[1]
print(f"\nARI train={adjusted_rand_score(y_train, ctr):.3f}, test={adjusted_rand_score(y_test, cte):.3f}")
print("Crosstab (test) rows=cluster, cols=[benign, malignant]:\n", pd.crosstab(cte, y_test.values))
km_pred = (cte == mal_cluster).astype(int)
print(f"K-Means-as-classifier on test: accuracy={accuracy_score(y_test, km_pred):.3f}, recall={recall_score(y_test, km_pred):.3f}, precision={precision_score(y_test, km_pred):.3f}")

pca = PCA(2, random_state=SEED).fit(Xtr_s); Z = pca.transform(Xte_s); evr = pca.explained_variance_ratio_
print(f"PCA explained variance: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, total={evr.sum():.3f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, lab, title in [(ax[0], cte, "K-Means clusters (k=2)"), (ax[1], y_test.values, "True diagnosis (hidden from K-Means)")]:
    s = a.scatter(Z[:, 0], Z[:, 1], c=lab, cmap="coolwarm", edgecolor="k", lw=0.3, s=40)
    a.set(title=title, xlabel=f"PC1 ({evr[0]:.1%})", ylabel=f"PC2 ({evr[1]:.1%})")
plt.tight_layout(); plt.savefig("kmeans_pca.png", dpi=150); plt.close()

# Does the cluster label add anything as an extra feature for the supervised model?
Xtr_aug = np.column_stack([Xtr_s, ctr]); Xte_aug = np.column_stack([Xte_s, cte])
lr = LogisticRegression(C=best["Logistic Regression"].get_params()["logisticregression__C"], max_iter=2000, random_state=SEED)
base = lr.fit(Xtr_s, y_train).predict(Xte_s); aug = lr.fit(Xtr_aug, y_train).predict(Xte_aug)
print(f"\nLR without cluster feature: F1={f1_score(y_test, base):.3f}; with cluster feature: F1={f1_score(y_test, aug):.3f}")

# Top features separating clusters
prof = pd.DataFrame(Xtr_s, columns=X.columns).assign(c=ctr).groupby("c").mean()
d = (prof.loc[mal_cluster] - prof.loc[1 - mal_cluster]).sort_values(ascending=False)
print("\nFeatures most elevated in the 'malignant' cluster:", list(d.index[:5]))
