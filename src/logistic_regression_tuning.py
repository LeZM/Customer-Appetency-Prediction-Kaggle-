import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif

# ============================================================
# STEP 1: Tune C with cross-validation (L1)
# ============================================================
print("=" * 50)
print("STEP 1: C Grid Search (L1)")
print("=" * 50)

c_values = [0.005, 0.01, 0.015, 0.017, 0.02, 0.03, 0.05]
best_c = None
best_score = 0

for c in c_values:
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=1000)
    scores = cross_val_score(lr, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    mean_score = scores.mean()
    print(f"  C={c:.3f}: AUC={mean_score:.4f} (+/- {scores.std():.4f})")
    if mean_score > best_score:
        best_score = mean_score
        best_c = c

print(f"\n  Best C: {best_c} with AUC: {best_score:.4f}")


# ============================================================
# STEP 2: Add class_weight='balanced' to handle imbalance
# ============================================================
print("\n" + "=" * 50)
print("STEP 2: Class Weight Balanced (L1)")
print("=" * 50)

lr_balanced = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=best_c,
    class_weight='balanced',
    max_iter=1000
)
lr_balanced.fit(X_train, y_train)

y_pred_train_b = lr_balanced.predict_proba(X_train)[:, 1]
y_pred_val_b   = lr_balanced.predict_proba(X_val)[:, 1]

print(f"  Train AUC : {roc_auc_score(y_train, y_pred_train_b):.4f}")
print(f"  Val   AUC : {roc_auc_score(y_val,   y_pred_val_b):.4f}")


# ============================================================
# STEP 3: Increase SelectKBest features (L2)
# ============================================================
print("\n" + "=" * 50)
print("STEP 3: SelectKBest with more features (L2)")
print("=" * 50)

for k in [500, 1000, 2000]:
    selector_k = SelectKBest(f_classif, k=k)
    X_train_k  = selector_k.fit_transform(X_train, y_train)
    X_val_k    = selector_k.transform(X_val)

    lr_k = LogisticRegression(penalty='l2', solver='lbfgs', C=0.001, max_iter=1000)
    lr_k.fit(X_train_k, y_train)

    val_auc = roc_auc_score(y_val, lr_k.predict_proba(X_val_k)[:, 1])
    print(f"  k={k:>5}: Val AUC={val_auc:.4f}")


# ============================================================
# STEP 4: Two-stage L1 → L2
# ============================================================
print("\n" + "=" * 50)
print("STEP 4: Two-stage L1 feature selection + L2 refitting")
print("=" * 50)

# Stage 1: L1 to select features
lr_l1 = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=best_c,
    class_weight='balanced',
    max_iter=1000
)
lr_l1.fit(X_train, y_train)

mask = lr_l1.coef_[0] != 0
print(f"  Features selected by L1: {mask.sum()} / {X_train.shape[1]}")

X_train_sel = X_train.loc[:, mask]
X_val_sel   = X_val.loc[:, mask]

# Stage 2: L2 on selected features
for c2 in [0.01, 0.1, 1.0]:
    lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=c2, max_iter=1000)
    lr_l2.fit(X_train_sel, y_train)
    val_auc = roc_auc_score(y_val, lr_l2.predict_proba(X_val_sel)[:, 1])
    print(f"  L2 C={c2}: Val AUC={val_auc:.4f}")


# ============================================================
# STEP 5: Final best model + submission
# ============================================================
print("\n" + "=" * 50)
print("STEP 5: Final model summary")
print("=" * 50)

# --- Pick the best combo from above and refit here ---
# Example: L1 balanced (often best — update C if Step 1 found something better)
final_model = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=best_c,
    class_weight='balanced',
    max_iter=1000
)
final_model.fit(X_train, y_train)

y_pred_train_f = final_model.predict_proba(X_train)[:, 1]
y_pred_val_f   = final_model.predict_proba(X_val)[:, 1]
y_pred_all_f   = final_model.predict_proba(X)[:, 1]
y_pred_test_f  = final_model.predict_proba(test_final)[:, 1]

print(f"  Train AUC : {roc_auc_score(y_train, y_pred_train_f):.4f}")
print(f"  Val   AUC : {roc_auc_score(y_val,   y_pred_val_f):.4f}")
print(f"  Full  AUC : {roc_auc_score(y,       y_pred_all_f):.4f}")

# Save submission
pred_test = pd.Series(y_pred_test_f, index=test_final.index, name='Target_appetency')
submission = pd.DataFrame({'Target_appetency': pred_test})
submission.index.name = 'ID'
submission.to_csv('submission_final.csv')
print("\n  Saved → submission_final.csv")
