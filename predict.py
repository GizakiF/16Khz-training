import os
import joblib
import pickle
import numpy as np

# ✅ Load models
svm_path = os.path.expanduser(
    "~/Research/Sleep Deprivation Detection using voice/output/day_session/models/svm_combined.pkl"
)
pca_path = os.path.expanduser(
    "~/Research/Sleep Deprivation Detection using voice/output/day_session/models/pca_combined.pkl"
)
test_sample_path = os.path.expanduser(
    "~/Research/Sleep Deprivation Detection using voice/dataset/osf/stmtf/strf_session_pre_subjectNb_01_daySession_01_segmentNb_0.pkl"
)

svm = joblib.load(svm_path)
pca = joblib.load(pca_path)

with open(test_sample_path, "rb") as f:
    test_sample = pickle.load(f)

strf = np.asarray(test_sample["strf"]).flatten()

expected_features = pca.components_.shape[1]
if strf.shape[0] != expected_features:
    raise ValueError(
        f"Feature mismatch! Expected {expected_features} features, but got {
            strf.shape[0]
        }."
    )

X_pca = pca.transform(strf.reshape(1, -1))

y_pred = svm.predict(X_pca)

# 0 - post. 1 - pre
print("All classes in the model:", svm.classes_)
print("Predicted class:", y_pred)
