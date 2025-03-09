import os
import pickle
import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score


# svm_path = os.path.expanduser(
#     "~/Research/Sleep Deprivation Detection using voice/output/day_session/models/svm_combined.pkl"
# )
#
svm_path = os.path.expanduser(
    "~/Research/Sleep Deprivation Detection using voice/output/pop_level/population_level_svm.pkl"
)
pca_path = os.path.expanduser(
    "~/Research/Sleep Deprivation Detection using voice/output/pop_level/population_level_pca.pkl"
)
test_sample_path = os.path.expanduser(
    "~/Research/Sleep Deprivation Detection using voice/dataset/osf/stmtf/strf_session_post_subjectNb_01_daySession_03_segmentNb_1.pkl"
)

with open(svm_path, "rb") as f:
    svm = pickle.load(f)
with open(pca_path, "rb") as f:
    pca = pickle.load(f)

# test_sample = pickle.load(test_sample_path)
with open(test_sample_path, "rb") as f:
    test_sample = pickle.load(f)

# original_shape = test_sample.shape
test_sample_flattened = np.asarray(test_sample["strf"]).flatten()
test_sample = test_sample_flattened.reshape(1, -1)

expected_features = pca.components_.shape[1]
if test_sample_flattened.shape[0] != expected_features:
    raise ValueError(
        f"Feature mismatch! Expected {expected_features} features, but got {
            test_sample_flattened.shape[0]
        }."
    )

max_test_sample = np.max(np.abs(test_sample))
if max_test_sample != 0:
    test_sample_normalized = test_sample / max_test_sample
else:
    test_sample_normalized = test_sample

test_sample_pca = pca.transform(test_sample_normalized)


# X_pca = pca.transform(test_sample_flattened.reshape(1, -1))

y_pred = svm.predict(test_sample_pca)

# 0 - post. 1 - pre
# print(f"bacc: {balanced_accuracy_score(svm.classes_, y_pred)}")
print(f"best params of the trained model: {svm.best_params_}")
print("All classes in the model:", svm.classes_)
print("Predicted class:", y_pred)
