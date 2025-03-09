import os
import joblib

# data_path = os.path.expanduser(
#     "~/Research/Sleep Deprivation Detection using voice/output/day_session/models/pca_combined.pkl"
# )

data_path = os.path.expanduser(
    "~/Research/Sleep Deprivation Detection using voice/output/day_session/models/svm_combined.pkl"
)

data = joblib.load(data_path)

print(type(data))
print(data)
