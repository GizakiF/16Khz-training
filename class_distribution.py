import numpy as np
import pickle
from collections import defaultdict
import os

# Load Data
data_path = os.path.expanduser(
    "~/Research/Sleep Deprivation Detection using voice/dataset/osf/data.pkl")
with open(data_path, "rb") as f:
    data = pickle.load(f)

# Extract labels and day sessions
tabSession = np.asarray(data['tabSession'])  # Labels ('pre', 'post')
tabDaySession = np.asarray(data['tabDaySession'])  # Day session numbers

# Determine class distribution
class_distribution = defaultdict(lambda: defaultdict(int))

for day, label in zip(tabDaySession, tabSession):
    class_distribution[day][label] += 1

# Print results
print("\n📊 **Class Distribution Per Day Session**")
for day in sorted(class_distribution.keys()):
    print(f"🔍 **Day {day}**: {dict(class_distribution[day])}")
