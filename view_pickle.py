import pickle
import os

data_path = os.path.expanduser(
    "~/github/modified-strf-like-model/test_strf.pkl")

# data_path = os.path.expanduser(
#     "~/Research/Sleep Deprivation Detection using voice/dataset/osf/data.pkl"
# )

with open(data_path, "rb") as f:
    data = pickle.load(f)

print(type(data))  # Check the type of data
print(data)  # Print the contents (if it's not too large)


if isinstance(data, dict):
    print("Keys:", data.keys())  # Print dictionary keys
elif isinstance(data, list):
    print("Length of list:", len(data))  # Print list length
    print("First item:", data[0])  # Print the first item
elif isinstance(data, tuple):
    print("Tuple length:", len(data))  # Print tuple length
elif isinstance(data, set):
    print("Set size:", len(data))  # Print set size

for key in data.keys():
    print(
        f"{key}: {type(data[key])}, Size: {
            len(data[key]) if hasattr(data[key], '__len__') else 'N/A'
        }"
    )

# for key, value in data.items():
#     print(f"\nKey: {key}")
#     if isinstance(value, list) or isinstance(value, tuple):
#         print("First 3 items:", value[:3])
#     elif isinstance(value, dict):
#         print("Keys:", value.keys())
#     else:
#         print("Value:", value)

# print(
#     "\nUnique subjects and their counts:",
#     np.unique(data["tabDaySession"], return_counts=True),
# )
#
print(f"session: {set(data['session'])}")
unique_day_sessions = set(data["tabDaySession"])
print(f"\nDay Session: {unique_day_sessions}")

unique_session = set(data["tabSession"])
print(f"\nSession: {unique_session}")
