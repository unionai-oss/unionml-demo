import json
import matplotlib.pyplot as plt
import requests
from sklearn.datasets import load_digits

from pictionary_app.dataset import QuickDrawDataset, get_quickdraw_class_names


dataset = QuickDrawDataset("/tmp/quickdraw_data", 500, class_limit=10)
class_names = get_quickdraw_class_names()
index = 200
features, target = dataset[index]

# generate predictions
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": features.tolist()},
)

predictions = dict(reversed(sorted(response.json().items(), key=lambda x: x[1])))
print(f"Ground truth: {class_names[index]}")
print("Predictions:")
print(json.dumps(predictions, indent=4))
plt.imshow(features[0])
plt.show()
