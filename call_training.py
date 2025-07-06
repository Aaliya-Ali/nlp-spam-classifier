# import requests
# import json

# # Define the Flask API endpoint
# url = 'http://127.0.0.1:5000/training'

# # Optional: Define hyperparameter grid (can leave empty `{}` for default)
# hyperparams = {
#     "n_estimators": [100, 200],
#     "max_depth": [10, 20, None],
#     "min_samples_split": [2, 5],
#     "min_samples_leaf": [1, 2]
# }

# # Send POST request with optional hyperparameters
# response = requests.post(url, json=hyperparams)

# # Print status code and response JSON
# print("Status Code:", response.status_code)
# print("Response JSON:", json.dumps(response.json(), indent=4))

import requests
import json

# Define the URL of the Flask app's training endpoint
url = "http://127.0.0.1:5000/training"

# Optional: Define hyperparameter grid (can leave empty {})
hyperparams = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# Send POST request with hyperparameters as JSON
response = requests.post(url, json=hyperparams)

# Print the response from the server
print("Status Code:", response.status_code)

try:
    print("Response JSON:")
    print(json.dumps(response.json(), indent=4))
except Exception as e:
    print("Error parsing response:", str(e))
