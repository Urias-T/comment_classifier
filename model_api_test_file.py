"""
Used to test the model api.

This is to be run after the uvicorn server has been started.
"""

# Please leave terminal open and uvicorn server running before running this script in an IDE or
# separate terminal.

import requests

url = "http://127.0.0.1:8000/predict"  # Local URL (do not change)

test_comment = {
    "raw_comment": "Jesus is Lord."  # Do not change "raw_comment" key
}

response = requests.post(url=url, json=test_comment)

print(response.text)
