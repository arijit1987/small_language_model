import requests

url = "http://0.0.0.0:8000/chat"
payload = {"query": "Hi doctor, I am a 26 year old male. I am 5 feet and 9 inches tall and weigh 255 pounds. When I eat spicy food, I poop blood. Sometimes when I have constipation as well, I poop a little bit of blood. I am really scared that I have colon cancer. I do have diarrhea often. I do not have a family history of colon cancer. I got blood tests done last night. Please find my reports attached."}
headers = {"Content-Type": "application/json"}
response = requests.post(url, json=payload, headers=headers)

# Print the raw response text for debugging
#print(response.text)

# Try to parse the JSON response
try:
    print(response.json())
except requests.exceptions.JSONDecodeError as e:
    print("Failed to decode JSON response:", e)