import requests

# POST sends data to the server (unlike GET which retrieves data)
payload = {"username": "Olivia", "password": "123"}
response = requests.post("https://httpbin.org/post", data=payload)

data = response.json()
print(f"Status: {response.status_code}")
print("\nThe server received this form data:")
print(data["form"])

print(f"\nOur IP address: {data['origin']}")
