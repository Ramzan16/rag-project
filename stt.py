import requests

# Replace with your actual Mistral API key
API_KEY = "VOXTRAL_API_KEY"

url = "https://api.mistral.ai/v1/audio/transcriptions"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Audio file to transcribe
files = {
    "file": (r"C:\Users\Ramzan.Agriya\Downloads\Prophecy.mp3", open(r"C:\Users\Ramzan.Agriya\Downloads\Prophecy.mp3", "rb"), "mp3")
}

# Optional: specify transcription model
data = {
    "model": "voxtral-mini-latest"
}

response = requests.post(url, headers=headers, files=files, data=data)

print(response.status_code)
print(response.json())
