import tinker
import requests

# Create service client
service_client = tinker.ServiceClient()
rest_client = service_client.create_rest_client()

# Get model path from your training logs or wandb
model_path = "3a6a95f1-0804-5c83-a156-b272e36f088b:train:0"

# Download checkpoint
url = rest_client.get_checkpoint_archive_url_from_tinker_path(model_path)
print(f"Download URL: {url}")

# Use the URL to download manually or with requests
import requests
response = requests.get(url)
with open("model-checkpoint.tar.gz", "wb") as f:
    f.write(response.content)
print("Checkpoint downloaded successfully!")
