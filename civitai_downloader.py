import os
import requests

class CivitaiAPIError(Exception):
    pass

def download_lora_from_civitai(model_id: str, api_key: str):
    """
    Downloads a LoRA model from Civitai.

    Args:
        model_id: The ID of the model to download.
        api_key: The user's Civitai API key.
    """
    LORA_DIR = "loras"
    if not os.path.exists(LORA_DIR):
        os.makedirs(LORA_DIR)

    api_url = f"https://civitai.com/api/v1/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        model_data = response.json()

        model_version = model_data.get("modelVersions")[0]
        download_url = model_version.get("downloadUrl")
        file_name = model_version.get("files")[0].get("name")

        if not download_url:
            raise CivitaiAPIError("Could not find download URL in model data.")

        download_response = requests.get(download_url, stream=True)
        download_response.raise_for_status()

        file_path = os.path.join(LORA_DIR, file_name)
        with open(file_path, "wb") as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"Successfully downloaded {file_name} to {file_path}"

    except requests.exceptions.RequestException as e:
        raise CivitaiAPIError(f"Network error: {e}")
    except Exception as e:
        raise CivitaiAPIError(f"An unexpected error occurred: {e}")
