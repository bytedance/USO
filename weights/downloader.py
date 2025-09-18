from huggingface_hub import snapshot_download, hf_hub_download

from dotenv import load_dotenv
import os
load_dotenv() 

token = os.getenv("HF_TOKEN", None)


def download_flux():
    # Since user has access to FLUX.1-dev, try it first
    try:
        print("Attempting to download FLUX.1-dev (you have access)...")
        snapshot_download("black-forest-labs/FLUX.1-dev",
                          allow_patterns=["flux1-dev.safetensors"],
                          local_dir="./weights/FLUX.1-dev",
                          local_dir_use_symlinks=False,
                          token=token)
        print("Successfully downloaded FLUX.1-dev!")
    except Exception as e:
        print(f"FLUX.1-dev download failed: {e}")
        print("Trying FLUX.1-schnell as fallback...")
        try:
            snapshot_download("black-forest-labs/FLUX.1-schnell",
                              allow_patterns=["flux1-schnell.safetensors"],
                              local_dir="./weights/FLUX.1-schnell",
                              local_dir_use_symlinks=False,
                              token=token)
            print("Successfully downloaded FLUX.1-schnell!")
        except Exception as e2:
            print(f"FLUX.1-schnell download also failed: {e2}")
            print("Please ensure you have access to at least one FLUX model")
            return
    
    # Download AE (autoencoder) - usually available
    try:
        print("Downloading AE (autoencoder)...")
        snapshot_download("black-forest-labs/FLUX.1-dev",
                          allow_patterns=["ae.safetensors"],
                          local_dir="./weights/FLUX.1-dev",
                          local_dir_use_symlinks=False,
                          token=token)
        print("Successfully downloaded AE!")
    except Exception as e:
        print(f"AE download failed: {e}")
        print("Note: AE might already be downloaded or you may need to request access to FLUX.1-dev")


def update_env_for_dev():
    """Update .env file to use dev variant paths"""
    env_path = ".env"
    try:
        with open(env_path, "r") as f:
            content = f.read()
        
        # Replace schnell paths with dev paths
        content = content.replace(
            "FLUX_DEV=./weights/FLUX.1-schnell/flux1-schnell.safetensors",
            "FLUX_DEV=./weights/FLUX.1-dev/flux1-dev.safetensors"
        )
        content = content.replace(
            "FLUX_DEV_FP8=./weights/FLUX.1-schnell/flux1-schnell.safetensors", 
            "FLUX_DEV_FP8=./weights/FLUX.1-dev/flux1-dev.safetensors"
        )
        content = content.replace(
            "FLUX_SCHNELL=./weights/FLUX.1-schnell/flux1-schnell.safetensors",
            "FLUX_SCHNELL=./weights/FLUX.1-dev/flux1-dev.safetensors"
        )
        
        with open(env_path, "w") as f:
            f.write(content)
        
        print("Updated .env to use FLUX.1-dev paths")
    except Exception as e:
        print(f"Failed to update .env file: {e}")
# optional 
def download_flux_krea():
    snapshot_download("black-forest-labs/FLUX.1-Krea-dev",
                    allow_patterns=["flux1-krea-dev.safetensors"],
                    local_dir="./weights/FLUX.1-Krea-dev",
                    local_dir_use_symlinks=False,
                    token=token)

def download_uso():
    snapshot_download("bytedance-research/USO",
                      local_dir="./weights/USO",
                      local_dir_use_symlinks=False)

def download_t5():
    for f in ["config.json", "tokenizer_config.json", "special_tokens_map.json",
              "spiece.model", "pytorch_model.bin"]:
        hf_hub_download("google/t5-v1_1-xxl", f, local_dir="./weights/t5-xxl",
                        local_dir_use_symlinks=False)

def download_clip():
    for f in ["config.json", "merges.txt", "vocab.json",
              "tokenizer_config.json", "special_tokens_map.json",
              "pytorch_model.bin"]:
        hf_hub_download("openai/clip-vit-large-patch14", f,
                        local_dir="./weights/clip-vit-l14",
                        local_dir_use_symlinks=False)

def download_siglip():
    snapshot_download("google/siglip-so400m-patch14-384",
                      local_dir="./weights/siglip",
                      local_dir_use_symlinks=False)

if __name__ == "__main__":
    download_uso()
    download_flux()
    # download_flux_krea()
    download_t5()
    download_clip()
    download_siglip()
