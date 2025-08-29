from huggingface_hub import snapshot_download, hf_hub_download

from dotenv import load_dotenv
import os
load_dotenv() 

token = os.getenv("HF_TOKEN", None)


def download_flux():
    snapshot_download("black-forest-labs/FLUX.1-dev",
                      allow_patterns=["flux1-dev.safetensors", "ae.safetensors"],
                      local_dir="./weights/FLUX.1-dev",
                      local_dir_use_symlinks=False,
                      token=token)
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
