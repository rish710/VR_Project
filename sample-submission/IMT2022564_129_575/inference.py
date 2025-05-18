import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import os
import requests
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel

# Correct base URL for raw file access
MODEL_BASE_URL = "https://huggingface.co/rish710/rishit_model/resolve/main/"
ADAPTER_DIR = "lora_adapter_final"

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_model():
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    adapter_path = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")

    adapter_url = MODEL_BASE_URL + "adapter_model.safetensors"
    config_url = MODEL_BASE_URL + "adapter_config.json"

    if not os.path.exists(adapter_path):
        print("Downloading adapter weights...")
        download_file(adapter_url, adapter_path)

    if not os.path.exists(config_path):
        print("Downloading adapter config...")
        download_file(config_url, config_path)

def normalize_answer(s):
    s = str(s).lower().strip()
    s = ''.join(c for c in s if c.isalnum() or c.isspace())
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # Download adapter model if not already present
    download_model()


    # Load processor and base model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    base_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    #print("LoRA adapter loaded:", model.peft_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    generated_answers = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.image_dir, row['image_name'])
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = f"Question: {question} Answer:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=10)
                answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(e)
            answer = "error"

        # Keep just the first word, lowercase
        print(f"answer: {answer}")
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()
