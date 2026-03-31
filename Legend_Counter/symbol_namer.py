import os
import base64
import requests
import re

# ==============================
# CONFIG
# ==============================
API_KEY = "ak_OFIKwiNCW2UcDWkLrVRhMR-tVb9SIwaGvGeGueDk1tM"
BASE_URL = "http://localhost:8000"
FOLDER_PATH = "output/small_test/symbols"  # folder containing images
MODEL_NAME = "qwen-3.5vl-Q4"

# ==============================
# HELPER FUNCTIONS
# ==============================

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def clean_filename(name):
    # Remove invalid characters for Windows filenames
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.strip().replace(" ", "_")
    return name

def ask_ai_for_symbol_name(legend_base64, symbol_base64):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = """
You are a strict symbol identification assistant.

You will receive:
1) A legend table image.
2) A cropped symbol image.

Your task:
Find the exact name of the cropped symbol from the legend.
Return ONLY the symbol name.
No explanation.
No extra words.
"""

    data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{legend_base64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{symbol_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": 50
    }

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )

    response.raise_for_status()
    result = response.json()

    name = result["choices"][0]["message"]["content"].strip()
    return name

# ==============================
# MAIN LOGIC
# ==============================

def main(folder_path=None):
    if folder_path is None:
        folder_path = FOLDER_PATH

    legend_path = os.path.join(folder_path, "legend_table.png")

    if not os.path.exists(legend_path):
        print(f"❌ {legend_path} not found.")
        return

    legend_base64 = encode_image(legend_path)

    for file in os.listdir(folder_path):
        if file.startswith("symbol_") and file.endswith(".png"):

            symbol_path = os.path.join(folder_path, file)
            print(f"\nProcessing: {file}")

            symbol_base64 = encode_image(symbol_path)

            try:
                symbol_name = ask_ai_for_symbol_name(
                    legend_base64,
                    symbol_base64
                )

                clean_name = clean_filename(symbol_name)
                new_filename = f"{clean_name}.png"
                new_path = os.path.join(folder_path, new_filename)

                # Avoid overwrite
                if os.path.exists(new_path):
                    print(f"⚠ File {new_filename} already exists. Skipping.")
                    continue

                os.rename(symbol_path, new_path)
                print(f"✅ Renamed to: {new_filename}")

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

if __name__ == "__main__":
    main()
