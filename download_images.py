import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

def download_images(image_urls, image_dir):
    for image_url in image_urls:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        print(os.path.join(image_dir, image_url.split("/")[-1]))
        # img.save(os.path.join(image_dir, image_url.split("/")[-1]))

def main():
    df = pd.read_csv("us_politician.csv")
    download_images(df["image"], "imgs")

if __name__ == "__main__":
    main()