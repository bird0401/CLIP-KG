import pandas as pd
from fetch import fetch

import logging, logging.config
from yaml import safe_load
with open("../conf/logging.yml") as f:
    cfg = safe_load(f)
logging.config.dictConfig(cfg)
logger = logging.getLogger("main")


def download_image(url, save_path):
    logger.info(f"url: {url}")
    res = fetch(url)
    if res:
        logger.info(f"save_path: {save_path}")
        with open(save_path, "wb") as f:
            f.write(res.content)

def download_images(df, image_dir):
    for index, row in df.iterrows():
        entity_id = row["id_us_politician"].split("/")[-1]
        download_image(row["image"], save_path = f"{image_dir}/{entity_id}.jpg")

def main():
    df = pd.read_csv("../us_politician.csv")
    logger.info(f"len(df): {len(df)}")
    download_images(df, "../imgs")

if __name__ == "__main__":
    main()