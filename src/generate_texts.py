import pandas as pd
import pickle

import logging, logging.config
from yaml import safe_load
with open("../conf/logging.yml") as f:
    cfg = safe_load(f)
logging.config.dictConfig(cfg)
logger = logging.getLogger("main")

def create_texts(df):
    df = df.drop("image", axis=1)
    texts_by_entity = {}
    for index, row in df.iterrows():
        prompt = f"a photo of {row['name']}"
        for column, value in row.items():
            # if column != "id_us_politician" and column != "name":
            #     prompt += f", {column} is {value}"
            if column == "sex_or_gender":
                prompt += f", {column} is {value}"
        entity_id = row["id_us_politician"].split("/")[-1]
        texts_by_entity[entity_id] = f"{prompt}."
    return texts_by_entity

def main():
    df = pd.read_csv("../us_politician.csv")
    texts_by_entity = create_texts(df)
    logger.info(len(texts_by_entity))
    pickle.dump(texts_by_entity, open("../entity_texts_shorter.pkl", "wb"))
    # pickle.dump(texts_by_entity, open("../entity_texts.pkl", "wb"))

if __name__ == "__main__":
    main()