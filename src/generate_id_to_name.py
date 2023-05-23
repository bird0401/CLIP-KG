import pandas as pd
import pickle

import logging, logging.config
from yaml import safe_load
with open("../conf/logging.yml") as f:
    cfg = safe_load(f)
logging.config.dictConfig(cfg)
logger = logging.getLogger("main")

def create_id_to_name(df):
    id_to_name = {}
    for index, row in df.iterrows():
        id_to_name[row["id_us_politician"].split("/")[-1]] = row["name"]
    return id_to_name

def main():
    df = pd.read_csv("../us_politician.csv")
    id_to_name = create_id_to_name(df)
    logger.info(len(id_to_name))
    pickle.dump(id_to_name, open("../id_to_name.pkl", "wb"))

if __name__ == "__main__":
    main()