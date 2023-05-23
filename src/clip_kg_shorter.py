import glob, pickle
from clip_matching.util import *

import logging, logging.config
from yaml import safe_load
with open("../conf/logging.yml") as f:
    cfg = safe_load(f)
logging.config.dictConfig(cfg)
logger = logging.getLogger("main")

def main():
    entity_imgs = glob.glob("../imgs/*")
    # entity_imgs = entity_imgs[:1000] # for debug
    logger.info(f"len(entity_imgs): {len(entity_imgs)}")

    entity_id_to_text = pickle.load(open("../entity_texts_shorter.pkl", "rb"))
    texts = create_texts_from_structured_data(entity_imgs, entity_id_to_text)
    logger.info(f"len(texts): {len(texts)}")
    
    inference(entity_imgs, texts)

if __name__ == "__main__":
    main()