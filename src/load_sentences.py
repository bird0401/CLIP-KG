import logging, logging.config
from yaml import safe_load
with open("../conf/logging.yml") as f:
    cfg = safe_load(f)
logging.config.dictConfig(cfg)
logger = logging.getLogger("main")

import pickle

def main():
    entity_texts = pickle.load(open("../entity_texts.pkl", "rb"))
    print(len(entity_texts))
    # for i, (k, v) in enumerate(entity_texts.items()):
    #     print(k)
    #     print(v)
    #     print()
    #     if i == 10: break

if __name__ == "__main__":
    main()
