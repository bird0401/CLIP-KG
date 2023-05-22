import numpy as np
import torch
from PIL import Image
import clip
from tqdm import tqdm
import glob, traceback, time, pickle

import logging, logging.config
from yaml import safe_load
with open("../conf/logging.yml") as f:
    cfg = safe_load(f)
logging.config.dictConfig(cfg)
logger = logging.getLogger("main")

entity_imgs = glob.glob("../imgs/*")
logger.info(f"len(entity_imgs): {len(entity_imgs)}")
entity_texts = pickle.load(open("../entity_texts.pkl", "rb"))
texts, labels = [], []

logger.info("start extracting texts")
for entity_img in tqdm(entity_imgs):
    entity_id = entity_img.split("/")[-1].split(".")[0]
    texts.append(entity_texts[entity_id])

# for i, (img, text) in enumerate(zip(entity_imgs, texts)):
#     print(img, text)
#     if i == 10:
#         break

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text_inputs = clip.tokenize(texts, truncate=True).to(device)
correct_cnt = 0

start = time.time()
logger.info("start inference")
for i, entity_img in tqdm(enumerate(entity_imgs)):
    try:
        image = preprocess(Image.open(entity_img)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        if i == np.argmax(probs[0]):
            correct_cnt += 1
    except FileNotFoundError:
        logger.info(f"File: {entity_img} is not found")
    except Exception:
        traceback.print_exc()
        continue

logger.info(f"Time: {time.time() - start}")
logger.info(f"The number of entities: {len(entity_imgs)}")
accuracy = correct_cnt / len(entity_imgs)
logger.info(f"Accuracy: {accuracy}")