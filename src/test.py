import numpy as np
import torch
from PIL import Image
import clip
from tqdm import tqdm
import glob, traceback, time

import logging, logging.config
from yaml import safe_load
with open("../conf/logging.yml") as f:
    cfg = safe_load(f)
logging.config.dictConfig(cfg)
logger = logging.getLogger("main")

import pickle

def load_sentences():
    sentences_by_entity = pickle.load(open("../sentences_by_entity.pkl", "rb"))
    return sentences_by_entity

def main():
    sentences_by_entity = load_sentences()
    print(len(sentences_by_entity))
    for i, (k, v) in enumerate(sentences_by_entity.items()):
        print(k)
        print(v)
        if i == 10: break


if __name__ == "__main__":
    main()

# entity_dirs = glob.glob("dogs/*/")
# # entity_dirs = glob.glob("wikipedia/*/")
# # entity_dirs = entity_dirs[:10] # For testing
# texts, labels = [], []

# logger.info("start extracting texts")
# for entity_dir in tqdm(entity_dirs):
#     entity_text = open(f"{entity_dir}/entity_page.txt", "r").read()
#     texts.append(entity_text)
#     labels.append(entity_dir.split("/")[1])

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {l}") for l in labels]).to(device)
# # text_inputs = clip.tokenize(texts, truncate=True).to(device)
# correct_cnt = 0

# start = time.time()
# logger.info("start inference")
# for i, entity_dir in tqdm(enumerate(entity_dirs)):
#     try:
#         file_name = f"{entity_dir}image.jpg"
#         image = preprocess(Image.open(file_name)).unsqueeze(0).to(device)
#         with torch.no_grad():
#             logits_per_image, logits_per_text = model(image, text_inputs)
#             probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#         if i == np.argmax(probs[0]):
#             correct_cnt += 1
#     except FileNotFoundError:
#         logger.info(f"File: {file_name} is not found")
#     except Exception:
#         traceback.print_exc()
#         continue

# logger.info(f"Time: {time.time() - start}")
# logger.info(f"The number of entities: {len(entity_dirs)}")
# accuracy = correct_cnt / len(entity_dirs)
# logger.info(f"Accuracy: {accuracy}")

# # print("Similarities between input image and text descriptions:")
# # top5_indices = np.argsort(probs[0])[::-1][:5]
# # for index in top5_indices:
# #     print(f"{entities[index]}: {probs[0][index]:.3f}")