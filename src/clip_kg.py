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


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def main():
    entity_imgs = glob.glob("../imgs/*")
    logger.info(f"len(entity_imgs): {len(entity_imgs)}")
    entity_texts = pickle.load(open("../entity_texts_shorter.pkl", "rb"))
    # entity_texts = pickle.load(open("../entity_texts.pkl", "rb"))
    logger.info(f"len(entity_texts): {len(entity_texts)}")
    texts, labels = [], []

    logger.info("start extracting texts")
    for entity_img in tqdm(entity_imgs):
        entity_id = entity_img.split("/")[-1].split(".")[0]
        texts.append(entity_texts[entity_id])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_inputs = clip.tokenize(texts, truncate=True).to(device)
    # correct_cnt = 0
    top1, top5 = 0., 0.

    start = time.time()
    logger.info("start inference")
    for i, entity_img in enumerate(entity_imgs):
        try:
            image = preprocess(Image.open(entity_img)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text_inputs)
                # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            print(logits_per_image)
            print(logits_per_image.topk(max(5), 1, True, True))
            print(i)
            print()
            acc1, acc5 = accuracy(logits_per_image, i, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            # if i == np.argmax(probs[0]):
            #     correct_cnt += 1
            if i % 10 == 0:
                logger.info(f"acc1 at {i}th entity: {(top1 / (i + 1)) * 100}")
            if i % 10 == 0:
                logger.info(f"acc5 at {i}th entity: {(top5 / (i + 1)) * 100 }")
        except FileNotFoundError:
            logger.info(f"File: {entity_img} is not found")
        except Exception:
            traceback.print_exc()
            continue

    logger.info(f"Time: {time.time() - start}")
    logger.info(f"The number of entities: {len(entity_imgs)}")
    # logger.info(f"Accuracy: {correct_cnt / len(entity_imgs)}")
    logger.info(f"Top-1 accuracy: {(top1 / (i + 1)) * 100:.2f}")
    logger.info(f"Top-5 accuracy: {(top5 / (i + 1)) * 100:.2f}")


if __name__ == "__main__":
    main()