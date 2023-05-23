import torch
from PIL import Image
import clip
from tqdm import tqdm
import traceback, time
from PIL import UnidentifiedImageError

import logging, logging.config
from yaml import safe_load
with open("../conf/logging.yml") as f:
    cfg = safe_load(f)
logging.config.dictConfig(cfg)
logger = logging.getLogger("util")

def accuracy(output, target, topk = (1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def inference(entity_imgs, texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text_inputs = clip.tokenize(texts, truncate=True).to(device)
    cnt_top1, cnt_top5, n = 0, 0, 0

    start = time.time()
    logger.info("start inference")
    for i, entity_img in enumerate(entity_imgs):
        try:
            image = preprocess(Image.open(entity_img)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text_inputs)
                target = torch.tensor([i]).to(device)
            has_target_in_top1, has_target_in_top5 = accuracy(logits_per_image, target, topk=(1, 5))
            cnt_top1 += has_target_in_top1
            cnt_top5 += has_target_in_top5
            n += 1
            if n % 100 == 0:
                logger.info(f"acc1 at {n}th entity: {(cnt_top1 / n)}")
                logger.info(f"acc5 at {n}th entity: {(cnt_top5 / n)}")
                print()
        except FileNotFoundError:
            logger.info(f"File: {entity_img} is not found")
        except UnidentifiedImageError:
            logger.info(f"File: {entity_img} is not identified")
        except Exception:
            traceback.print_exc()
            continue

    logger.info(f"Time: {time.time() - start}")
    logger.info(f"The number of entities: {n}")
    logger.info(f"Top-1 accuracy: {(cnt_top1 / n)}")
    logger.info(f"Top-5 accuracy: {(cnt_top5 / n)}")

def create_baseline_texts(entity_imgs, id_to_name):
    texts = []
    logger.info("start extracting texts")
    for entity_img in tqdm(entity_imgs):
        entity_id = entity_img.split("/")[-1].split(".")[0]
        texts.append(f"a photo of a {id_to_name[entity_id]}")
    return texts

def create_texts_from_structured_data(entity_imgs, entity_id_to_text):
    texts = []
    logger.info("start extracting texts")
    for entity_img in tqdm(entity_imgs):
        entity_id = entity_img.split("/")[-1].split(".")[0]
        texts.append(entity_id_to_text[entity_id])
    return texts