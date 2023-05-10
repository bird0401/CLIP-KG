import numpy as np
import torch
from PIL import Image
import clip
import glob

entity_dirs = glob.glob("dogs/*/")
entities, texts, image_file_names = [], [], []
for entity_dir in entity_dirs:
    entity = entity_dir.split("/")[1]
    entities.append(entity)
    entity_text = open(f"dogs/{entity}/entity_page.txt", "r").read()
    texts.append(entity_text)
    image_file_names.append(f"{entity_dir}/image.jpg")

# print(entities, texts, image_file_names)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(texts).to(device)
correct_cnt = 0
for i, image_file_name in enumerate(image_file_names):
    image = preprocess(Image.open(image_file_name)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # print(f"Correct entity: {entities[i]}")
    # print(f"Pred entity: {entities[np.argmax(probs)]}")
    if entities[i] == entities[np.argmax(probs)]:
        correct_cnt += 1
    # print("Similarities between input image and text descriptions:")
    # for entity, prob in zip(entities, probs[0]):
    #     print(f"{entity}: {prob:.3f}")
    # print()
accuracy = correct_cnt / len(entities)
print(f"Accuracy on {len(entities)} entities: {accuracy}")
