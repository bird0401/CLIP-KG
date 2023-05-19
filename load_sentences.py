import pickle

def load_sentences():
    sentences_by_entity = pickle.load(open("sentences_by_entity.pkl", "rb"))
    return sentences_by_entity

def main():
    sentences_by_entity = load_sentences()
    print(len(sentences_by_entity))
    for i, (k, v) in enumerate(sentences_by_entity.items()):
        print(v)
        print(k)
        if i == 10: break


if __name__ == "__main__":
    main()