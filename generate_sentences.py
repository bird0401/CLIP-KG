import pandas as pd
import pickle

def create_sentences(df):
    df = df.drop("image", axis=1)
    sentences_by_entity = {}
    for index, row in df.iterrows():
        prompt = f"a photo of {row['name']}"
        for column, value in row.items():
            if column != "id_us_politician" and column != "name":
                prompt += f", {column} is {value}"
        sentences_by_entity[row["id_us_politician"]] = f"{prompt}."
    # for k, v in sentences_by_entity.items():
    #     print(k)
    #     print(v)
    return sentences_by_entity

def main():
    df = pd.read_csv("us_politician.csv")
    sentences_by_entity = create_sentences(df)
    print(len(sentences_by_entity))
    pickle.dump(sentences_by_entity, open("sentences_by_entity.pkl", "wb"))

if __name__ == "__main__":
    main()