from pororo import Pororo
import pandas as pd


mt = Pororo(task="mt", lang="multi", model="transformer.large.multi.mtpg")

def cycling_translation_en(s, entity1, entity2) :
    s, entity1, entity2 = str(s), str(entity1), str(entity2)
    s_en = mt(s, 'ko', 'en')
    s_ko = mt(s_en, 'en', 'ko')
    if entity1 in s_ko and entity2 in s_ko :
        return s_ko
    else :
        return None


if __name__ == '__main__':
    dataset = pd.read_csv("/opt/ml/input/data/train/train.tsv", delimiter='\t', header=None)

    # print(dataset[0])

    dataset[9] = dataset.apply(lambda x: cycling_translation_en(x[1], x[2], x[5]), axis=1)

    dataset[9].to_csv("/opt/ml/input/data/train/augmented_pororo.tsv", index=False, header=False, sep='\t')

    new_df = dataset[[0, 9, 2, 3, 4, 5, 6, 7, 8]]

    new_df.to_csv("/opt/ml/input/data/train/augmented_dataset.tsv", index=False, header=False, sep='\t')