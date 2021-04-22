from load_data import load_data
from googletrans import Translator

from tqdm import tqdm
from typing import Optional


def augment_sentence(translator, record, language='en'):
    """한국어 -> {lang}언어 -> 한국어 번역을 합니다.
    
    Args:
       record: 원본 학습 데이터 하나에 대한 dictionary 입니다.
                record: {
                    "sentence": str,
                    "entity_01": str,
                    "entity_02": str,
                    "label": int
                }
        lang: 중간 번역 언어 입니다. en, ja 등을 지정할 수 있습니다.
    """
    sentence = record['sentence']
    entity01 = record['entity_01']
    entity02 = record['entity_02']

    entity_code01 = "ZQWXEC"
    entity_code02 = "QZWXEC"

    sentence = sentence.replace(entity01, entity_code01).replace(entity02, entity_code02)

    try:
        result_trans = translator.translate(sentence, src='ko', dest=language)
        result_trans_back = translator.translate(result_trans.text, src=language, dest='ko')

        aug_sentence = result_trans_back.text.replace(entity_code01, entity01).replace(entity_code02, entity02)
    except:
        print('something wrong')
        return None
    
    if entity01 in aug_sentence and entity02 in aug_sentence:    
        return {
            "sentence": aug_sentence,
            "entity_01": entity01,
            "entity_02": entity02,
            "label": record['label']
        }
    else:
        return None


if __name__ == '__main__':
    data = load_data("/opt/ml/input/data/train/train.tsv")
    translator = Translator()
    

    target_langs = ['en']

    aug_sentences = []
    for i in range(data.shape[0]):
    # for i in tqdm(range(data.shape[0]), desc="Augmenting ..."):
        aug_sentences += [augment_sentence(translator, data.iloc[i].to_dict(), language=t_lang) for t_lang in target_langs]
            
    aug_sentences = [record for record in aug_sentences if record is not None]

    print(len(aug_sentences))

    aug_data = data.append(aug_sentences)

    aug_data.to_csv("/opt/ml/input/data/train/augmented_google.tsv", index=False, header=False, sep='\t')