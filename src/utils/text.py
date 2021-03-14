import re
from typing import List, Optional

import mojimoji
from ginza import *
import spacy

def detokenize(tk_list : List[str]) -> List[str]:
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

def ascii_print(text : str) -> None:
    text = text.encode("ascii", "ignore")
    print(text)

class TextCleaner:
    # タイトルの括弧内に含まれていたら削除
    pettern = ["発売", "版", "書籍", "コミカライズ", "完結", "受賞", "配信", "決定", "刊行", "予定", "マンガ", "開始", "連載", "アニメ化"]
    pattern = re.compile(r'|'.join(pettern))

    story_drop_pettern = [
        "発売中", "発売予定", "連載中", "連載予定", "配信中", "配信予定",
        "書籍化", "書籍版", "Web版", "WEB版", "web版", "コミカライズ", "マンガ化", "アニメ化",
        "完結",  "公式サイト", "公式Twitter", "特設サイト",
        "転載", "刊行", "投稿中", "お願いします", "申し訳ありません", "参照ください",
        "累計", "公開中",
    ]
    story_drop_pettern = re.compile(r'|'.join(story_drop_pettern))
    nlp = spacy.load('ja_ginza')

    @classmethod
    def clean(cls, text: str) -> str:
        # urlの削除
        text = re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "" , text)

        # 全角に変換
        text = mojimoji.han_to_zen(text)

        # 英数字のみ半角
        replaces = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz;:/."
        text = text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94) if chr(0x21+i) in replaces}))

        #改行をスペースに置換
        text = re.sub(r"[\r\n]", " ", text)
        # 全角スペース，半角スペース，タブを半角スペースに置換
        text = re.sub(r"[\u3000 \t]", " ", text)

        # 【】,（）の文章抽出
        opts = re.findall("【.+?】", text) + re.findall("（.+?）", text) + re.findall("〈.+?〉", text) + re.findall("《.+?》", text)
        for opt in opts:
            if bool(cls.pattern.search(opt)):
                text = text.replace(opt, '')

        # 記号の削除
        text = re.sub(r"[◆★●❖]", " ", text)
        doc = cls.nlp(text)
        for sent in doc.sents:
            sent = str(sent)
            if bool(cls.story_drop_pettern.search(sent)):
                text = text.replace(sent, '')

        # 2連続以上のスペースを1つにする
        text = re.sub(r"\s+", " ", text)

        return text