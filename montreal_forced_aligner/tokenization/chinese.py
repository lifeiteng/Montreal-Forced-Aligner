from __future__ import annotations

import logging
import re
from typing import List

import numpy as np
from kaldialign import align as kaldi_align

try:
    import hanziconv
    import spacy_pkuseg
    from dragonmapper.hanzi import to_pinyin

    ZH_AVAILABLE = True
except ImportError:
    ZH_AVAILABLE = False
    spacy_pkuseg = None
    hanziconv = None
    to_pinyin = None


class ChineseTokenizer:
    def __init__(self, ignore_case):
        self.tokenizer = spacy_pkuseg.pkuseg(postag=True)
        self.ignore_case = ignore_case

    def __call__(self, text):
        for t in [
            "·",
            ",",
            "!",
            '"',
            "～",
            "?",
            "•",
            "‧",
        ]:  # Remove punctuation that pkuseg doesn't recognize
            text = text.replace(t, " ")
        new_text = []

        # pkuseg was trained on simplified characters
        simplified = hanziconv.HanziConv.toSimplified(text)
        is_traditional = simplified != text

        morphs = self.tokenizer.cut(simplified)
        pronunciations = []
        for normalized, pos in morphs:
            join = False
            if pos in {"w"} and normalized not in {"<", "(", "{", "["}:
                continue
            m = re.search(r"[]})>][<({[]", normalized)
            p = to_pinyin(normalized)
            if new_text and m:
                new_text[-1] += normalized[: m.start() + 1]
                normalized = normalized[m.end() - 1 :]
            elif new_text and re.match(r"^[<({\[].*", new_text[-1]):
                join = True
            elif new_text and re.match(r".*[-_~]$", new_text[-1]):
                join = True
            elif new_text and re.match(r".*[>)}\]]$", normalized):
                join = True
            elif new_text and re.match(r"^[-_~].*", normalized):
                join = True
            if new_text and any(new_text[-1].endswith(x) for x in {">", ")", "}", "]"}):
                join = False
            if join:
                new_text[-1] += normalized
                pronunciations[-1] += p
                continue
            if pos == "m":  # numerals
                for c in normalized:
                    new_text.append(c)
                    pronunciations.append(to_pinyin(c))
                continue
            new_text.append(normalized)
            pronunciations.append(p)
        assert len(new_text) == len(pronunciations)
        pronunciations = " ".join(pronunciations)
        if is_traditional:
            orig_text = "".join(text.split())
            join_text = "".join(new_text)
            lengths_cumsum = np.cumsum([len(word) for word in new_text])

            EPSILON = "※"
            ali = kaldi_align(orig_text, join_text, eps_symbol=EPSILON, sclite_mode=True)

            ts, ns = 0, 0
            for k, (t, n) in enumerate(ali):
                logging.debug(f"{k:02d}: {t} {n}")
                if t != EPSILON and n != EPSILON and t != n:
                    c = find_segment_index(lengths_cumsum, ns)
                    cs = ns - (lengths_cumsum[c - 1] if c > 0 else 0)
                    logging.debug(
                        f"{t} -> {n}: {new_text[c]} << {ns} {lengths_cumsum[c-1]} c={c} cs={cs}"
                    )
                    new_text[c] = new_text[c][:cs] + t + new_text[c][cs + 1 :]
                    logging.debug(f"           {new_text[c]}")

                if t != EPSILON:
                    ts += 1
                if n != EPSILON:
                    ns += 1

        new_text = " ".join(new_text)

        if self.ignore_case:
            pronunciations = pronunciations.lower()
        return new_text, pronunciations


def zh_spacy(ignore_case: bool = True):
    if not ZH_AVAILABLE:
        raise ImportError(
            "Please install Chinese tokenization support via `pip install spacy-pkuseg dragonmapper hanziconv`"
        )
    return ChineseTokenizer(ignore_case)


def find_segment_index(segment_lengths_cumsum: List[int], word_idx: int) -> int:
    """
    Given a list of segment lengths and a word index, find the segment index where the word belongs to.
    """
    segment_idx = np.searchsorted(segment_lengths_cumsum, word_idx, side="right")
    return segment_idx


if __name__ == "__main__":
    tokenizer = zh_spacy()
    for text in [
        "易居中国钜派投资成功赴※美上市",
        "易居 中国 钜 派 投资 成功 赴美 上市",
        "易居～ 中国? 钜• 派 巒讀讅 投资 成功 讀 how 赴美 上市 clean",
    ]:
        simplified = hanziconv.HanziConv.toSimplified(text)

        new_text, pronunciations = tokenizer(text)
        clean_text = re.sub(r"[^\w]", "", text)
        logging.info(f"  Original: {text} -> {clean_text}")
        logging.info(f"Simplified: {new_text}")

        assert clean_text == re.sub(r"[^\w]", "", new_text)
        logging.info("++++++++++++++++++++++++++++")
