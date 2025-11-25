from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import spacy

from .data_loader import Chunk


def _clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass
class SentenceRecord:
    text: str
    chunk_id: int
    sentence_id: int


@dataclass
class NodeRecord:
    node_id: int
    label: str
    occurrences: List[Tuple[int, int]]  # (chunk_id, sentence_id)


class NodeExtractor:
    def __init__(self, min_frequency: int = 1) -> None:
        self.min_frequency = min_frequency
        self._nlp = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer"])
        self._nlp.enable_pipe("senter")

    def extract(self, chunks: Sequence[Chunk]) -> Tuple[List[NodeRecord], List[SentenceRecord]]:
        sentences: List[SentenceRecord] = []
        counts: Counter[str] = Counter()
        occurrences: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for chunk_id, chunk in enumerate(chunks):
            doc = self._nlp(chunk.content)
            sent_info: List[Tuple[int, int]] = []
            for sent_id, sent in enumerate(doc.sents):
                sent_text = _clean(sent.text)
                if not sent_text:
                    continue
                sentences.append(SentenceRecord(sent_text, chunk_id, sent_id))
                sent_info.append((sent.start, sent.end))

            for ent in doc.ents:
                label = _clean(ent.text)
                if not label:
                    continue
                sent_index = self._find_sentence_index(ent.start, sent_info)
                if sent_index is None:
                    continue
                occurrences[label].append((chunk_id, sent_index))
                counts[label] += 1

            for noun in doc.noun_chunks:
                label = _clean(noun.text)
                if not label:
                    continue
                sent_index = self._find_sentence_index(noun.start, sent_info)
                if sent_index is None:
                    continue
                occurrences[label].append((chunk_id, sent_index))
                counts[label] += 1

        nodes: List[NodeRecord] = []
        node_id = 0
        for label, freq in counts.most_common():
            if freq < self.min_frequency:
                continue
            nodes.append(NodeRecord(node_id=node_id, label=label, occurrences=occurrences[label]))
            node_id += 1
        return nodes, sentences

    @staticmethod
    def _find_sentence_index(token_start: int, ranges: Iterable[Tuple[int, int]]) -> int | None:
        for idx, (start, end) in enumerate(ranges):
            if start <= token_start < end:
                return idx
        return None
