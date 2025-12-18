from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str

class CorpusLoader:
    def __init__(self, corpus_dir: str, encoding: str = "utf-8"):
        self.corpus_dir = Path(corpus_dir)
        self.encoding = encoding

    def load_txt(self) -> List[Document]:
        files = sorted(self.corpus_dir.glob("*.txt"))
        docs: List[Document] = []
        for fp in files:
            text = fp.read_text(encoding=self.encoding, errors="ignore")
            docs.append(Document(doc_id=fp.stem, text=text))
        return docs

    def to_dict(self, docs: List[Document]) -> List[Dict]:
        return [{"doc_id": d.doc_id, "text": d.text} for d in docs]
