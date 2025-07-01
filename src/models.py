import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.config.settings import settings


class DocsSection(BaseModel):
    url: str
    title: str
    content: str
    section_id: str = Field(default_factory=lambda: f"section_{hash(datetime.now())}")
    page_num: Optional[int] = None
    word_count: int = Field(default=0, exclude=True)
    source_document: Optional[str] = None
    was_used_in_context: bool = Field(default=False, exclude=True)

    @field_validator("content")
    def clean_content(cls, v):
        v = " ".join(v.split())
        v = v.replace("\x00", "").replace("\ufffd", "").strip()
        return v

    @field_validator("word_count", mode="before")
    def compute_word_count(cls, v, values):
        if "content" in values:
            return len(values["content"].split())
        return v

    def embedding_content(self) -> str:
        return f"{self.title}\n\n{self.content}"

    def display_content(self) -> str:
        page_info = f" (Page {self.page_num})" if self.page_num else ""
        return f"**{self.title}**{page_info}\n{self.content}"


class DocStore(BaseModel):
    sections: List[DocsSection]
    embeddings: List[np.ndarray]
    created_at: datetime
    doc_hash: str
    source_document: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def get_cache_path(self) -> Path:
        if self.doc_hash == "multi_pdf_store":
            return settings.CACHE_DIR / "multi_pdf_store.pkl"
        return settings.CACHE_DIR / f"docstore_{self.doc_hash}.pkl"

    def save_to_cache(self):
        try:
            embeddings_list = [emb.tolist() for emb in self.embeddings]
            store_data = {
                "sections": [sect.dict() for sect in self.sections],
                "embeddings": embeddings_list,
                "created_at": self.created_at,
                "doc_hash": self.doc_hash,
            }
            with open(self.get_cache_path(), "wb") as f:
                pickle.dump(store_data, f)
        except Exception as e:
            print(f"Error saving to cache: {e}")

    @classmethod
    def load_from_cache(cls, doc_hash: str) -> Optional["DocStore"]:
        cache_file = settings.CACHE_DIR / f"docstore_{doc_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    store_data = pickle.load(f)
                    embeddings = [np.array(emb) for emb in store_data["embeddings"]]
                    sections = [DocsSection(**sect) for sect in store_data["sections"]]

                    return cls(
                        sections=sections,
                        embeddings=embeddings,
                        created_at=store_data["created_at"],
                        doc_hash=store_data["doc_hash"],
                    )
            except Exception as e:
                print(f"Error loading from cache: {e}")
        return None
