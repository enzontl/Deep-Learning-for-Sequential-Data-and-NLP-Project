"""
Personne 3 - Moteur de recherche dans la banque de clauses
==========================================================
Trois modes de recherche :
  - "dense"  : par le SENS (sentence-transformers, similarite cosinus)
  - "bm25"   : par les MOTS exacts (BM25), garde les chiffres ("30" != "60")
  - "hybrid" : les deux classements fusionnes (RRF)
Un filtre optionnel par categorie fonctionne avec les trois modes.

Lancer une demonstration (depuis la racine du projet) :
    uv run python -m src.retrieval --max-bank 2000
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Dossier ou l'on range les empreintes calculees (evite de les recalculer a chaque fois)
RETRIEVAL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "retrieval"

MODES = ("dense", "bm25", "hybrid")


def tokenize(text: str) -> List[str]:
    """Decoupe un texte en mots en minuscules. Les chiffres sont GARDES (important pour BM25)."""
    return re.findall(r"[a-z0-9]+", text.lower())


class ClauseBankRetriever:
    """Moteur de recherche Sentence-Transformer + BM25 sur la banque de clauses."""

    def __init__(
        self,
        corpus_texts: List[str],
        corpus_labels: np.ndarray,
        label_names: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = RETRIEVAL_DIR,
        batch_size: int = 64,
    ):
        # Les imports sont ici pour que "import src.retrieval" reste rapide
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.texts = list(corpus_texts)
        self.labels = np.asarray(corpus_labels, dtype=np.int64)
        self.label_names = list(label_names)
        self.model_name = model_name

        # 1) Moteur "sens" : une empreinte numerique par clause
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_compute_embeddings(cache_dir, batch_size)

        # 2) Moteur "mots" : index BM25
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

    # ------------------------------------------------------------------ empreintes
    def _load_or_compute_embeddings(self, cache_dir: Optional[Path], batch_size: int) -> np.ndarray:
        cache_file = None
        if cache_dir is not None:
            model_slug = self.model_name.replace("/", "_")
            cache_file = Path(cache_dir) / f"bank_embeddings_{model_slug}_{len(self.texts)}.npy"
            if cache_file.exists():
                cached = np.load(cache_file)
                if cached.shape[0] == len(self.texts):
                    return cached
        embeddings = self.model.encode(
            self.texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # vecteurs de longueur 1 : cosinus = produit scalaire
            show_progress_bar=True,
        ).astype(np.float32)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, embeddings)
        return embeddings

    # ------------------------------------------------------------------ utilitaires
    def _label_ids(self, label_filter: Optional[Sequence[Union[int, str]]]) -> Optional[np.ndarray]:
        """Accepte des numeros de classe ou des noms de classe."""
        if label_filter is None:
            return None
        ids = [self.label_names.index(x) if isinstance(x, str) else int(x) for x in label_filter]
        return np.array(ids, dtype=np.int64)

    @staticmethod
    def _ranks(scores: np.ndarray) -> np.ndarray:
        """Rang de chaque clause (0 = meilleure)."""
        order = np.argsort(-scores)
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(len(scores))
        return ranks

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        return self.embeddings @ q

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float64)

    # ------------------------------------------------------------------ recherche
    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        label_filter: Optional[Sequence[Union[int, str]]] = None,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Renvoie les `top_k` clauses les plus proches de `query`.
        mode         : "dense", "bm25" ou "hybrid".
        label_filter : liste de categories (numeros ou noms) a garder ; None = toutes.
        """
        if mode not in MODES:
            raise ValueError(f"mode doit etre l'un de {MODES}, pas {mode!r}")

        if mode == "dense":
            scores = self._dense_scores(query)
        elif mode == "bm25":
            scores = self._bm25_scores(query)
        else:  # hybride : fusion par rangs (RRF), insensible aux echelles de scores
            scores = (
                1.0 / (rrf_k + self._ranks(self._dense_scores(query)))
                + 1.0 / (rrf_k + self._ranks(self._bm25_scores(query)))
            )

        allowed = self._label_ids(label_filter)
        if allowed is not None:
            scores = np.where(np.isin(self.labels, allowed), scores, -np.inf)

        top = np.argsort(-scores)[:top_k]
        return [
            {
                "rank": r + 1,
                "index": int(i),
                "score": float(scores[i]),
                "label_id": int(self.labels[i]),
                "label": self.label_names[self.labels[i]],
                "text": self.texts[i],
            }
            for r, i in enumerate(top)
            if np.isfinite(scores[i])
        ]


if __name__ == "__main__":
    import argparse

    from src.data import load_ledgar

    parser = argparse.ArgumentParser(description="Demonstration du moteur de recherche de clauses")
    parser.add_argument("--max-bank", type=int, default=None, help="limiter la banque (ex: 2000 pour un test rapide)")
    args = parser.parse_args()

    dataset, names = load_ledgar(max_train=args.max_bank)
    bank = dataset["train"]
    retriever = ClauseBankRetriever(bank["text"], np.array(bank["label"]), names)

    demo_query = "Either party may terminate this agreement upon 30 days prior written notice."
    for mode in MODES:
        print(f"\n=== mode = {mode} ===")
        for hit in retriever.search(demo_query, top_k=3, mode=mode):
            print(f"[{hit['rank']}] ({hit['label']}) score={hit['score']:.3f} :: {hit['text'][:140]}")