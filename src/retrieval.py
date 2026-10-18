"""
Personne 3 - Moteur de recherche dans la banque de clauses (version 2)
======================================================================
Trois modes de recherche :
  - "dense"  : par le SENS (sentence-transformers, similarite cosinus)
  - "bm25"   : par les MOTS exacts (BM25), garde les chiffres ("30" != "60")
  - "hybrid" : les deux classements fusionnes (RRF, rangs a partir de 1)
Un filtre optionnel par categorie fonctionne avec les trois modes : il est applique
AVANT le classement, donc la fusion se fait a l'interieur des categories gardees.

Nouveautes par rapport a la version 1 :
  - search_batch() : plusieurs requetes d'un coup (indispensable pour evaluer) ;
  - BM25 en matrices creuses (meme formule que rank_bm25.BM25Okapi, ~100x plus rapide) ;
  - tri stable (resultats identiques a chaque lancement) ;
  - le cache d'empreintes depend du CONTENU de la banque et du modele (plus de risque
    de recharger de mauvaises empreintes) ;
  - rangs RRF commencant a 1 (definition standard).

Demonstration (depuis la racine du projet) :
    python -m src.retrieval --max-bank 2000
"""

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

# Dossier ou l'on range les empreintes calculees (ignore par Git, voir .git/info/exclude)
RETRIEVAL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "retrieval"

MODES = ("dense", "bm25", "hybrid")


def tokenize(text: str) -> List[str]:
    """Decoupe un texte en mots en minuscules. Les chiffres sont GARDES (important pour BM25)."""
    return re.findall(r"[a-z0-9]+", text.lower())


class SparseBM25:
    """
    BM25 Okapi avec matrices creuses. Meme formule que `rank_bm25.BM25Okapi`
    (k1=1.5, b=0.75, epsilon=0.25 ; un mot repete dans la requete compte plusieurs fois),
    mais calcule les scores de plusieurs requetes d'un seul produit matriciel.
    """

    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.vocab: Dict[str, int] = {}
        indptr, indices, counts, doc_len = [0], [], [], []
        for doc in tokenized_docs:
            doc_len.append(len(doc))
            for term, n in Counter(doc).items():
                indices.append(self.vocab.setdefault(term, len(self.vocab)))
                counts.append(n)
            indptr.append(len(indices))

        self.n_docs, self.n_terms = len(doc_len), len(self.vocab)
        indices = np.asarray(indices, dtype=np.int64)
        counts = np.asarray(counts, dtype=np.float64)
        doc_len = np.asarray(doc_len, dtype=np.float64)
        indptr = np.asarray(indptr, dtype=np.int64)

        # idf comme rank_bm25 : les idf negatifs sont remplaces par epsilon * idf moyen
        doc_freq = np.bincount(indices, minlength=self.n_terms).astype(np.float64)
        idf = np.log(self.n_docs - doc_freq + 0.5) - np.log(doc_freq + 0.5)
        idf = np.where(idf < 0, epsilon * idf.mean(), idf)

        avgdl = doc_len.sum() / self.n_docs
        rows = np.repeat(np.arange(self.n_docs), np.diff(indptr))
        length_norm = k1 * (1.0 - b + b * doc_len[rows] / avgdl)
        weights = idf[indices] * counts * (k1 + 1.0) / (counts + length_norm)
        matrix = sparse.csr_matrix((weights, indices, indptr), shape=(self.n_docs, self.n_terms))
        self._weights_t = matrix.T.tocsr()  # (mots, documents)

    def get_scores(self, tokenized_queries: List[List[str]]) -> np.ndarray:
        """Scores BM25 de chaque requete contre chaque document : tableau (n_requetes, n_docs)."""
        rows, cols = [], []
        for i, tokens in enumerate(tokenized_queries):
            for token in tokens:
                j = self.vocab.get(token)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        queries = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(tokenized_queries), self.n_terms)
        )  # les mots repetes sont additionnes
        return (queries @ self._weights_t).toarray()


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
        rrf_k: int = 60,
    ):
        # L'import est ici pour que "import src.retrieval" reste rapide
        from sentence_transformers import SentenceTransformer

        self.texts = list(corpus_texts)
        self.labels = np.asarray(corpus_labels, dtype=np.int64)
        self.label_names = list(label_names)
        self.model_name = model_name
        self.rrf_k = rrf_k
        self.fingerprint = self._compute_fingerprint()

        # 1) Moteur "sens" : une empreinte numerique par clause
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_compute_embeddings(cache_dir, batch_size)

        # 2) Moteur "mots" : index BM25
        self.bm25 = SparseBM25([tokenize(t) for t in self.texts])

    # ------------------------------------------------------------------ empreintes
    def _compute_fingerprint(self) -> str:
        """Empreinte du contenu de la banque + modele : change si une seule clause change."""
        h = hashlib.sha1(self.model_name.encode("utf-8"))
        for text in self.texts:
            h.update(text.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:10]

    def _load_or_compute_embeddings(self, cache_dir: Optional[Path], batch_size: int) -> np.ndarray:
        cache_file = None
        if cache_dir is not None:
            slug = self.model_name.replace("/", "_")
            cache_file = Path(cache_dir) / f"bank_embeddings_{slug}_{len(self.texts)}_{self.fingerprint}.npy"
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
        """Rang de chaque clause pour chaque requete (1 = meilleure), tri stable."""
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.empty(order.shape, dtype=np.int64)
        positions = np.broadcast_to(np.arange(1, scores.shape[1] + 1), order.shape)
        np.put_along_axis(ranks, order, positions, axis=1)
        return ranks

    def _dense_scores(self, queries: Sequence[str]) -> np.ndarray:
        q = self.model.encode(list(queries), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return q.astype(np.float32) @ self.embeddings.T

    def _bm25_scores(self, queries: Sequence[str]) -> np.ndarray:
        return self.bm25.get_scores([tokenize(q) for q in queries])

    def _chunk_scores(self, queries: Sequence[str], mode: str, allowed: Optional[np.ndarray]) -> np.ndarray:
        """Scores (n_requetes, n_clauses) ; les clauses exclues par le filtre valent -inf."""
        excluded = None if allowed is None else ~np.isin(self.labels, allowed)

        def apply_filter(scores: np.ndarray) -> np.ndarray:
            if excluded is not None:
                scores[:, excluded] = -np.inf
            return scores

        if mode == "dense":
            return apply_filter(self._dense_scores(queries))
        if mode == "bm25":
            return apply_filter(self._bm25_scores(queries))
        # hybride : fusion par rangs (RRF), insensible aux echelles de scores
        dense = apply_filter(self._dense_scores(queries))
        lexical = apply_filter(self._bm25_scores(queries))
        fused = 1.0 / (self.rrf_k + self._ranks(dense)) + 1.0 / (self.rrf_k + self._ranks(lexical))
        return apply_filter(fused)

    def _search(
        self,
        queries: Sequence[str],
        top_k: int,
        mode: str,
        label_filter: Optional[Sequence[Union[int, str]]],
        chunk_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if mode not in MODES:
            raise ValueError(f"mode doit etre l'un de {MODES}, pas {mode!r}")
        allowed = self._label_ids(label_filter)
        queries = list(queries)
        k = min(top_k, len(self.texts))
        indices = np.full((len(queries), top_k), -1, dtype=np.int64)
        scores_out = np.full((len(queries), top_k), -np.inf)
        for start in range(0, len(queries), chunk_size):
            scores = self._chunk_scores(queries[start:start + chunk_size], mode, allowed)
            order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            top_scores = np.take_along_axis(scores, order, axis=1)
            order = np.where(np.isfinite(top_scores), order, -1)  # -1 = pas assez de clauses
            end = start + order.shape[0]
            indices[start:end, :k] = order
            scores_out[start:end, :k] = top_scores
        return indices, scores_out

    # ------------------------------------------------------------------ recherche
    def search_batch(
        self,
        queries: Sequence[str],
        top_k: int = 5,
        mode: str = "hybrid",
        label_filter: Optional[Sequence[Union[int, str]]] = None,
        chunk_size: int = 128,
    ) -> np.ndarray:
        """
        Plusieurs requetes d'un coup. Renvoie un tableau (n_requetes, top_k) des POSITIONS
        des clauses trouvees dans la banque (-1 quand il n'y a pas assez de resultats).
        """
        return self._search(queries, top_k, mode, label_filter, chunk_size)[0]

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        label_filter: Optional[Sequence[Union[int, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Renvoie les `top_k` clauses les plus proches de `query`.
        mode         : "dense", "bm25" ou "hybrid".
        label_filter : liste de categories (numeros ou noms) a garder ; None = toutes.
        """
        indices, scores = self._search([query], top_k, mode, label_filter, chunk_size=1)
        return [
            {
                "rank": r + 1,
                "index": int(i),
                "score": float(s),
                "label_id": int(self.labels[i]),
                "label": self.label_names[self.labels[i]],
                "text": self.texts[i],
            }
            for r, (i, s) in enumerate(zip(indices[0], scores[0]))
            if i >= 0
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