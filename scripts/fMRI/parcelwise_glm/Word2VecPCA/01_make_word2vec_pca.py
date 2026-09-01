#!/usr/bin/env python3
"""Create an eight-PC Word2Vec control feature table for the 13,850 EFA words.

The lookup order is deliberately explicit: Word, lowercase Word, Lemma, then
lowercase Lemma.  This preserves the controlled 13,850-row vocabulary while
recording every lemma substitution in a separate audit table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_SCOPE = Path("/work/xy6/resources/SCOPE_wordonly_by052026.csv")
DEFAULT_VOCAB = Path(
    "/work/desai-lab/xuanyang/Project/Semantic/analysis/FactorAnalysis/"
    "github/data/SCOPE_lemma_var106.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
EXPECTED_WORDS = 13_850
EXPECTED_DIMENSIONS = 300
N_COMPONENTS = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--normalization",
        choices=("l2", "none"),
        default="l2",
        help="L2-normalize word vectors before PCA (default: l2).",
    )
    return parser.parse_args()


def valid_vector(value):
    return isinstance(value, str) and value not in {"", "NA", "NaN", "nan"}


def build_lookups(scope):
    exact = {}
    lower = {}
    for word, vector in scope[["Word", "Word2Vec"]].itertuples(index=False):
        if not valid_vector(vector):
            continue
        word = str(word)
        exact.setdefault(word, vector)
        key = word.lower()
        # Prefer the explicitly lowercase SCOPE entry for lowercase fallback.
        if key not in lower or word == key:
            lower[key] = vector
    return exact, lower


def lookup_vector(word, lemma, exact, lower):
    candidates = (
        (word, "Word"),
        (word.lower(), "Word_lower"),
        (lemma, "Lemma"),
        (lemma.lower(), "Lemma_lower"),
    )
    for value, source in candidates:
        lookup = exact if source in {"Word", "Lemma"} else lower
        if value in lookup:
            return lookup[value], source, value
    raise KeyError("No Word2Vec vector for Word={!r}, Lemma={!r}".format(word, lemma))


def parse_vector(value, word):
    vector = np.fromstring(value, sep="|", dtype=np.float64)
    if vector.size != EXPECTED_DIMENSIONS:
        raise ValueError(
            "{} has {} Word2Vec dimensions; expected {}".format(
                word, vector.size, EXPECTED_DIMENSIONS
            )
        )
    if not np.isfinite(vector).all():
        raise ValueError("{} has a non-finite Word2Vec value".format(word))
    return vector


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = pd.read_csv(
        args.vocabulary,
        usecols=["idx", "Word", "Lemma"],
        keep_default_na=False,
    )
    if len(vocabulary) != EXPECTED_WORDS:
        raise ValueError(
            "Controlled vocabulary contains {} rows; expected {}".format(
                len(vocabulary), EXPECTED_WORDS
            )
        )
    if vocabulary["Word"].duplicated().any():
        duplicates = vocabulary.loc[vocabulary["Word"].duplicated(), "Word"].tolist()
        raise ValueError("Duplicate controlled-vocabulary words: {}".format(duplicates[:10]))

    scope = pd.read_csv(
        args.scope,
        usecols=["Word", "Word2Vec"],
        keep_default_na=False,
        encoding="utf-8",
    )
    exact, lower = build_lookups(scope)

    vectors = []
    audit_rows = []
    for row in vocabulary.itertuples(index=False):
        word = str(row.Word)
        lemma = str(row.Lemma)
        encoded, source, lookup_value = lookup_vector(word, lemma, exact, lower)
        vectors.append(parse_vector(encoded, word))
        audit_rows.append(
            {
                "idx": row.idx,
                "Word": word,
                "Lemma": lemma,
                "embedding_source": source,
                "embedding_lookup": lookup_value,
            }
        )

    matrix_raw = np.vstack(vectors)
    norms = np.linalg.norm(matrix_raw, axis=1)
    if np.any(norms == 0):
        words = vocabulary.loc[norms == 0, "Word"].tolist()
        raise ValueError("Zero-length Word2Vec vectors: {}".format(words[:10]))

    if args.normalization == "l2":
        matrix_pca = matrix_raw / norms[:, None]
    else:
        matrix_pca = matrix_raw.copy()

    pca = PCA(n_components=N_COMPONENTS, svd_solver="full")
    scores_raw = pca.fit_transform(matrix_pca)

    # Standardize amplitudes so PC beta scaling is comparable to factor scores.
    score_sd = scores_raw.std(axis=0, ddof=0)
    if np.any(score_sd == 0):
        raise ValueError("At least one retained PC has zero score variance")
    scores_z = (scores_raw - scores_raw.mean(axis=0)) / score_sd
    pc_columns = ["PC{}_n{}".format(i, N_COMPONENTS) for i in range(1, 9)]

    features = vocabulary.copy()
    features[pc_columns] = scores_z
    features.to_csv(args.output_dir / "data_W2V_PCA_n8.csv", index=False)

    raw_scores = vocabulary.copy()
    raw_scores[pc_columns] = scores_raw
    raw_scores.to_csv(args.output_dir / "scores_W2V_PCA_n8_raw.csv", index=False)

    audit = pd.DataFrame(audit_rows)
    audit["vector_norm_raw"] = norms
    audit.to_csv(args.output_dir / "word2vec_lookup_audit.csv", index=False)

    component_names = ["W2V_{:03d}".format(i) for i in range(1, 301)]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=component_names,
        columns=pc_columns,
    )
    loadings.index.name = "embedding_dimension"
    loadings.to_csv(args.output_dir / "W2V_PCA_n8_loadings.csv")

    explained = pd.DataFrame(
        {
            "component": pc_columns,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    explained.to_csv(args.output_dir / "W2V_PCA_n8_explained_variance.csv", index=False)

    np.savez_compressed(
        args.output_dir / "W2V_PCA_n8_model.npz",
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        score_mean=scores_raw.mean(axis=0),
        score_sd=score_sd,
        normalization=np.array(args.normalization),
    )

    source_counts = audit["embedding_source"].value_counts().sort_index().to_dict()
    metadata = {
        "scope_file": str(args.scope),
        "vocabulary_file": str(args.vocabulary),
        "n_words": int(len(vocabulary)),
        "embedding_dimensions": EXPECTED_DIMENSIONS,
        "n_components": N_COMPONENTS,
        "normalization": args.normalization,
        "pc_scores_z_standardized": True,
        "lookup_order": ["Word", "Word_lower", "Lemma", "Lemma_lower"],
        "embedding_source_counts": {k: int(v) for k, v in source_counts.items()},
    }
    with open(args.output_dir / "W2V_PCA_n8_metadata.json", "w") as stream:
        json.dump(metadata, stream, indent=2)

    print("Wrote {} controlled Word2Vec PC rows to {}".format(len(features), args.output_dir))
    print("Embedding source counts: {}".format(source_counts))
    print(
        "PC1-PC8 cumulative explained variance: {:.4f}".format(
            explained["cumulative_explained_variance_ratio"].iloc[-1]
        )
    )


if __name__ == "__main__":
    main()
