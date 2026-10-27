from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import BOILERPLATE_WORDS, load_spacy

log = logging.getLogger("analysis")

TOKEN_PATTERN = r"(?u)\b[a-z]{3,}\b"
KW_COLS = ["Keyword", "TF-IDF Score", "Raw Frequency", "Doc Frequency"]


def _sources(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(df["source_label"]))


def term_counter(texts) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(t.split())
    return c


def word_frequencies(texts, top_n: int = 50) -> pd.DataFrame:
    c = term_counter(texts)
    total = sum(c.values()) or 1
    rows = [(w, n, round(n / total * 10_000, 2)) for w, n in c.most_common(top_n)]
    return pd.DataFrame(rows, columns=["term", "frequency", "per_10k_tokens"])


def top_ngrams(texts, n: int = 2, top_k: int = 20) -> pd.DataFrame:
    texts = [t for t in texts if t and t.strip()]
    empty = pd.DataFrame(columns=["ngram", "frequency", "documents"])
    if not texts:
        return empty
    try:
        cv = CountVectorizer(ngram_range=(n, n), token_pattern=TOKEN_PATTERN)
        X = cv.fit_transform(texts)
    except ValueError:
        return empty
    freq = np.asarray(X.sum(axis=0)).ravel()
    docs = np.asarray((X > 0).sum(axis=0)).ravel()
    terms = cv.get_feature_names_out()
    idx = np.argsort(-freq, kind="stable")[:top_k]
    return pd.DataFrame({"ngram": terms[idx], "frequency": freq[idx], "documents": docs[idx]})


def distinctive_terms(df: pd.DataFrame, target: str, top_k: int = 15, min_count: int = 3) -> pd.DataFrame:
    a_mask = df["source_label"] == target
    if a_mask.all() or not a_mask.any():
        return pd.DataFrame(columns=["term", "z_score", "count_target", "count_rest", "favored"])
    cv = CountVectorizer(token_pattern=TOKEN_PATTERN)
    X = cv.fit_transform(df["cleaned_text"])
    terms = cv.get_feature_names_out()
    ya = np.asarray(X[a_mask.values].sum(axis=0)).ravel().astype(float)
    yb = np.asarray(X[~a_mask.values].sum(axis=0)).ravel().astype(float)
    prior = ya + yb
    a0 = 500.0
    alpha = a0 * prior / prior.sum()
    na, nb = ya.sum(), yb.sum()
    delta = (np.log((ya + alpha) / (na + a0 - ya - alpha)) - np.log((yb + alpha) / (nb + a0 - yb - alpha)))
    z = delta / np.sqrt(1 / (ya + alpha) + 1 / (yb + alpha))
    keep = (ya + yb) >= min_count
    out = pd.DataFrame({"term": terms, "z_score": z, "count_target": ya.astype(int), "count_rest": yb.astype(int)})
    out = out[keep]
    pos = out.nlargest(top_k, "z_score").assign(favored=target)
    neg = out.nsmallest(top_k, "z_score").assign(favored="others")
    return pd.concat([pos, neg]).reset_index(drop=True)


@dataclass
class KeywordResults:
    vectorizer: TfidfVectorizer
    tfidf: sparse.csr_matrix
    counts: sparse.csr_matrix
    terms: np.ndarray
    overall: pd.DataFrame
    per_source: dict = field(default_factory=dict)
    per_document: pd.DataFrame = field(default_factory=pd.DataFrame)


def _fit_tfidf(docs: list[str], ngram_range, min_df, max_df, max_features) -> TfidfVectorizer:
    n = len(docs)
    if min_df is None:
        min_df = 1 if n < 10 else 2
    if n < 3:
        max_df = 1.0
    attempts = [(min_df, max_df), (1, 1.0)]
    for mdf, xdf in attempts:
        try:
            vec = TfidfVectorizer(ngram_range=ngram_range, min_df=mdf, max_df=xdf, sublinear_tf=True,
                                  token_pattern=TOKEN_PATTERN, max_features=max_features)
            vec.fit(docs)
            return vec
        except ValueError:
            continue
    raise ValueError("No usable vocabulary - are the cleaned documents empty?")


def _keyword_table(scores: np.ndarray, raw: np.ndarray, docfreq: np.ndarray, terms: np.ndarray,
                   top_k: int) -> pd.DataFrame:
    order = np.argsort(-scores, kind="stable")
    order = [i for i in order if raw[i] > 0][:top_k]
    return pd.DataFrame({"Keyword": terms[order], "TF-IDF Score": np.round(scores[order], 4),
                         "Raw Frequency": raw[order].astype(int), "Doc Frequency": docfreq[order].astype(int)},
                        columns=KW_COLS)


def extract_keywords(df: pd.DataFrame, top_k: int = 20, ngram_range=(1, 2), min_df: Optional[int] = None,
                     max_df: float = 0.95, max_features: int = 50_000, per_doc_k: int = 10) -> KeywordResults:
    docs = df["cleaned_text"].tolist()
    vec = _fit_tfidf(docs, ngram_range, min_df, max_df, max_features)
    X = vec.transform(docs).tocsr()
    counts = CountVectorizer(ngram_range=ngram_range, vocabulary=vec.vocabulary_,
                             token_pattern=TOKEN_PATTERN).transform(docs).tocsr()
    terms = vec.get_feature_names_out()

    def summarise(mask: np.ndarray) -> pd.DataFrame:
        Xm, Cm = X[mask], counts[mask]
        return _keyword_table(np.asarray(Xm.mean(axis=0)).ravel(), np.asarray(Cm.sum(axis=0)).ravel(),
                              np.asarray((Cm > 0).sum(axis=0)).ravel(), terms, top_k)

    overall = summarise(np.ones(len(df), dtype=bool))
    per_source = {s: summarise((df["source_label"] == s).values) for s in _sources(df)}

    rows = []
    for i in range(X.shape[0]):
        start, end = X.indptr[i], X.indptr[i + 1]
        if start == end:
            continue
        data, cols = X.data[start:end], X.indices[start:end]
        for rank, j in enumerate(np.argsort(-data, kind="stable")[:per_doc_k], 1):
            rows.append((int(df["id"].iloc[i]), df["source_label"].iloc[i], df["title"].iloc[i], rank,
                         terms[cols[j]], round(float(data[j]), 4), int(counts[i, cols[j]])))
    per_doc = pd.DataFrame(rows, columns=["id", "source_label", "title", "rank", "Keyword",
                                          "TF-IDF Score", "Raw Frequency"])
    return KeywordResults(vec, X, counts, terms, overall, per_source, per_doc)


def format_keyword_table(table: pd.DataFrame) -> str:
    return table.to_string(index=False)


@dataclass
class TopicResults:
    method: str
    n_topics: int
    topics: pd.DataFrame
    doc_topics: pd.Series
    term_weights: dict
    samples: dict
    source_distribution: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def _assemble_topics(method, df, assignments, scores, term_weights, n_samples, metrics) -> TopicResults:
    k = scores.shape[1]
    doc_topics = pd.Series(assignments, index=df.index, name="topic")
    rows, samples = [], {}
    for t in range(k):
        mask = np.where(assignments == t)[0]
        best = mask[np.argsort(-scores[mask, t])][:n_samples] if len(mask) else []
        samples[t] = [dict(title=df["title"].iloc[i], url=df["url"].iloc[i],
                           source_label=df["source_label"].iloc[i], score=round(float(scores[i, t]), 3))
                      for i in best]
        words = [w for w, _ in term_weights[t]]
        rows.append(dict(topic_id=t, label=" / ".join(words[:3]), top_terms=", ".join(words),
                         n_docs=int(len(mask)), share=round(len(mask) / len(df), 3),
                         sample_docs=" | ".join(f"{str(s['title'])[:60]} ({s['source_label']})" for s in samples[t])))
    dist = pd.crosstab(doc_topics, df["source_label"]).reindex(range(k), fill_value=0)
    return TopicResults(method, k, pd.DataFrame(rows), doc_topics, term_weights, samples, dist, metrics)


def run_lda(df: pd.DataFrame, n_topics: int = 5, n_top_words: int = 10, n_samples: int = 3,
            max_features: int = 5000, random_state: int = 42) -> TopicResults:
    n = len(df)
    k = max(2, min(n_topics, n))
    cv = CountVectorizer(max_df=0.9 if n >= 5 else 1.0, min_df=2 if n >= 10 else 1,
                         max_features=max_features, token_pattern=TOKEN_PATTERN)
    try:
        X = cv.fit_transform(df["cleaned_text"])
    except ValueError:
        cv = CountVectorizer(token_pattern=TOKEN_PATTERN, max_features=max_features)
        X = cv.fit_transform(df["cleaned_text"])
    lda = LatentDirichletAllocation(n_components=k, learning_method="batch", max_iter=30,
                                    random_state=random_state)
    theta = lda.fit_transform(X)
    terms = cv.get_feature_names_out()
    comp = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    weights = {t: [(terms[j], float(comp[t, j])) for j in np.argsort(-comp[t])[:n_top_words]] for t in range(k)}
    metrics = {"perplexity": round(float(lda.perplexity(X)), 2), "vocabulary_size": int(X.shape[1])}
    return _assemble_topics("LDA", df, theta.argmax(axis=1), theta, weights, n_samples, metrics)


def run_kmeans(df: pd.DataFrame, n_topics: int = 5, n_top_words: int = 10, n_samples: int = 3,
               max_features: int = 5000, random_state: int = 42) -> TopicResults:
    n = len(df)
    k = max(2, min(n_topics, n))
    vec = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9 if n >= 5 else 1.0, min_df=2 if n >= 10 else 1,
                          sublinear_tf=True, max_features=max_features, token_pattern=TOKEN_PATTERN)
    try:
        X = vec.fit_transform(df["cleaned_text"])
    except ValueError:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=TOKEN_PATTERN)
        X = vec.fit_transform(df["cleaned_text"])
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
    terms = vec.get_feature_names_out()
    weights = {t: [(terms[j], float(km.cluster_centers_[t, j])) for j in np.argsort(-km.cluster_centers_[t])[:n_top_words]]
               for t in range(k)}
    metrics = {}
    if 2 <= len(set(km.labels_)) < n:
        metrics["silhouette_cosine"] = round(float(silhouette_score(X, km.labels_, metric="cosine")), 3)
    metrics["inertia"] = round(float(km.inertia_), 3)
    return _assemble_topics("K-Means", df, km.labels_, -km.transform(X), weights, n_samples, metrics)


def run_topic_model(df: pd.DataFrame, method: str = "lda", **kw) -> TopicResults:
    if len(df) < 2:
        raise ValueError("Topic modelling needs at least 2 documents")
    return run_lda(df, **kw) if method.lower() == "lda" else run_kmeans(df, **kw)


@dataclass
class ModelResult:
    name: str
    accuracy: float
    macro_f1: float
    report_text: str
    confusion: np.ndarray
    labels: list
    top_features: pd.DataFrame
    cv_macro_f1: Optional[tuple] = None
    pipeline: Optional[Pipeline] = None


@dataclass
class ClassificationResults:
    models: dict = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_train: int = 0
    n_test: int = 0
    warnings: list = field(default_factory=list)
    error: Optional[str] = None


def _top_features(pipe: Pipeline, n: int = 12) -> pd.DataFrame:
    terms = pipe.named_steps["tfidf"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    classes = list(clf.classes_)
    rows = []
    if clf.coef_.shape[0] == 1:
        w = clf.coef_[0]
        for j in np.argsort(-w)[:n]:
            rows.append((classes[1], terms[j], float(w[j])))
        for j in np.argsort(w)[:n]:
            rows.append((classes[0], terms[j], float(-w[j])))
    else:
        for ci, c in enumerate(classes):
            w = clf.coef_[ci]
            for j in np.argsort(-w)[:n]:
                rows.append((c, terms[j], float(w[j])))
    return pd.DataFrame(rows, columns=["source_label", "term", "weight"])


def train_source_classifier(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> ClassificationResults:
    res = ClassificationResults()
    y = df["source_label"].values.ravel()
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        res.error = "Classification needs at least two sources."
        return res
    if counts.min() < 2:
        res.error = (f"Source '{counts.idxmin()}' has only {counts.min()} document(s); a stratified split "
                     "needs at least 2 per source. Scrape more pages or raise max_pages.")
        return res
    n, n_classes = len(df), len(counts)
    eff_test = max(test_size, n_classes / n)
    if eff_test > test_size:
        res.warnings.append(f"Test fraction raised to {eff_test:.2f} so every class appears in the test set.")
    try:
        Xtr, Xte, ytr, yte, itr, ite = train_test_split(
            df["cleaned_text"].values, y, np.arange(n), test_size=eff_test, stratify=y, random_state=random_state)
    except ValueError as exc:
        res.error = f"Could not create a stratified split: {exc}"
        return res
    res.n_train, res.n_test = len(Xtr), len(Xte)
    labels = sorted(counts.index)
    min_df = 2 if len(Xtr) >= 40 else 1

    def make(clf):
        return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=min_df,
                                                    token_pattern=TOKEN_PATTERN)),
                         ("clf", clf)])

    candidates = {
        "LogisticRegression": make(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "LinearSVC": make(LinearSVC(class_weight="balanced", random_state=random_state)),
    }
    preds = pd.DataFrame({"url": df["url"].values[ite], "title": df["title"].values[ite], "true": yte})
    rows = []
    for name, pipe in candidates.items():
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        preds[f"pred_{name}"] = pred
        acc = accuracy_score(yte, pred)
        mf1 = f1_score(yte, pred, average="macro", zero_division=0)
        cv = None
        if counts.min() >= 5:
            sc = cross_val_score(clone(pipe), df["cleaned_text"].values, y, scoring="f1_macro",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=random_state))
            cv = (float(sc.mean()), float(sc.std()))
        res.models[name] = ModelResult(
            name, float(acc), float(mf1), classification_report(yte, pred, labels=labels, zero_division=0),
            confusion_matrix(yte, pred, labels=labels), labels, _top_features(pipe), cv, pipe)
        rows.append(dict(model=name, accuracy=round(acc, 4), macro_f1=round(mf1, 4),
                         cv_macro_f1=("%.3f ± %.3f" % cv) if cv else "n/a (needs >=5 docs/source)"))
    res.summary = pd.DataFrame(rows)
    res.predictions = preds
    if len(yte) < 30:
        res.warnings.append(f"Only {len(yte)} test documents - metrics are noisy; prefer the CV score when available.")
    if counts.max() / counts.min() > 3:
        res.warnings.append("Sources are imbalanced (>3:1); macro-F1 is more trustworthy than accuracy.")
    if all(m.accuracy == 1.0 for m in res.models.values()):
        res.warnings.append("Perfect accuracy usually means site-specific vocabulary (names, navigation text) "
                            "makes the sources trivially separable - inspect the top features before drawing conclusions.")
    return res


ENTITY_TYPES = {"ORG": "Organization", "GPE": "Location", "LOC": "Location", "PERSON": "Person"}


def _norm_entity(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip(" \t\n.,;:'\"()[]")
    t = re.sub(r"^(the|The)\s+", "", t)
    return re.sub(r"['’]s$", "", t)


def extract_entities(df: pd.DataFrame, max_chars: int = 100_000, model: str = "en_core_web_sm") -> pd.DataFrame:
    cols = ["source_label", "entity", "type", "mentions", "documents"]
    nlp = load_spacy(model, ("parser", "lemmatizer"))
    if nlp is None:
        return pd.DataFrame(columns=cols)
    mentions: Counter = Counter()
    docs: Counter = Counter()
    texts = [t[:max_chars] for t in df["raw_text"].fillna("")]
    for src, doc in zip(df["source_label"], nlp.pipe(texts, batch_size=4)):
        seen = set()
        for ent in doc.ents:
            etype = ENTITY_TYPES.get(ent.label_)
            name = _norm_entity(ent.text)
            if not etype or len(name) < 3 or name.isdigit():
                continue
            key = (src, name, etype)
            mentions[key] += 1
            seen.add(key)
        docs.update(seen)
    rows = [(*k, mentions[k], docs[k]) for k in mentions]
    return (pd.DataFrame(rows, columns=cols).sort_values(["source_label", "mentions"], ascending=[True, False])
            .reset_index(drop=True))


def top_entities(entities: pd.DataFrame, per_type: int = 10) -> pd.DataFrame:
    if entities.empty:
        return entities
    return (entities.sort_values("mentions", ascending=False)
            .groupby(["source_label", "type"], sort=False).head(per_type)
            .sort_values(["source_label", "type", "mentions"], ascending=[True, True, False]).reset_index(drop=True))


def _padded(texts) -> list[str]:
    return [f" {t} " for t in texts]


def evaluate_keywords(keywords: pd.DataFrame, docs: pd.DataFrame, top_k: int = 20, title: str = "",
                      corpus: Optional[pd.DataFrame] = None) -> dict:
    kw = keywords.head(top_k)["Keyword"].tolist()
    n_docs = len(docs)
    pad = _padded(docs["cleaned_text"])
    corpus_pad = pad if corpus is None else _padded(corpus["cleaned_text"])
    src = docs["source_label"].values
    rows = []
    for k in kw:
        needle = f" {k} "
        per_doc = np.array([p.count(needle) for p in pad], dtype=float)
        corpus_cov = float(np.mean([needle in p for p in corpus_pad])) if corpus_pad else 0.0
        total = per_doc.sum()
        coverage = float((per_doc > 0).mean()) if n_docs else 0.0
        top_share = float(per_doc.max() / total) if total else 0.0
        by_src = pd.Series(per_doc).groupby(src).sum()
        src_share = float(by_src.max() / total) if total else 0.0
        flags = []
        if corpus_cov > 0.8 and len(corpus_pad) >= 5:
            flags.append("near-ubiquitous (>80% of all documents)")
        if (per_doc > 0).sum() == 1 and n_docs > 3:
            flags.append("appears in a single document")
        elif top_share > 0.5 and n_docs > 3:
            flags.append("dominated by one document")
        if any(w in BOILERPLATE_WORDS for w in k.split()):
            flags.append("boilerplate-like")
        others = [o for o in kw if o != k and f" {k} " in f" {o} "]
        if others:
            flags.append(f"redundant with '{others[0]}'")
        rows.append(dict(keyword=k, n_words=len(k.split()), coverage=round(coverage, 3),
                         corpus_coverage=round(corpus_cov, 3), top_doc_share=round(top_share, 3), source_share=round(src_share, 3),
                         flags="; ".join(flags) or "ok"))
    table = pd.DataFrame(rows)
    if table.empty:
        return {"table": table, "summary": {}, "report": f"### Keyword quality report {title}\nNo keywords available."}
    flagged = float((table["flags"] != "ok").mean())
    summary = {
        "keywords_evaluated": len(table),
        "mean_doc_coverage": round(float(table["coverage"].mean()), 3),
        "median_doc_coverage": round(float(table["coverage"].median()), 3),
        "bigram_share": round(float((table["n_words"] == 2).mean()), 3),
        "redundancy_rate": round(float(table["flags"].str.contains("redundant").mean()), 3),
        "flagged_share": round(flagged, 3),
        "source_specific_share": round(float((table["source_share"] > 0.8).mean()), 3) if docs["source_label"].nunique() > 1 else None,
    }
    rating = "Good" if flagged <= 0.2 else "Fair" if flagged <= 0.45 else "Needs attention"
    summary["overall_rating"] = rating
    notes = []
    if summary["mean_doc_coverage"] < 0.1 and n_docs > 5:
        notes.append("Keywords are very sparse across documents - consider a larger corpus or a lower min_df.")
    if summary["mean_doc_coverage"] > 0.5:
        notes.append("Keywords recur in most documents, i.e. they capture corpus-wide themes rather than "
                     "page-specific details (expected for source-level keywords).")
    if summary["redundancy_rate"] > 0.3:
        notes.append("Many unigrams duplicate listed bigrams; consider de-duplicating for presentation.")
    if flagged > 0.45:
        notes.append("Many keywords are boilerplate, near-ubiquitous or single-page artefacts; extend the custom stop-word list.")
    if not notes:
        notes.append("No systematic problems detected.")
    bad = table[table["flags"] != "ok"]
    lines = [f"### Keyword quality report {('- ' + title) if title else ''}".rstrip(),
             f"**Overall rating: {rating}** ({int(round(flagged * 100))}% of the top-{len(table)} keywords flagged)", "",
             f"- Mean / median document coverage: {summary['mean_doc_coverage']:.0%} / {summary['median_doc_coverage']:.0%}",
             f"- Bigram share: {summary['bigram_share']:.0%}   |   Redundancy: {summary['redundancy_rate']:.0%}"]
    if summary["source_specific_share"] is not None:
        lines.append(f"- Source-specific keywords (>80% of mentions in one source): {summary['source_specific_share']:.0%}")
    lines += ["", "**Observations**"] + [f"- {n}" for n in notes]
    if len(bad):
        lines += ["", "**Flagged keywords**"] + [f"- `{r.keyword}` - {r.flags}" for r in bad.itertuples()]
    return {"table": table, "summary": summary, "report": "\n".join(lines)}


@dataclass
class TemporalResults:
    trend: Optional[pd.DataFrame] = None
    freq: str = ""
    n_docs: int = 0
    message: str = ""


def temporal_keyword_trends(df: pd.DataFrame, keywords: list[str], freq: Optional[str] = None) -> TemporalResults:
    d = df.copy()
    if "date_source" in d.columns:
        d = d[d["date_source"] == "published"]
    d["_dt"] = pd.to_datetime(d["publish_date"], errors="coerce", utc=True).dt.tz_localize(None)
    d = d.dropna(subset=["_dt"])
    if len(d) < 5:
        return TemporalResults(message=f"Only {len(d)} page(s) with a detected publish date (need >= 5) - skipping temporal analysis.")
    span = (d["_dt"].max() - d["_dt"].min()).days
    freq = freq or ("D" if span < 14 else "W" if span < 120 else "M")
    d["_period"] = d["_dt"].dt.to_period(freq)
    if d["_period"].nunique() < 2:
        return TemporalResults(n_docs=len(d), message="All dated pages fall in one period - no trend to plot.")
    kws = [k for k in dict.fromkeys(keywords)]
    if not kws:
        return TemporalResults(message="No keywords supplied.")
    cv = CountVectorizer(ngram_range=(1, 2), vocabulary=kws, token_pattern=TOKEN_PATTERN)
    C = cv.transform(d["cleaned_text"])
    counts = pd.DataFrame(C.toarray(), columns=kws, index=d["_period"].values).groupby(level=0).sum()
    tokens = d.groupby("_period")["token_count"].sum()
    trend = counts.div(tokens.reindex(counts.index), axis=0) * 1000
    trend.index = trend.index.to_timestamp()
    return TemporalResults(trend=trend.sort_index(), freq=freq, n_docs=len(d))
