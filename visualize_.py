"""
visualize.py - Charts for the Keyword & Theme Intelligence System.

Every function returns a `matplotlib.figure.Figure` (so Streamlit can `st.pyplot` it and the CLI
can save it). The "Agg" backend is selected so nothing needs a display.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt              
import numpy as np              
import pandas as pd              
import seaborn as sns              

from analysis import term_counter              

sns.set_theme(style="whitegrid", context="notebook")


                                                                               
         
                                                                               
def source_palette(labels) -> dict:
    labels = list(dict.fromkeys(labels))
    return dict(zip(labels, sns.color_palette("Set2", max(len(labels), 3))))


def _placeholder(message: str, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")
    ax.text(0.5, 0.5, textwrap.fill(message, 70), ha="center", va="center", fontsize=11, color="#555")
    if title:
        ax.set_title(title)
    return fig


def _wrap(label: str, width: int = 28) -> str:
    return textwrap.shorten(str(label), width=width, placeholder="…")


def save_figure(fig: plt.Figure, path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _barh_grid(data: dict, label_col: str, value_col: str, title: str, xlabel: str,
               top_n: int = 20, palette: Optional[dict] = None, annotate: bool = True) -> plt.Figure:
    """One horizontal bar chart per key of `data` (label -> DataFrame), side by side."""
    data = {k: v for k, v in data.items() if v is not None and len(v)}
    if not data:
        return _placeholder("No data available for this chart.", title)
    palette = palette or source_palette(data.keys())
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(max(6.5, 6.2 * n), max(4, 0.32 * top_n + 1.6)), squeeze=False)
    for ax, (name, frame) in zip(axes[0], data.items()):
        f = frame.head(top_n).iloc[::-1]
        ax.barh([_wrap(x) for x in f[label_col]], f[value_col], color=palette.get(name, "#4c72b0"))
        ax.set_title(name if n > 1 else "", fontsize=11)
        ax.set_xlabel(xlabel)
        if annotate:
            for y, v in enumerate(f[value_col]):
                ax.text(v, y, f" {v:g}", va="center", fontsize=8)
        ax.margins(x=0.12)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


                                                                               
     
                                                                               
def plot_top_words(freq_by_source: dict, title: str = "Top 20 words", top_n: int = 20) -> plt.Figure:
    """`freq_by_source`: label -> DataFrame(term, frequency). Pass {'All sources': df} for one panel."""
    return _barh_grid(freq_by_source, "term", "frequency", title, "Frequency", top_n)


def plot_ngrams(ngrams_by_source: dict, title: str, top_n: int = 15) -> plt.Figure:
    return _barh_grid(ngrams_by_source, "ngram", "frequency", title, "Frequency", top_n)


def plot_wordcloud(counter, title: str = "") -> plt.Figure:
    return plot_wordclouds({title or "Word cloud": counter}, cols=1)


def plot_wordclouds(counters: dict, cols: int = 3) -> plt.Figure:
    """Grid of word clouds. `counters`: name -> Counter/dict of term frequencies (aggregate first)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return _placeholder("The 'wordcloud' package is not installed (pip install wordcloud).", "Word clouds")
    counters = {k: v for k, v in counters.items() if v}
    if not counters:
        return _placeholder("No text available for word clouds.")
    n = len(counters)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, freqs) in zip(axes.ravel(), counters.items()):
        wc = WordCloud(width=900, height=520, background_color="white", max_words=120,
                       colormap="viridis", collocations=False, random_state=42).generate_from_frequencies(dict(freqs))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(name, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_comparative_terms(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Grouped bars: relative frequency (per 10k tokens) of the most common terms in each source."""
    labels = list(dict.fromkeys(df["source_label"]))
    if len(labels) < 2:
        return _placeholder("Comparison needs at least two sources.", "Term frequency comparison")
    counters = {s: term_counter(df.loc[df["source_label"] == s, "cleaned_text"]) for s in labels}
    rel = pd.DataFrame({s: pd.Series(c) / max(sum(c.values()), 1) * 10_000 for s, c in counters.items()}).fillna(0)
    terms = rel.mean(axis=1).nlargest(top_n).index[::-1]
    pal = source_palette(labels)
    h = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(8, 0.5 * top_n + 1.8))
    y = np.arange(len(terms))
    for i, s in enumerate(labels):
        ax.barh(y + i * h, rel.loc[terms, s], height=h, label=s, color=pal[s])
    ax.set_yticks(y + h * (len(labels) - 1) / 2)
    ax.set_yticklabels(terms)
    ax.set_xlabel("Occurrences per 10,000 tokens")
    ax.set_title("Term frequency across sources", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_distinctive_terms(dist: pd.DataFrame, target: str, top_n: int = 12) -> plt.Figure:
    """Diverging bars of log-odds z-scores: right = favours `target`, left = favours the other sources."""
    if dist is None or dist.empty:
        return _placeholder("No distinctive-term data (needs at least two sources).", "Distinctive terms")
    pos = dist[dist["favored"] == target].nlargest(top_n, "z_score")
    neg = dist[dist["favored"] != target].nsmallest(top_n, "z_score")
    d = pd.concat([neg, pos]).sort_values("z_score")
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(d) + 1.8))
    ax.barh(d["term"], d["z_score"], color=["#c44e52" if z < 0 else "#4c72b0" for z in d["z_score"]])
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel(f"log-odds z-score   (← others | {target} →)")
    ax.set_title(f"Terms that distinguish {target}", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_length_distribution(df: pd.DataFrame) -> plt.Figure:
    labels = list(dict.fromkeys(df["source_label"]))
    pal = source_palette(labels)
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 4, 4))
    parts = ax.boxplot([df.loc[df["source_label"] == s, "token_count"].values for s in labels],
                       patch_artist=True, showmeans=True)
    ax.set_xticklabels(labels)
    for patch, s in zip(parts["boxes"], labels):
        patch.set_facecolor(pal[s])
    ax.set_ylabel("Tokens per document (after cleaning)")
    ax.set_title("Document length by source", fontweight="bold")
    fig.tight_layout()
    return fig


                                                                               
                   
                                                                               
def plot_keyword_scores(tables: dict, title: str = "Top TF-IDF keywords", top_n: int = 20) -> plt.Figure:
    return _barh_grid(tables, "Keyword", "TF-IDF Score", title, "Mean TF-IDF score", top_n, annotate=False)


def plot_topic_terms(topics, n_terms: int = 10) -> plt.Figure:
    k = topics.n_topics
    cols = min(3, k)
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    pal = sns.color_palette("tab10", k)
    for ax in axes.ravel():
        ax.axis("off")
    for t, ax in zip(range(k), axes.ravel()):
        ax.axis("on")
        w = topics.term_weights[t][:n_terms][::-1]
        ax.barh([x for x, _ in w], [v for _, v in w], color=pal[t % 10])
        n = int(topics.topics.loc[t, "n_docs"])
        ax.set_title(f"Topic {t}  ({n} docs)", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
    fig.suptitle(f"{topics.method}: top terms per topic", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_topic_sources(topics) -> plt.Figure:
    dist = topics.source_distribution
    fig, ax = plt.subplots(figsize=(7.5, 4))
    pal = source_palette(dist.columns)
    dist.plot(kind="bar", stacked=True, ax=ax, color=[pal[c] for c in dist.columns], width=0.75)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Documents")
    ax.set_title("Documents per topic, by source", fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="")
    fig.tight_layout()
    return fig


                                                                               
                
                                                                               
def plot_confusion_matrix(cm: np.ndarray, labels, title: str = "Confusion matrix") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3.5, 1.1 * len(labels) + 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 14})
    ax.set_xlabel("Predicted source")
    ax.set_ylabel("True source")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_model_comparison(summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["accuracy"], 0.36, label="Accuracy", color="#4c72b0")
    ax.bar(x + 0.18, summary["macro_f1"], 0.36, label="Macro-F1", color="#dd8452")
    for xi, (a, f) in enumerate(zip(summary["accuracy"], summary["macro_f1"])):
        ax.text(xi - 0.18, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(xi + 0.18, f + 0.01, f"{f:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"])
    ax.set_ylim(0, 1.12)
    ax.set_title("Hold-out performance (20% stratified test set)", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_top_features(features: pd.DataFrame, model_name: str, top_n: int = 10) -> plt.Figure:
    data = {s: g.sort_values("weight", ascending=False).head(top_n) for s, g in features.groupby("source_label")}
    return _barh_grid(data, "term", "weight", f"{model_name}: most source-indicative terms", "Model weight",
                      top_n, annotate=False)


                                                                               
                
                                                                               
def plot_entities(entities: pd.DataFrame, entity_type: str, top_n: int = 12) -> plt.Figure:
    if entities is None or entities.empty:
        return _placeholder("No entities available (is the spaCy model installed?).", entity_type)
    sub = entities[entities["type"] == entity_type]
    data = {s: g.sort_values("mentions", ascending=False) for s, g in sub.groupby("source_label")}
    return _barh_grid(data, "entity", "mentions", f"Top {entity_type.lower()} entities", "Mentions", top_n)


def plot_keyword_trends(trend: pd.DataFrame, freq: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for col in trend.columns:
        ax.plot(trend.index, trend[col], marker="o", lw=1.8, label=col)
    ax.set_ylabel("Occurrences per 1,000 tokens")
    ax.set_title(f"Keyword trends over time (period = {freq})", fontweight="bold")
    ax.legend(ncol=2, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
