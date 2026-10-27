#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd

import analysis as an
import scraper as sc
import visualize as viz
from preprocess import PreprocessReport, preprocess_dataframe

log = logging.getLogger("pipeline")
ProgressCB = Callable[[str, float], None]


class PipelineError(RuntimeError):
    """Fatal, user-presentable pipeline failure (carries the scrape log when available)."""

    def __init__(self, message: str, scrape_log: Optional[pd.DataFrame] = None):
        super().__init__(message)
        self.scrape_log = scrape_log


@dataclass
class PipelineConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    scraper: sc.ScraperConfig = field(default_factory=sc.ScraperConfig)
    skip_scrape: bool = False
    lemmatizer: str = "auto"
    extra_stopwords: tuple = ()
    min_tokens: int = 20
    top_k_keywords: int = 20
    n_topics: int = 5
    topic_method: str = "lda"
    run_classification: bool = True
    run_ner: bool = True
    run_temporal: bool = True
    test_size: float = 0.2
    random_state: int = 42

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "scraped_data.db"

    @property
    def csv_path(self) -> Path:
        return Path(self.data_dir) / "scraped_dataset.csv"


@dataclass
class PipelineResults:
    config: PipelineConfig
    df: pd.DataFrame
    sources: list
    scrape_log: pd.DataFrame
    prep: PreprocessReport
    word_freq: dict
    bigrams: dict
    trigrams: dict
    keywords: an.KeywordResults
    keyword_eval: dict
    distinctive: Optional[pd.DataFrame] = None
    topics: Optional[an.TopicResults] = None
    classification: Optional[an.ClassificationResults] = None
    entities: Optional[pd.DataFrame] = None
    temporal: Optional[an.TemporalResults] = None
    figures: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    paths: dict = field(default_factory=dict)


ALL = "All sources"


def _noop(msg: str, frac: float) -> None:
    pass


def run_pipeline(sources: Optional[dict], cfg: Optional[PipelineConfig] = None,
                 progress: Optional[ProgressCB] = None) -> PipelineResults:
    """
    sources : {source_label: [url, ...]}  (ignored when cfg.skip_scrape is True)
    """
    cfg = cfg or PipelineConfig()
    progress = progress or _noop
    warnings: list[str] = []
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    fig_dir = Path(cfg.output_dir) / "figures"
    tab_dir = Path(cfg.output_dir) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    sc.init_db(cfg.db_path)

    scrape_log = pd.DataFrame(columns=["source_label", "url", "status", "detail"])
    if cfg.skip_scrape:
        progress("Loading pages from SQLite…", 0.05)
        raw = sc.load_pages(cfg.db_path)
        if sources:
            raw = raw[raw["source_label"].isin(sources.keys())]
        if raw.empty:
            raise PipelineError(f"No pages found in {cfg.db_path}. Run a scrape first.")
    else:
        if not sources or not any(sources.values()):
            raise PipelineError("No URLs supplied.")
        pages, events = sc.scrape_sources(sources, cfg.scraper,
                                          lambda m, f: progress(m, 0.02 + 0.38 * f))
        scrape_log = sc.events_to_frame(events)
        if not pages:
            reasons = scrape_log["status"].value_counts().to_dict()
            raise PipelineError(f"No usable pages were scraped. Outcomes: {reasons}", scrape_log)
        ids = sc.save_pages(pages, cfg.db_path)
        raw = sc.load_pages(cfg.db_path, ids)

    progress("Cleaning, de-duplicating and lemmatizing…", 0.42)
    df, prep = preprocess_dataframe(raw, cfg.extra_stopwords, cfg.lemmatizer, cfg.min_tokens,
                                    lambda i, n: progress(f"Lemmatizing {i}/{n}", 0.42 + 0.13 * i / max(n, 1)))
    if df.empty:
        raise PipelineError("All pages were empty, duplicates, or too short after cleaning.", scrape_log)
    if prep.backend == "rule":
        warnings.append("Neither spaCy nor NLTK lemmatizer available - used a simple rule-based fallback. "
                        "Install spaCy + en_core_web_sm for better results.")
    sc.delete_pages(cfg.db_path, prep.duplicate_ids)
    sc.update_cleaned_text(cfg.db_path, df)
    csv_path = sc.export_csv(cfg.db_path, cfg.csv_path)
    labels = list(dict.fromkeys(df["source_label"]))
    for lab, n in df["source_label"].value_counts().items():
        if n < 10:
            warnings.append(f"'{lab}' has only {n} usable page(s); statistics will be noisy. Increase max pages.")

    progress("Exploratory analysis…", 0.58)
    groups = OrderedDict([(ALL, df["cleaned_text"])] + [(s, df.loc[df["source_label"] == s, "cleaned_text"]) for s in labels])
    word_freq = {k: an.word_frequencies(v, 50) for k, v in groups.items()}
    bigrams = {k: an.top_ngrams(v, 2, 20) for k, v in groups.items()}
    trigrams = {k: an.top_ngrams(v, 3, 20) for k, v in groups.items()}
    distinctive = an.distinctive_terms(df, labels[0]) if len(labels) > 1 else None

    progress("TF-IDF keyword extraction…", 0.64)
    kw = an.extract_keywords(df, top_k=max(cfg.top_k_keywords, 20))
    keyword_eval = {ALL: an.evaluate_keywords(kw.overall, df, cfg.top_k_keywords, ALL)}
    for s in labels:
        keyword_eval[s] = an.evaluate_keywords(kw.per_source[s], df[df["source_label"] == s], cfg.top_k_keywords, s, corpus=df)

    res = PipelineResults(cfg, df, labels, scrape_log, prep, word_freq, bigrams, trigrams, kw, keyword_eval,
                          distinctive, warnings=warnings, paths={"db": cfg.db_path, "csv": csv_path})

    def stage(name: str, frac: float, fn):
        progress(name, frac)
        try:
            return fn()
        except Exception as exc:
            log.exception("%s failed", name)
            res.warnings.append(f"{name} failed: {exc}")
            return None

    res.topics = stage("Topic modelling…", 0.72, lambda: an.run_topic_model(df, cfg.topic_method, n_topics=cfg.n_topics, random_state=cfg.random_state))
    if cfg.run_classification and len(labels) > 1:
        res.classification = stage("Training source classifiers…", 0.80, lambda: an.train_source_classifier(df, cfg.test_size, cfg.random_state))
    if cfg.run_ner:
        res.entities = stage("Named-entity recognition…", 0.86, lambda: an.extract_entities(df))
        if res.entities is not None and res.entities.empty:
            res.warnings.append("NER produced no entities (spaCy model 'en_core_web_sm' not installed?).")
    if cfg.run_temporal:
        top_unigrams = [k for k in kw.overall["Keyword"] if " " not in k][:5]
        res.temporal = stage("Temporal analysis…", 0.90, lambda: an.temporal_keyword_trends(df, top_unigrams))

    progress("Rendering figures and saving tables…", 0.93)
    res.figures = build_figures(res)
    for name, fig in res.figures.items():
        viz.save_figure(fig, fig_dir / f"{name}.png")
    save_tables(res, tab_dir)
    res.paths.update(figures=fig_dir, tables=tab_dir)
    progress("Done", 1.0)
    return res


def build_figures(r: PipelineResults) -> "OrderedDict[str, object]":
    """Render every chart once. A failing chart is logged and skipped, never fatal."""
    figs: "OrderedDict[str, object]" = OrderedDict()
    multi = len(r.sources) > 1

    def add(name: str, fn: Callable) -> None:
        try:
            figs[name] = fn()
            plt.close(figs[name])
        except Exception as exc:
            log.exception("figure %s failed", name)
            r.warnings.append(f"Figure '{name}' could not be rendered: {exc}")

    add("eda_top_words_all", lambda: viz.plot_top_words({ALL: r.word_freq[ALL]}, "Top 20 words (all sources)"))
    if multi:
        add("eda_top_words_by_source", lambda: viz.plot_top_words({s: r.word_freq[s] for s in r.sources}, "Top 20 words by source"))
        add("eda_comparative_terms", lambda: viz.plot_comparative_terms(r.df))
        add("eda_distinctive_terms", lambda: viz.plot_distinctive_terms(r.distinctive, r.sources[0]))
    ng_keys = r.sources if multi else [ALL]
    add("eda_bigrams", lambda: viz.plot_ngrams({k: r.bigrams[k] for k in ng_keys}, "Top bigrams"))
    add("eda_trigrams", lambda: viz.plot_ngrams({k: r.trigrams[k] for k in ng_keys}, "Top trigrams"))

    def clouds():
        c = OrderedDict([(ALL, an.term_counter(r.df["cleaned_text"]))])
        for s in r.sources:
            c[s] = an.term_counter(r.df.loc[r.df["source_label"] == s, "cleaned_text"])
        return viz.plot_wordclouds(c)

    add("eda_wordclouds", clouds)
    add("eda_doc_lengths", lambda: viz.plot_length_distribution(r.df))
    kw_tables = {s: r.keywords.per_source[s] for s in r.sources} if multi else {ALL: r.keywords.overall}
    add("keywords_tfidf", lambda: viz.plot_keyword_scores(kw_tables, "Top TF-IDF keywords"))
    if r.topics is not None:
        add("topics_terms", lambda: viz.plot_topic_terms(r.topics))
        if multi:
            add("topics_sources", lambda: viz.plot_topic_sources(r.topics))
    c = r.classification
    if c is not None and not c.error and c.models:
        add("clf_comparison", lambda: viz.plot_model_comparison(c.summary))
        for name, m in c.models.items():
            add(f"clf_confusion_{name}", lambda m=m, name=name: viz.plot_confusion_matrix(m.confusion, m.labels, f"{name}: confusion matrix"))
            add(f"clf_features_{name}", lambda m=m, name=name: viz.plot_top_features(m.top_features, name))
    if r.entities is not None and not r.entities.empty:
        for etype in ("Organization", "Location", "Person"):
            add(f"ner_{etype.lower()}", lambda etype=etype: viz.plot_entities(r.entities, etype))
    if r.temporal is not None and r.temporal.trend is not None:
        add("temporal_trends", lambda: viz.plot_keyword_trends(r.temporal.trend, r.temporal.freq))
    return figs


def save_tables(r: PipelineResults, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    r.keywords.overall.to_csv(out_dir / "keywords_overall.csv", index=False)
    for s, t in r.keywords.per_source.items():
        t.to_csv(out_dir / f"keywords_{_slug(s)}.csv", index=False)
    r.keywords.per_document.to_csv(out_dir / "keywords_per_document.csv", index=False)
    for k, v in r.word_freq.items():
        v.to_csv(out_dir / f"word_freq_{_slug(k)}.csv", index=False)
    for k, v in r.bigrams.items():
        v.to_csv(out_dir / f"bigrams_{_slug(k)}.csv", index=False)
    for k, v in r.trigrams.items():
        v.to_csv(out_dir / f"trigrams_{_slug(k)}.csv", index=False)
    if r.distinctive is not None:
        r.distinctive.to_csv(out_dir / "distinctive_terms.csv", index=False)
    if r.topics is not None:
        r.topics.topics.to_csv(out_dir / "topics.csv", index=False)
        r.df.assign(topic=r.topics.doc_topics)[["id", "source_label", "title", "url", "topic"]].to_csv(out_dir / "document_topics.csv", index=False)
    if r.classification is not None and not r.classification.error:
        r.classification.summary.to_csv(out_dir / "classification_summary.csv", index=False)
        r.classification.predictions.to_csv(out_dir / "classification_predictions.csv", index=False)
        for name, m in r.classification.models.items():
            (out_dir / f"classification_report_{name}.txt").write_text(m.report_text, encoding="utf-8")
    if r.entities is not None and not r.entities.empty:
        r.entities.to_csv(out_dir / "entities.csv", index=False)
    if r.temporal is not None and r.temporal.trend is not None:
        r.temporal.trend.to_csv(out_dir / "keyword_trends.csv")
    (out_dir / "keyword_quality_report.md").write_text(
        "\n\n".join(v["report"] for v in r.keyword_eval.values()), encoding="utf-8")


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Keyword & Theme Intelligence System",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("input (choose one style)")
    g.add_argument("--urls", nargs="+", metavar="URL", help="single URL or list of URLs, analysed as one source")
    g.add_argument("--source1", nargs="+", metavar="URL", help="seed URL(s) / domain of source 1")
    g.add_argument("--source2", nargs="+", metavar="URL", help="seed URL(s) / domain of source 2")
    g.add_argument("--label1", default="Source_1")
    g.add_argument("--label2", default="Source_2")
    g.add_argument("--skip-scrape", action="store_true", help="analyse the pages already stored in the SQLite DB")
    s = p.add_argument_group("scraping (polite defaults)")
    s.add_argument("--no-crawl", action="store_true", help="only fetch the given URLs; do not discover links")
    s.add_argument("--max-pages", type=int, default=30, help="max pages fetched per source")
    s.add_argument("--max-depth", type=int, default=2)
    s.add_argument("--delay", type=float, default=2.0, help="seconds between requests to the same host (>= 1 recommended)")
    s.add_argument("--timeout", type=int, default=15)
    s.add_argument("--min-words", type=int, default=60, help="discard pages with fewer words")
    s.add_argument("--no-ssl-fallback", action="store_true", help="never retry without SSL verification")
    a = p.add_argument_group("analysis")
    a.add_argument("--topics", type=int, default=5)
    a.add_argument("--topic-method", choices=["lda", "kmeans"], default="lda")
    a.add_argument("--top-k", type=int, default=20, help="keywords reported/evaluated")
    a.add_argument("--lemmatizer", choices=["auto", "spacy", "nltk"], default="auto")
    a.add_argument("--extra-stopwords", nargs="*", default=[], help="additional words to drop (e.g. site/brand names)")
    a.add_argument("--no-ner", action="store_true")
    a.add_argument("--no-classification", action="store_true")
    a.add_argument("--no-temporal", action="store_true")
    o = p.add_argument_group("output")
    o.add_argument("--data-dir", default="data", help="holds scraped_data.db and scraped_dataset.csv")
    o.add_argument("--output-dir", default="outputs", help="figures/ and tables/ are written here")
    o.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sources: dict = {}
    if args.urls:
        sources[args.label1] = args.urls
    if args.source1:
        sources[args.label1] = args.source1
    if args.source2:
        sources[args.label2] = args.source2
    if not sources and not args.skip_scrape:
        build_parser().error("provide --urls, or --source1 [--source2], or --skip-scrape")
    if args.delay < 1.0:
        print("Note: delay < 1s is impolite to most servers; consider --delay 2 or higher.", file=sys.stderr)

    cfg = PipelineConfig(
        data_dir=args.data_dir, output_dir=args.output_dir, skip_scrape=args.skip_scrape,
        scraper=sc.ScraperConfig(delay=args.delay, timeout=args.timeout, min_words=args.min_words,
                                 crawl=not args.no_crawl, max_pages_per_source=args.max_pages,
                                 max_depth=args.max_depth, allow_insecure_ssl_fallback=not args.no_ssl_fallback),
        lemmatizer=args.lemmatizer, extra_stopwords=tuple(args.extra_stopwords), top_k_keywords=args.top_k,
        n_topics=args.topics, topic_method=args.topic_method, run_ner=not args.no_ner,
        run_classification=not args.no_classification, run_temporal=not args.no_temporal)

    last = {"msg": ""}

    def progress(msg: str, frac: float) -> None:
        if not msg.startswith(("Lemmatizing", "[")) and msg != last["msg"]:
            print(f"[{frac:4.0%}] {msg}")
            last["msg"] = msg

    try:
        r = run_pipeline(sources or None, cfg, progress)
    except PipelineError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if exc.scrape_log is not None and len(exc.scrape_log):
            print(exc.scrape_log.to_string(index=False), file=sys.stderr)
        return 1
    print_summary(r)
    return 0


def print_summary(r: PipelineResults) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    sep = "=" * 78
    print(f"\n{sep}\nDATASET\n{sep}")
    print(r.df.groupby("source_label").agg(documents=("id", "count"), mean_tokens=("token_count", "mean")).round(0).to_string())
    print(f"Preprocessing: {r.prep.as_dict()}")
    if len(r.scrape_log):
        print("Scrape outcomes:", r.scrape_log["status"].value_counts().to_dict())
    for s in r.sources:
        print(f"\n{sep}\nTOP KEYWORDS - {s}\n{sep}\n{an.format_keyword_table(r.keywords.per_source[s].head(r.config.top_k_keywords))}")
    print(f"\n{sep}\nKEYWORD QUALITY\n{sep}\n" + "\n\n".join(v["report"] for v in r.keyword_eval.values()))
    if r.topics is not None:
        print(f"\n{sep}\nTOPICS ({r.topics.method}) {r.topics.metrics}\n{sep}")
        print(r.topics.topics[["topic_id", "n_docs", "top_terms"]].to_string(index=False))
    c = r.classification
    if c is not None:
        print(f"\n{sep}\nSOURCE CLASSIFICATION\n{sep}")
        print(c.error or c.summary.to_string(index=False))
        for m in (c.models or {}).values():
            print(f"\n{m.name}\n{m.report_text}")
        for w in c.warnings:
            print("!", w)
    if r.temporal is not None and r.temporal.message:
        print("\nTemporal:", r.temporal.message)
    for w in r.warnings:
        print("WARNING:", w)
    print(f"\nSaved: {r.paths.get('db')}, {r.paths.get('csv')}, {r.paths.get('figures')}, {r.paths.get('tables')}")


if __name__ == "__main__":
    sys.exit(main())
