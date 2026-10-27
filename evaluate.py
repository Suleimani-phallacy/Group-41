#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as skm

import analysis as an
import scraper as sc
from preprocess import Lemmatizer, basic_clean, canonical_url, preprocess_dataframe


def load_df(a) -> pd.DataFrame:
    db = Path(a.data_dir) / "scraped_data.db"
    raw = sc.load_pages(db)
    if raw.empty:
        sys.exit(f"No pages in {db}. Run pipeline.py first.")
    df, rep = preprocess_dataframe(raw, tuple(a.extra_stopwords), a.lemmatizer, a.min_tokens,
                                   not a.no_strip_templates, a.template_threshold)
    print(f"Loaded {len(df)} analysable pages from {db} ({df['source_label'].nunique()} sources, "
          f"lemmatizer={rep.backend}, template lines removed={rep.template_lines_removed or 'none'})")
    return df


def out_dir(a) -> Path:
    d = Path(a.output_dir) / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def band(x: float, strong: float = 0.6, moderate: float = 0.3) -> str:
    return "strong" if x >= strong else "moderate" if x >= moderate else "weak"


def cluster_metrics(gold: pd.Series, pred: pd.Series):
    ct = pd.crosstab(pred.rename("topic"), gold.rename("gold"))
    rows, cols = linear_sum_assignment(-ct.values)
    mapping = {int(ct.index[i]): ct.columns[j] for i, j in zip(rows, cols)}
    mapped = pred.map(mapping).fillna("<unmapped>")
    m = {
        "ARI": skm.adjusted_rand_score(gold, pred), "NMI": skm.normalized_mutual_info_score(gold, pred),
        "homogeneity": skm.homogeneity_score(gold, pred), "completeness": skm.completeness_score(gold, pred),
        "purity": ct.max(axis=1).sum() / ct.values.sum(),
        "mapped_accuracy": float((mapped == gold).mean()),
        "mapped_macro_F1": skm.f1_score(gold, mapped, labels=sorted(gold.unique()), average="macro", zero_division=0),
    }
    return {k: round(float(v), 3) for k, v in m.items()}, ct, mapping


def cmd_template(a) -> None:
    df = load_df(a)
    per = max(1, a.n // df["source_label"].nunique())
    sample = pd.concat([g.sample(min(per, len(g)), random_state=a.seed) for _, g in df.groupby("source_label")])
    sample = sample.assign(excerpt=sample["raw_text"].str.replace(r"\s+", " ", regex=True).str[:300], theme="")
    path = out_dir(a) / "gold_template.csv"
    sample[["url", "source_label", "title", "excerpt", "theme"]].to_csv(path, index=False)
    print(f"\nWrote {len(sample)} pages to {path}\nOpen it, read each page, and type a short theme in the `theme` column.\n"
          "Tips: use 3-8 theme names, at least ~5 pages per theme; the same wording for the same theme "
          "(e.g. admissions / research / events / people); one theme per page = the page's MAIN topic; "
          "leave `theme` blank if you cannot decide. Label BEFORE looking at the model's topics, "
          "so the model cannot influence you.")


def cmd_themes(a) -> None:
    df = load_df(a)
    gold = pd.read_csv(a.labels)
    if not {"url", "theme"} <= set(gold.columns):
        sys.exit("The labels file needs `url` and `theme` columns (use `evaluate.py template`).")
    gold = gold.dropna(subset=["theme"]).assign(theme=lambda d: d["theme"].astype(str).str.strip().str.lower())
    gold = gold[gold["theme"] != ""]
    gold["_c"] = gold["url"].map(canonical_url)
    df = df.assign(_c=df["url"].map(canonical_url))
    lab = df.reset_index().merge(gold[["_c", "theme"]].drop_duplicates("_c"), on="_c").set_index("index")
    n_themes = lab["theme"].nunique()
    print(f"{len(lab)} hand-labelled pages matched, {n_themes} themes: {lab['theme'].value_counts().to_dict()}")
    if len(lab) < 20 or n_themes < 2:
        print("WARNING: fewer than 20 labelled pages or <2 themes - the numbers below are not reliable.")
    k = a.topics or n_themes
    print(f"Fitting topic models with k = {k} on ALL {len(df)} pages, scoring on the labelled ones.\n")
    rows = []
    for method in ("lda", "kmeans"):
        t = an.run_topic_model(df, method, n_topics=k, random_state=a.seed)
        m, ct, mapping = cluster_metrics(lab["theme"], t.doc_topics.loc[lab.index])
        mirror = skm.adjusted_rand_score(df["source_label"], t.doc_topics) if df["source_label"].nunique() > 1 else float("nan")
        rows.append({"method": method, **m, "topic_vs_source_ARI": round(float(mirror), 3),
                     "coherence_npmi": t.metrics["coherence_npmi"]})
        print(f"--- {method.upper()}: contingency (rows = topic, columns = your theme) ---\n{ct}\n"
              f"best one-to-one mapping topic->theme: {mapping}\n")
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(out_dir(a) / "eval_themes.csv", index=False)
    best = res.loc[res["ARI"].idxmax()]
    print(f"\nVerdict ({best['method']}): ARI {best['ARI']} -> agreement with your labels is {band(best['ARI'])}.")
    if best["topic_vs_source_ARI"] > 0.5:
        print("!! The topics largely mirror the WEBSITES (topic_vs_source_ARI > 0.5): site templates/brand words still "
              "dominate. Lower --template-threshold, add --extra-stopwords for brand/place names, or crawl more pages.")
    print("Rules of thumb (not standards): ARI/NMI >= 0.6 strong, 0.3-0.6 moderate, < 0.3 weak. ARI ~ 0 = chance. "
          "Low scores can also mean YOUR themes are finer/broader than the model's topics - read the contingency table.")


def cmd_stability(a) -> None:
    df = load_df(a)
    k = a.topics or 5
    print(f"\nStability of {a.topic_method.upper()} with k={k} over {a.runs} seeds")
    runs = [an.run_topic_model(df, a.topic_method, n_topics=k, random_state=s) for s in range(a.runs)]
    aris = [skm.adjusted_rand_score(x.doc_topics, y.doc_topics) for x, y in itertools.combinations(runs, 2)]
    coh = [r.metrics["coherence_npmi"] for r in runs]
    print(f"  pairwise ARI between runs: mean {np.mean(aris):.2f}, min {np.min(aris):.2f}   "
          f"coherence(NPMI): {np.mean(coh):.2f} +/- {np.std(coh):.2f}")
    print("  -> " + ("stable (>= 0.8)" if np.mean(aris) >= 0.8 else "moderately stable" if np.mean(aris) >= 0.5 else
                      "UNSTABLE: topics change with the random seed; reduce k, add pages, or add stop-words"))
    for method in ("lda", "kmeans"):
        best, sweep = an.select_n_topics(df, method, random_state=0)
        print(f"\n{method.upper()} k-sweep (best k by coherence = {best}):\n{sweep.to_string(index=False)}")
        sweep.to_csv(out_dir(a) / f"k_sweep_{method}.csv", index=False)
    print("\nStability/coherence are INTERNAL checks: they show the topics are consistent and coherent, not that they "
          "match what a human would call the themes. Use `themes` for that.")


def cmd_classifier(a) -> None:
    df = load_df(a)
    if df["source_label"].nunique() < 2:
        sys.exit("Need at least two sources.")
    res = an.train_source_classifier(df, random_state=a.seed)
    print("\n" + (res.error or res.summary.to_string(index=False)))
    for w in res.warnings:
        print("!", w)
    ch = an.chance_baseline(df, n_permutations=a.permutations, random_state=a.seed)
    if ch:
        print(f"\nChance check: observed CV macro-F1 {ch['observed_macro_f1']} vs shuffled-label mean {ch['shuffled_labels_mean']} "
              f"(max {ch['shuffled_labels_max']}); permutation p = {ch['p_value']} (smallest possible: 1/{ch['n_permutations'] + 1}); "
              f"majority-class accuracy {ch['majority_class_accuracy']}")
        pd.Series(ch).to_csv(out_dir(a) / "classifier_chance.csv", header=["value"])
    audit = an.leakage_audit(df, random_state=a.seed)
    if len(audit):
        print("\nLeakage audit - macro-F1 after deleting the top-N identifier terms of each source:\n" + audit.to_string(index=False))
        audit.to_csv(out_dir(a) / "classifier_leakage_audit.csv", index=False)
        print("  -> " + an.leakage_verdict(audit))
    print("\nReading it: observed >> shuffled means the model finds real signal; but that signal may be brand words. "
          "The audit tells you which. For a *thematic* comparison of sources use the theme profile "
          "(outputs/tables/theme_profile_by_source_pct.csv), not classifier accuracy.")


def cmd_keywords(a) -> None:
    df = load_df(a)
    gold = pd.read_csv(a.gold)
    if not {"source_label", "keyword"} <= set(gold.columns):
        sys.exit("The gold file needs columns `source_label,keyword` (use source_label=ALL for corpus-wide keywords).")
    lem = Lemmatizer(a.lemmatizer)
    norm = lambda terms: [t for t in lem.process([basic_clean(x) for x in terms]) ]
    kw = an.extract_keywords(df, top_k=max(a.k, 30))

    def match(k: str, g: str) -> bool:
        ks, gs = set(k.split()), set(g.split())
        return bool(ks and gs) and (gs <= ks or ks <= gs)

    rows = []
    for scope, g in gold.groupby("source_label"):
        sub = df if scope.upper() == "ALL" else df[df["source_label"] == scope]
        if sub.empty:
            print(f"skipping unknown source '{scope}'")
            continue
        gterms = [t for t in norm(g["keyword"].astype(str).tolist()) if t]
        table = kw.overall if scope.upper() == "ALL" else kw.per_source[scope]
        freq = an.word_frequencies(sub["cleaned_text"], max(a.k, 30))["term"].tolist()
        for name, ranked in (("TF-IDF (this tool)", table["Keyword"].tolist()), ("raw frequency baseline", freq)):
            top = ranked[:a.k]
            p = np.mean([any(match(k, t) for t in gterms) for k in top]) if top else 0
            r = np.mean([any(match(k, t) for k in top) for t in gterms]) if gterms else 0
            rows.append({"scope": scope, "method": name, f"precision@{a.k}": round(float(p), 3),
                         f"recall@{a.k}": round(float(r), 3), "gold_terms": len(gterms)})
    res = pd.DataFrame(rows)
    print("\n" + res.to_string(index=False))
    res.to_csv(out_dir(a) / "eval_keywords.csv", index=False)
    print("\nTF-IDF should beat the raw-frequency baseline (frequency just returns generic words). Matching is lenient: "
          "a keyword counts if its words contain, or are contained in, one of your terms. Recall is capped by how many of "
          "your terms actually occur on the scraped pages.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--lemmatizer", choices=["auto", "spacy", "nltk"], default="auto")
    p.add_argument("--extra-stopwords", nargs="*", default=[])
    p.add_argument("--min-tokens", type=int, default=20)
    p.add_argument("--no-strip-templates", action="store_true")
    p.add_argument("--template-threshold", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topics", type=int, default=0)
    p.add_argument("--topic-method", choices=["lda", "kmeans"], default="lda")
    sub = p.add_subparsers(dest="cmd", required=True)
    
    s = sub.add_parser("template")
    s.add_argument("--n", type=int, default=60)
    s.set_defaults(fn=cmd_template)
    
    s = sub.add_parser("themes")
    s.add_argument("--labels", required=True)
    s.set_defaults(fn=cmd_themes)
    
    s = sub.add_parser("stability")
    s.add_argument("--runs", type=int, default=5)
    s.set_defaults(fn=cmd_stability)
    
    s = sub.add_parser("classifier")
    s.add_argument("--permutations", type=int, default=50)
    s.set_defaults(fn=cmd_classifier)
    
    s = sub.add_parser("keywords")
    s.add_argument("--gold", required=True)
    s.add_argument("--k", type=int, default=20)
    s.set_defaults(fn=cmd_keywords)
    
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
