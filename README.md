# Keyword Identification System (GROUP 41)

An end-to-end data science and Natural Language Processing (NLP) project for identifying important keywords, discovering hidden themes, and comparing textual content across websites and online sources.

The system collects publicly accessible webpage content, preprocesses unstructured text, performs exploratory data analysis, extracts keywords using TF-IDF, discovers topics using machine learning, and generates comparative visualizations and analytical reports.

## 1. Project Overview

Websites, research articles, and online news platforms contain large amounts of unstructured textual information. Extracting meaningful keywords and identifying recurring themes can help us understand the subjects discussed across different sources.

The **Keyword Identification System** automates this process through a modular pipeline that transforms webpage content into structured datasets and interpretable analytical results.

The system supports comparisons between multiple sources, such as institutional websites, research platforms, and news websites. It also includes optional machine learning extensions for source classification, named-entity recognition, and temporal keyword analysis.

## 2. Project Objectives

- Collect textual content from publicly accessible websites.
- Extract page titles, URLs, publication dates when available, and main text.
- Clean and preprocess collected text for NLP analysis.
- Perform exploratory data analysis (EDA) on word frequencies and n-grams.
- Identify important keywords and phrases using TF-IDF.
- Discover latent themes using topic modelling and clustering.
- Compare vocabulary patterns and distinctive terms across sources.
- Train machine learning models to classify documents by source.
- Extract named entities such as people, organizations, and locations.
- Explore keyword trends over time when publication dates are available.
- Generate visualizations, structured tables, and evaluation reports.

## 3. Key Features

### 3.1 Ethical Web Scraping and Data Collection

The scraping module collects webpage titles, URLs, publication dates when available, and main textual content.

Features include:

- Checking website `robots.txt` rules before fetching pages.
- Considering website-specified crawl delays.
- Identifying the crawler through its `User-Agent` header.
- Sequential requests and per-host rate limiting.
- Configurable timeouts, retries, and error handling.
- Main-text extraction using Trafilatura, with BeautifulSoup as a fallback.
- Optional same-site link discovery using breadth-first crawling.
- Configurable crawl depth and maximum page counts.
- Persistent storage using SQLite and CSV export.
- Logging of successful requests and unsuccessful collection attempts.

Pages that cannot be accessed or do not contain enough usable text are skipped.

### 3.2 Text Preprocessing

Raw webpage text often contains HTML, navigation menus, advertisements, duplicate content, and other material that can interfere with meaningful analysis.

The preprocessing pipeline includes:

- HTML, script, and other irrelevant markup removal.
- URL and email removal.
- Lowercasing and punctuation removal.
- Removal of common website boilerplate phrases.
- English stopword removal.
- Lemmatization using spaCy, NLTK, or a rule-based fallback.
- Duplicate detection using canonical URLs and normalized-text hashes.
- Filtering of documents with insufficient text after cleaning.
- Token counting for subsequent analysis.

The output is a structured, analysis-ready dataset containing cleaned text and relevant document metadata.

### 3.3 Exploratory Data Analysis (EDA)

EDA helps identify recurring vocabulary, common phrases, differences in document length, and variations between sources.

The analysis includes:

- Most frequent individual words.
- Frequent bigrams and trigrams.
- Relative term frequencies per 10,000 tokens.
- Word clouds for the overall corpus and individual sources.
- Document-length distributions.
- Comparative word-frequency visualizations.
- Distinctive-term analysis using log-odds z-scores.

These analyses provide an initial understanding of the collected corpus before applying more advanced NLP methods.

### 3.4 TF-IDF Keyword Extraction

Term Frequency–Inverse Document Frequency (TF-IDF) identifies words and phrases that are informative within a collection of documents.

The implementation supports:

- Unigrams and bigrams.
- Overall keyword rankings.
- Source-specific keyword rankings.
- Document-level keyword extraction.
- TF-IDF scores.
- Raw term frequencies.
- Document frequencies.
- Evaluation reports for reviewing the quality of extracted keywords.

Keyword rankings can be compared across sources to identify shared themes and source-specific terminology.

### 3.5 Topic Modelling

The system supports two approaches to discovering groups of related terms and documents.

**Latent Dirichlet Allocation (LDA)**

LDA is a probabilistic topic-modelling algorithm that represents documents as mixtures of topics and topics as distributions over words. It helps identify recurring themes in a text corpus.

**K-Means Clustering**

K-Means groups documents according to their TF-IDF feature representations. Documents with similar feature patterns are assigned to the same cluster.

The topic-analysis outputs include:

- Top terms associated with each topic.
- Automatically generated topic labels based on leading terms.
- Dominant topic assignments for documents.
- Representative documents.
- Number and proportion of documents in each topic.
- Topic distributions across sources.
- Topic-term and topic-source visualizations.

The resulting topics require interpretation; their labels are descriptive summaries rather than guaranteed semantic classifications.

### 3.6 Source Classification

Source classification is an optional supervised machine learning task that predicts the source label of a document based on its textual features.

The implementation includes:

- TF-IDF-based text representation.
- Logistic Regression.
- Linear Support Vector Classification (Linear SVC).
- Training and test data splitting.
- Cross-validation where supported by the implementation.
- Accuracy and F1-score.
- Classification reports.
- Confusion matrices.
- Model comparison and informative-feature analysis.

This task can help investigate whether different websites exhibit distinguishable vocabulary patterns.

Classification requires multiple source labels and sufficient labelled data. High classification accuracy does not necessarily imply that the model has learned meaningful subject-matter differences; it may also learn differences in writing style or website-specific terminology.

### 3.7 Named-Entity Recognition (NER)

The NER module uses spaCy to identify named entities appearing in the collected text.

Entity categories may include:

- People.
- Organizations.
- Locations.
- Other entity types supported by the installed spaCy model.

Extracted entities can help reveal which individuals, institutions, and geographical locations are discussed across the collected documents.

NER results depend on the quality of the text and the availability of the required spaCy language model.

### 3.8 Temporal Keyword Analysis

When usable publication dates are available, the temporal analysis module examines changes in the frequency of selected keywords over time.

This can help investigate:

- Changes in the prominence of particular terms.
- Emerging or declining vocabulary.
- Differences in keyword trends across sources.
- Temporal patterns in the collected articles.

Temporal results depend on the availability and accuracy of publication-date information.

### 3.9 Visualization and Reporting

The visualization module generates charts for the different stages of the pipeline.

These include:

- Word-frequency plots.
- Bigram and trigram charts.
- Word clouds.
- Comparative term-frequency plots.
- Distinctive-term plots.
- Document-length distributions.
- TF-IDF keyword rankings.
- Topic-term charts.
- Topic distributions by source.
- Classification confusion matrices.
- Model-performance comparisons.
- Named-entity and temporal keyword visualizations when applicable.

Figures are saved as image files, while analysis results are exported as structured tables.

## 4. Technologies Used

| Category | Technologies |
|---|---|
| Programming language | Python |
| Web scraping | Requests, BeautifulSoup, Trafilatura |
| Data manipulation | Pandas, NumPy |
| Text preprocessing | spaCy, NLTK |
| NLP and feature extraction | scikit-learn, TF-IDF, CountVectorizer |
| Topic modelling | LDA, K-Means |
| Machine learning | Logistic Regression, Linear SVC |
| Numerical and statistical operations | SciPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Data storage | SQLite, CSV |
| Optional dashboard framework | Streamlit |

## 5. Project Structure

The core project files are organized as follows:

```text
keyword-theme-intelligence/
│
├── scraper.py          # Ethical web scraping and SQLite persistence
├── preprocess.py       # Text cleaning, deduplication and lemmatization
├── analysis.py         # EDA, TF-IDF, topics, classification and NER
├── visualize.py        # Chart and visualization functions
├── pipeline.py         # End-to-end orchestration and CLI
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
│
├── data/               # Generated data files
│   ├── scraped_data.db
│   └── scraped_dataset.csv
│
└── outputs/            # Generated analysis results
    ├── figures/
    └── tables/
```

The `data/` and `outputs/` directories are generated during execution. Their contents depend on the input sources and the analyses performed.

## 7. Running the Project

The project can be run using `pipeline.py`, which collects website data and performs the text analysis.

### 7.1 Compare Two Websites

Example :To compare the websites of IISER Mohali and IISER Pune, run:

```bash
python pipeline.py --source1 https://www.iisermohali.ac.in --source2 https://www.iiserpune.ac.in --label1 IISER_Mohali --label2 IISER_Pune
```

This collects accessible text from both websites, analyses the content, and generates keywords, topics, graphs, and comparison results.

### 7.2 Analyse Specific Webpages

To analyse specific webpages without collecting additional pages from the same websites, run:

```bash
python pipeline.py --urls https://example.org/page1 https://example.org/page2 --no-crawl
```

Replace the example URLs with the webpages you want to analyse.

### 7.3 Choose a Topic-Modelling Method

The project supports two methods for finding topics in text.

For LDA:

```bash
python pipeline.py --source1 https://example.org --topic-method lda --topics 5
```

For K-Means:

```bash
python pipeline.py --source1 https://example.org --topic-method kmeans --topics 5
```

Here, `--topics 5` sets the number of topics or groups to five.

### 7.4 Set the Number of Pages to Collect

To limit the number of pages collected and control how deeply the scraper follows links, run:

```bash
python pipeline.py --source1 https://example.org --max-pages 10 --max-depth 1 --delay 2
```

This sets a limit of 10 pages per source, a crawl depth of 1, and a delay of at least 2 seconds between requests to the same website.

Always follow the website's scraping rules and access restrictions.

### 7.5 Turn Off Optional Analyses

To skip source classification, named-entity recognition, and keyword trend analysis, run:

```bash
python pipeline.py --source1 https://example.org --no-classification --no-ner --no-temporal
```

### 7.6 Analyse Previously Collected Data

If you have already collected data and want to analyse it again without scraping the websites, run:

```bash
python pipeline.py --skip-scrape
```

This uses the previously saved data in `data/scraped_data.db`. The database must already contain usable data.

### 7.7 View All Available Options

To see all the commands and settings supported by the project, run:

```bash
python pipeline.py --help
```

**Note:** Replace example URLs with your chosen websites. The results depend on the data collected and whether the websites allow automated access.

## 8. Generated Outputs

The pipeline saves collected data, figures, and analytical tables.

| Output location | Description |
|---|---|
| `data/scraped_data.db` | SQLite database containing collected webpage records |
| `data/scraped_dataset.csv` | CSV export of the scraped dataset |
| `outputs/figures/` | Generated visualization images |
| `outputs/tables/` | Exported analysis tables and reports |
| `outputs/tables/keywords_overall.csv` | Overall keyword rankings |
| `outputs/tables/keywords_per_document.csv` | Document-level keyword rankings |
| `outputs/tables/topics.csv` | Topic summaries and associated terms |
| `outputs/tables/document_topics.csv` | Dominant topic assignments |
| `outputs/tables/distinctive_terms.csv` | Source-distinguishing terms, when applicable |
| `outputs/tables/classification_summary.csv` | Classifier performance summary, when available |
| `outputs/tables/classification_predictions.csv` | Source-classification predictions, when available |
| `outputs/tables/entities.csv` | Named-entity results, when available |
| `outputs/tables/keyword_trends.csv` | Temporal keyword data, when available |
| `outputs/tables/keyword_quality_report.md` | Keyword-quality evaluation report |

Additional exports may include source-specific keyword tables, word-frequency tables, n-gram tables, and classification reports.

Some outputs are conditional on the number of sources, availability of dates, successful model execution, and the data collected. The precise output filenames should be verified against the generated files for the version being run.


### Source Classification

Classification performance is evaluated using accuracy, F1-score, classification reports, and confusion matrices. Results should be interpreted in light of the number of documents, class balance, and the possibility of source-specific stylistic signals.

### Comparative Analysis

Relative term frequencies and distinctive-term scores are used to identify similarities and differences between sources.

### Temporal Analysis

When reliable publication dates are available, selected keyword frequencies are examined over time.

These evaluation methods provide different perspectives on the quality of extracted keywords, the interpretability of discovered themes, and the usefulness of source comparisons.

## 10. Ethical Web Scraping and Responsible Data Collection

This project is designed around responsible web collection practices intended to reduce unnecessary server load, respect website restrictions, and limit collection to content relevant to the analysis.

**Public accessibility does not automatically grant permission to scrape, store, reproduce, or redistribute website content.** Each source must be assessed individually before collection.

### 10.1 Respecting `robots.txt`

Before fetching a webpage, the scraper checks the site's `robots.txt` rules using Python's `urllib.robotparser`.

- URLs disallowed by the applicable crawler rules are skipped.
- The configured crawler token, `KeywordThemeBot`, is used when checking the rules.
- When the robots policy cannot be retrieved reliably, the implementation adopts a conservative approach and treats the site as disallowed.
- Applicable crawl-delay directives are considered, subject to the configured maximum delay.

The `robots.txt` file is a mechanism for communicating crawler preferences and restrictions. Checking it does not replace the need to review the site's terms of use or obtain permission where required.

### 10.2 Transparent Crawler Identification

The scraper sends a `User-Agent` header that includes the `KeywordThemeBot` identifier and indicates that the crawler is being used for an educational NLP project.

This helps website operators identify the automated requests and understand their intended purpose.

Crawler identification should support transparency and accountability rather than conceal automated access.

### 10.3 Rate Limiting and Server Load

The scraper uses sequential requests and per-host rate limiting to avoid generating unnecessary traffic.

The implementation includes:

- A configurable minimum delay between requests to the same host.
- Randomized additional waiting time.
- Consideration of the website's crawl-delay directive.
- Configurable maximum page counts and crawl depth.
- Request timeouts and limited retries.
- Backoff delays when handling temporary failures or rate limiting.

When the server returns HTTP 429 (Too Many Requests) or HTTP 503 (Service Unavailable), the scraper waits before retrying, using the `Retry-After` header when a usable value is provided.

These safeguards help reduce server load. They do not guarantee that a website permits the requested activity, and its access policies must still be respected.

### 10.4 No Circumvention of Access Restrictions

The project is intended to collect publicly accessible webpage content without bypassing access controls.

- It does not intentionally access login-protected content.
- It does not use CAPTCHA-solving services.
- It does not attempt to bypass paywalls or authentication requirements.
- Pages returning HTTP 403 (Forbidden) after the implemented retry are skipped.
- Failed requests are recorded rather than treated as reasons to circumvent restrictions.

A website's refusal of automated access must be respected. Technical accessibility is not evidence of authorization.

### 10.5 SSL and Secure Connections

Secure HTTPS connections help protect the integrity and authenticity of retrieved content.

The current scraper includes an optional fallback that retries a request without SSL certificate verification if certificate verification fails. **This fallback weakens connection security and is not recommended for a production deployment.**

For a stronger security posture, disable the insecure SSL fallback and investigate certificate or network configuration problems instead of accepting an unverified connection.

### 10.6 Respecting Terms of Use and Copyright

Before collecting content from a website, users should:

- Review its terms of use and automated-access policies.
- Check whether an official API or downloadable dataset is available.
- Respect copyright and applicable attribution requirements.
- Avoid redistributing complete articles or substantial copyrighted text without authorization.
- Comply with applicable data-protection and other legal requirements.

If a website prohibits the proposed collection, that restriction should be respected even if the scraper can technically access its pages.

### 10.7 Data Minimization and Duplicate Removal

The pipeline collects fields relevant to textual analysis, including page titles, URLs, publication dates when available, and main textual content.

It also applies several measures to reduce irrelevant or redundant data:

- Skipping non-HTML resources and irrelevant file types.
- Discarding pages that contain insufficient usable text.
- Removing duplicate URLs.
- Identifying duplicate normalized text.
- Removing scripts, navigation elements, and common website boilerplate during preprocessing.
- Keeping the collection scope configurable through page limits and crawl-depth settings.

These measures improve the quality of the dataset and limit unnecessary processing. They do not eliminate every possible privacy or copyright concern.

### 10.8 Error Handling and Conservative Behaviour

Websites may block automated requests, become unavailable, or return incomplete content.

The scraper records collection outcomes and handles failures through controlled retries, logging, and skipped pages. Where access restrictions or persistent errors prevent reliable collection, the preferred response is to stop requesting that page rather than attempt to bypass the restriction.

Users should review collection logs and investigate repeated failures before running the scraper again.

### 10.9 Responsible Use of Collected Data

The collected content is intended for educational text analysis, including keyword extraction, topic modelling, source classification, and comparison of aggregate textual patterns.

Users should avoid unnecessary collection of sensitive personal information, unauthorized redistribution of source material, and uses that violate website policies or applicable laws.

Where possible, reports should emphasize derived results, such as keyword rankings, topic summaries, and aggregate statistics, rather than reproducing the original articles.

**Ethical principle:** The ability to retrieve a webpage programmatically does not establish permission to collect or reuse its content. Responsible collection requires transparency, minimal server impact, respect for access restrictions, and appropriate handling of collected text.

## 11. Limitations

- Website structure and access restrictions can affect text extraction.
- Some websites may block automated access or prohibit scraping.
- Publication dates may be missing or unreliable.
- Stopword removal and lemmatization can introduce errors or remove meaningful terms.
- TF-IDF scores do not necessarily identify semantically important keywords.
- Topic models require human interpretation.
- Classification results depend on dataset size, source balance, and feature quality.
- Small datasets may produce unstable or misleading statistical patterns.
- Named-entity recognition depends on the language model and quality of the collected text.
- Temporal analysis is limited by the availability and reliability of publication dates.
- The optional SSL-verification fallback should be disabled for stronger connection security.


## 12. Conclusion

The Keyword & Theme Intelligence System combines web scraping, text preprocessing, exploratory data analysis, TF-IDF keyword extraction, topic modelling, supervised classification, named-entity recognition, temporal analysis, and visualization in a configurable Python pipeline.

It provides a framework for transforming unstructured webpage content into interpretable keywords, themes, source comparisons, and quantitative results.

By combining analytical methods with responsible data-collection practices, the project demonstrates how NLP and machine learning can be applied to real-world textual data.

## 13. Acknowledgements

Developed as part of a data science course project focused on keyword identification, topic discovery, and comparative text analysis.

## 14. License

No license has been specified for this repository yet. 
