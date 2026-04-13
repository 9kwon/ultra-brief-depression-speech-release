# ultra-brief-depression-speech-release

Public release for the manuscript:
**An Ultra-Brief Acoustic–Lexical Framework for Detecting Depressive Symptoms From Five Daily Spoken Words in Active-Duty Police Officers**

This repository contains a **public, manuscript-focused release** of the analysis pipeline.
It is organized to support reproduction of the main reported analyses while avoiding release of the full curated Korean lexical resource.

## What is included

- `src/acoustic_feature_extraction.py`
  - candidate acoustic feature extraction from sliced WAV files
  - diary and start-cue outputs are saved separately
- `src/main_analysis.py`
  - manuscript main analyses only
  - binary healthy vs. depressive-symptom classification
  - demographic baseline
  - temporal analysis
  - nested stability-selected reduced model
- `resources/lexical_category_schema.json`
  - public lexical category schema used in the main manuscript
  - category-level mapping only
- `requirements.txt`
  - Python package list

## What is not included

This public release does **not** include the full curated Korean lexical resource used during the study.
Specifically omitted:

- the full Korean lexicon used for rule-based word-to-subdomain assignment
- occupation-specific vocabulary lists
- misspelling normalization dictionary
- iterative lexicon expansion artifacts

Instead, this repository provides the **category schema** used for manuscript-level lexical features.

## Repository structure

```text
ultra-brief-depression-speech-release/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── example_commands.txt
├── resources/
│   └── lexical_category_schema.json
└── src/
    ├── acoustic_feature_extraction.py
    └── main_analysis.py
```

## Acoustic pipeline note

`src/acoustic_feature_extraction.py` extracts a broader set of **candidate** acoustic descriptors.
The manuscript main model uses a retained subset of these descriptors after feature screening for robustness and interpretability in ultra-brief single-word utterances.

The start cue (`start_recording`) is exported separately for inspection and quality control.
It was **not** included in the final manuscript acoustic model.

## Main lexical features used in the manuscript

The public schema exposes the category structure for the nine lexical features used in the main model:

- `lex_pos_affect`
- `lex_neg_dep`
- `lex_neg_stress`
- `lex_wellness`
- `lex_daily_burden`
- `lex_self_focus`
- `domain_entropy`
- `global_sub_entropy`
- `lag1_word_jaccard_mean`

## Installation

```bash
pip install -r requirements.txt
```

## Example usage

### 1. Extract candidate acoustic features

```bash
python src/acoustic_feature_extraction.py   --target-folder /path/to/sound   --save-diary-csv outputs/diary_features.csv   --save-start-csv outputs/start_recording_features.csv
```

### 2. Run the manuscript main analyses

```bash
python src/main_analysis.py   --diary-csv outputs/diary_features.csv   --vocab-csv /path/to/embedded_words.csv   --survey-csv /path/to/total_basic_survey_final.csv   --out-dir results/main_release
```

Optional:

```bash
python src/main_analysis.py   --diary-csv outputs/diary_features.csv   --vocab-csv /path/to/embedded_words.csv   --survey-csv /path/to/total_basic_survey_final.csv   --out-dir results/main_release   --skip-temporal
```

## Scope of the public release

The default execution path in `src/main_analysis.py` is intentionally limited to the manuscript core analyses.
Exploratory or defense-oriented analyses are not part of the default public release path.

## Citation

If you use this repository, please cite the associated manuscript.
