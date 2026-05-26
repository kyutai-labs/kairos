# OLMES Evaluation Implementation

This directory implements the OLMES (Open Language Model Evaluation Standard) benchmark.
https://arxiv.org/abs/2406.08446

## Overview

OLMES standardizes evaluation across 10 multiple choice question answering benchmarks with consistent prompting, few-shot configuration, and probability normalization methods.


| Task | Split | Normalization (CF) |
|------|------|-------------------|
| `arc_challenge` | test | PMI |
| `arc_easy` | test | Character |
| `boolq` | validation | None |
| `csqa` | validation | PMI |
| `hellaswag` | validation | Character |
| `mmlu` | test | Character |
| `obqa` | test | PMI |
| `piqa` | validation | Character |
| `siqa` | validation | Character |
| `winogrande` | validation | None |
