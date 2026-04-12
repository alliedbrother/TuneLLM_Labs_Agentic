---
name: data-quality
description: Data quality assessment, filtering, deduplication, and contamination checking for training datasets
---

# Data Quality Skill

Assess and enforce quality standards on training datasets. This skill defines quality metrics, filtering criteria, and contamination detection protocols.

## Quality Dimensions

### Correctness
- Verify that expected outputs are factually correct
- For code: outputs must compile/run and produce expected results
- For math: solutions must arrive at the correct answer with valid reasoning
- For factual: claims must be verifiable against reliable sources

### Format Compliance
- Input/output pairs match the fine-tuning format specification
- No truncated, malformed, or empty fields
- Consistent tokenization-friendly formatting

### Diversity
- N-gram diversity scores (unigram, bigram, trigram)
- Topic distribution coverage
- Structural variety (question types, response lengths, complexity levels)

### Deduplication
- Exact match deduplication
- Fuzzy deduplication (MinHash/LSH at configurable similarity threshold)
- Cross-dataset dedup against previous training sets

## Contamination Detection

- Maintain a hash set of all evaluation benchmark examples
- Check every training candidate against the eval set
- Flag and remove any matches (exact or near-match)
- Report contamination rate to the Evaluation Agent

## Quality Scoring

Each data sample receives a composite quality score (0.0 - 1.0):

```
quality_score = w1 * correctness + w2 * relevance + w3 * complexity + w4 * clarity
```

Default weights: correctness=0.4, relevance=0.3, complexity=0.15, clarity=0.15

Minimum quality threshold: 0.6 (configurable per improvement cycle)

## Dataset Report Format

```yaml
dataset_version: ds-v1.2.0-math
total_candidates: 50000
after_dedup: 42000
after_quality_filter: 35000
final_selected: 30000
contamination_removed: 12
quality_distribution:
  mean: 0.78
  p25: 0.68
  p50: 0.79
  p75: 0.88
source_breakdown:
  - source: feedback_logs
    count: 15000
  - source: curated_corpus
    count: 12000
  - source: synthetic
    count: 3000
```
