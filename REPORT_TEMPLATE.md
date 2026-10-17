# Robustness Analysis Report

## 1. Research question

State the question and the hypotheses before looking at the final results.

## 2. Models

Record exact model IDs, library versions, device, input handling, and whether weights were pretrained.

## 3. Dataset

Describe image sources, licenses, inclusion criteria, scene distribution, exclusions, and final sample counts. Separate the private test set from any public benchmark.

## 4. Perturbations

Copy the committed configuration and explain why each intensity is realistic. State the random seed.

## 5. Metrics

Explain which metrics measure stability without ground truth. If a public benchmark is used, document its official protocol separately.

## 6. Results

Include per-condition sample counts, mean, standard deviation, curves, representative cases, and worst cases. Do not report more precision than the experiment supports.

## 7. Failure analysis

Inspect fine objects, object boundaries, reflections, transparent surfaces, low light, overexposure, and scene changes. Distinguish plausible depth changes from obvious failures.

## 8. Limitations and threats to validity

Discuss missing ground truth, synthetic-versus-real corruptions, dataset size, model training-data overlap, metric sensitivity, and hardware/runtime constraints.

## 9. Reproducibility

Provide environment setup, exact commands, configuration, seed, commit hash, and links to permitted data or data-collection instructions.

## 10. Conclusion and next experiment

Answer only what the evidence supports. Propose one focused next experiment, such as low-light enhancement evaluated on the same fixed set.
