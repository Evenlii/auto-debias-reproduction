## Overview

This project extends the **Context-Debias** framework to mitigate biases related to **age** and **disability** in pre-trained language models (e.g., BERT). Our approach employs **orthogonal projection techniques** to reduce biased associations while preserving semantic integrity and downstream task performance. By addressing underexplored dimensions of bias, this work advances the development of fair and equitable NLP systems.

---

## Key Features

- **Extended Bias Mitigation**: Expands the scope of Context-Debias to age and disability biases.
- **Evaluation Framework**: Leverages SEAT (Sentence Encoder Association Test) for bias measurement and GLEU for downstream task assessment.
- **Two-Part Loss Function**:
  - **Debiasing Loss**: Reduces biased associations.
  - **Regularization Loss**: Preserves semantic integrity.

---

## Methodology

1. **Identifying Biases**:
   - Curated word lists representing age-related and disability-related attributes and targets.
   - Attribute examples: "elderly," "teenager."
   - Target examples: "stubborn," "burden."

2. **Debiasing Framework**:
   - Orthogonal projection techniques applied to contextualized embeddings.
   - Layer-wise debiasing to address biases across embedding layers.

3. **Evaluation**:
   - **SEAT Tests**: Measure bias reduction.
   - **GLEU Benchmarks**: Assess downstream task performance.

---

## Results

### Bias Mitigation

- **SEAT-9 (Disability Bias)**: Effect size reduced from 0.51 to 0.04.
- **SEAT-10 (Age Bias)**: Effect size reduced from 0.51 to 0.13.

### Downstream Task Performance

- Maintained GLEU scores across tasks such as sentiment analysis, paraphrase detection, and textual entailment.

## To Reproduce Result

1. Context-Debias code is adapted from this repository of the original paper (https://github.com/kanekomasahiro/context-debias). Run preprocessing and then run debiasing on BERT-based pre-trained model using the word lists (age or disability). Save the model in checkpoint-best. These experiments were executed on a computer with local GPU. 
2. To perform SEAT tests, make sure the model being loaded is from step 1 and then execute run_seat.sh. GPU not required.
3. To perform downstream GLEU tests, run GLEU.ipynb. Since GPU is required, we ran this code on Google Colab. 
