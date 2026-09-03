# Contextual Representation Extraction via Top-K Logit Lens1. 

### Objective
The objective of this phase is to extract the hidden contextual probability distributions for both individual concepts ($h^A$, $h^B$) and their composed form ($h^{A \land B}$) from a minimal open-weight Large Language Model. These distributions will serve as the empirical basis for computing the categorical limits (pointwise minimum) proposed by the Enriched Category Theory framework.

### Model Selection & Infrastructure
To establish a robust minimal-model baseline, **Qwen/Qwen3.5-2B** is selected (nearest official Qwen3.5 size to the originally planned 1.5B). This scale provides a favorable balance between rapid offline inference capabilities (operable on a single GPU) and sufficient structural complexity to exhibit logical composition. The model and its corresponding tokenizer are instantiated locally using the HuggingFace transformers library.

### Prompt Framing (Control Variables)
To rigorously isolate the semantic composition and eliminate syntactic confounding variables, a unified "Typicality Template" is applied to all instances derived from the generated 5D Orthogonality Corpus. For each concept pair (word_a, word_b), three parallel prompts are constructed:
- $P_A$ (Concept A): "Describe the typical properties and context of a {word_a}:"
- $P_B$ (Concept B): "Describe the typical properties and context of a {word_b}:"
- $P_{AB}$ (Composition): "Describe the typical properties and context of an entity that is both a {word_a} and a {word_b}:"

### Extraction Protocol & Logit Lens Application
Each constructed prompt is passed through the target model via a forward pass. The representations are extracted utilizing a Logit Lens approach specifically at the final token position (the colon :), capturing the model's aggregated contextual expectation. A Softmax function is applied to the final logits to obtain a normalized probability distribution $p(x \mid P)$ over the entire vocabulary space ($V \approx 248{,}320$ for Qwen3.5).


### Top-K Truncation Strategy (Dimensionality Optimization)
Storing the full-vocabulary logit distributions for thousands of concept pairs introduces severe memory bottlenecks (e.g., producing massive data files exceeding 10GB) and includes significant long-tail Softmax noise ($p < 10^{-8}$).To mathematically and computationally optimize this, a Top-K Truncation filter is implemented. Only the highest-probability $K$ tokens (default set to $K=4096$) and their exact probability values are recorded. This structural truncation eliminates noise that could skew the pointwise minimum calculation, heavily compresses the data footprint, and isolates the core semantic manifold.

### Data Persistence
The extracted Top-4096 token IDs, their corresponding probability values, and the structural metadata of the pairs are serialized and saved locally into extracted_distributions_top4096.json. This decoupled approach ensures that the computationally heavy extraction phase is performed once, preserving a lightweight and highly efficient dataset for downstream categorical divergence analysis.