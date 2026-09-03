# Aim 1: The 5-Dimensional Orthogonality Spectrum Corpus Generation

## 1. Project Context & Theoretical Grounding
This module serves as **Step 1** for the empirical evaluation of Large Language Models (LLMs) through the lens of Enriched Category Theory (ECT). 

To test whether the internal representations of LLMs respect categorical limits (Logical AND / Pointwise Minimum) and colimits (Logical OR / Pointwise Maximum), we must construct a highly controlled dataset of concept pairs: $(h^A, h^B)$. 

Instead of a simple binary text-similarity approach, this dataset maps concepts across a **5-Dimensional Orthogonality Spectrum**—ranging from strict mathematical entailment (subsets) to absolute logical paradoxes (mutually exclusive initial objects).

## 2. The 5-Category Dataset Architecture
We will use an advanced LLM API (SURF AI Hub Pilot) to iteratively generate 500-1,000 concept pairs for each of the following 5 strictly defined categories:

* **Category 1: Redundant / Strict Entailment ($A \subset B$)**
    * *Logic:* Word A must be a specific instance, sub-part, or strictly entailed subset of Word B. 
    * *Expected behavior (min):* Output collapses to Word A.
* **Category 2: Typical Composition (High Overlap / $A \cap B$)**
    * *Logic:* Grounded everyday human life and common visual/contextual co-occurrences.
    * *Expected behavior (min/max):* Smooth fusion state.
* **Category 3: Uncommon / Orthogonal (Cross-Domain Fusion)**
    * *Logic:* Physically possible but highly unusual or historically mismatched domains. Testing structural generalization.
    * *Expected behavior (min):* Weak localized activations.
* **Category 4: Surreal / Out-of-Distribution (Category Errors)**
    * *Logic:* Reifying abstract concepts, anthropomorphizing, breaking semantic common sense. 
    * *Expected behavior:* Measures geometric distance penalty for logical surrealism.
* **Category 5: Mutually Exclusive / Paradoxical ($A \cap B = \emptyset$)**
    * *Logic:* Physical-law inversions and logical NOT-gate contradictions. They must destroy each other conceptually.
    * *Expected behavior (min):* Complete structural collapse toward an "Initial Object" (zero/noise).

## 3. Crucial Prompting Constraints for the API
To prevent the generation API from outputting generic co-occurrences (e.g., generating "Heart and Lungs" when we actually want entailment like "Heart and Cardiovascular System"), the Python script **MUST** dynamically inject specific structural constraints into the System Prompt based on the current Category being generated.

**Meta-Prompt Templates by Category:**
* **Cat 1:** "Word A MUST be a specific instance, sub-part, or strictly entailed subset of Word B within the domain of [{seed}]. Do NOT generate lateral co-occurring items. Example: (Right Triangle, Polygon)."
* **Cat 2:** "Word A and Word B must heavily co-occur in typical daily life within the domain of [{seed}]."
* **Cat 3:** "Word A and Word B must come from two completely different sub-fields within the prompt [{seed}], representing a highly unusual but physically possible combination."
* **Cat 4:** "Word A and Word B must represent a surreal, hallucinatory, or category-error combination based on the domain [{seed}]."
* **Cat 5:** "Word A and Word B must be absolute antonyms or represent a physical/logical impossibility when combined within the domain of [{seed}]. They must destroy each other conceptually."

## 4. Instructions for Cursor (AI Architect)
Hi Cursor, please read this document and implement the `generate_corpus.py` script with the following requirements:

1.  **Hardcode the Dictionary:** Please ask me for the `domain_seeds_map` Python dictionary (containing the 5 categories and their string seeds) and place it at the top of the script.
2.  **Nested Loop Generation:** Implement a double-loop. The outer loop iterates through Categories 1 to 5. The inner loop iterates randomly/sequentially through the domain seeds of that category.
3.  **API Integration:** Connect to the `openai` Python client (configured for the SURF AI Hub base URL). Request **50 pairs per batch** to avoid output token cut-offs.
4.  **Dynamic Prompting:** Construct the prompt dynamically by combining the base task ("Generate 50 unique concept pairs in JSON format: [{'word_a': '...', 'word_b':