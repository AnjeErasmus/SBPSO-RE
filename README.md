[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19479292.svg)](https://doi.org/10.5281/zenodo.19479292)

# SBPSO-RE: Rule Extraction from Random Forests using Particle Swarm Optimization

## Overview
This repository presents **SBPSO-RE**, a rule extraction framework designed to improve the interpretability of Random Forest (RF) models. The method leverages a **Set-Based Particle Swarm Optimization (SBPSO)** approach within a Separate-and-Conquer (SeCo) strategy to generate compact, human-readable IF–THEN rule sets.

The extracted rules aim to balance:
- **Predictive performance** (accuracy, macro-F1)
- **Fidelity** (agreement with RF predictions)
- **Interpretability** (ruleset size and rule length)

This work was developed as part of research in **Explainable Artificial Intelligence (XAI)** and is associated with submission to **IEEE CEC**.

---

## Repository Structure


```
project/
│
├── README.md
├── requirements.txt
│
└── src/
    ├── main.py                # Example entry point
    ├── objective.py           # Hyperparameter tuning (Optuna)
    │
    ├── models/
    │   └── sbpso.py           # Core SBPSO + SeCo implementation
    │
    └── utils/
        └── utility.py         # Supporting functions
```

---

## Methodology
The SBPSO-RE framework consists of the following key steps:

1. **Candidate Condition Generation**
   - A universe of conditions \( U \) is generated from random forest tree structures.

2. **Rule Induction (SBPSO + SeCo)**
   - Particle Swarm Optimization is used to search for high-quality rules.
   - A Separate-and-Conquer strategy iteratively builds a ruleset.

3. **Rule Evaluation**
   - Rules are evaluated using accuracy, macro-F1, and fidelity.

4. **Rule Pruning**
   - Redundant or low-contribution rules are removed to improve interpretability.

---

## Installation
Install required dependencies:


pip install -r requirements.txt

---

## Key Features
- Interpretable IF–THEN rule extraction
- Optimization-driven rule induction (PSO-based)
- Supports pruning for compact rule sets
- Designed for benchmarking against methods such as RuleFit and SIRUS

---

## Experimental Context
The framework is designed to operate within a cross-validation setting using pre-defined splits and evaluates:

- Accuracy  
- Macro F1-score  
- Fidelity to Random Forest predictions  
- Ruleset size and average rule length  
- Runtime performance  

---

## Author
**Anje Erasmus**  
MEng Industrial Engineering (Data Science)  
Stellenbosch University  
## CoAuthor
**Andries Engelbrecht*
