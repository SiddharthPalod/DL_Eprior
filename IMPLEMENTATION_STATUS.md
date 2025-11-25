# Implementation Status: Pre-CoCaD Pipeline

This document analyzes what has been implemented from the plan up to just before the CoCaD step (Section 3.3.2).

## Overview

The current implementation covers **Section 3.2** (Initial Extraction) and **Section 3.3.1** (ACE - Active Candidate-Set Expansion), with varying levels of completeness.

---

## ✅ **FULLY IMPLEMENTED**

### 1. Section 3.2: Learning Correlational Matrices A_W

**Status: ✅ COMPLETE**

#### 1.1 Initial Extraction and Indexing
- ✅ **Node Extraction** (`src/node_extractor.py`):
  - Uses spaCy NER for entity extraction
  - Extracts noun chunks as additional nodes
  - Tracks occurrences with `(chunk_id, sentence_id)` tuples
  - Builds `T_map` equivalent (occurrences stored in `NodeRecord`)

- ✅ **Sentence Extraction** (`src/node_extractor.py`):
  - Segments text into sentences using spaCy
  - Creates `SentenceRecord` with text, chunk_id, sentence_id
  - Handles paragraph-level structure

- ✅ **Data Loading** (`src/data_loader.py`):
  - Loads pre-computed PDF embeddings from Excel files
  - Extracts chunks with metadata (source, page_number, section_title)

#### 1.2 Graph Building
- ✅ **Co-occurrence Graph** (`src/graph_builder.py`):
  - Builds paragraph-level co-occurrence graph
  - Creates sparse adjacency matrix (COO format)
  - Tracks which nodes appear in which paragraphs
  - Implements `A_co-occur` as binary symmetric graph

**Note**: The plan mentions K weighted relational graphs `A_W = {W_1, ..., W_K}`, but the current implementation only builds a single co-occurrence graph. This is a simplification but sufficient for the ACE pipeline.

---

### 2. Section 3.3.1: ACE - Structural Filter (GAE)

**Status: ✅ COMPLETE**

#### 2.1 Graph Autoencoder Architecture
- ✅ **Encoder** (`src/structural_filter.py`):
  - 2-layer GCN encoder (as specified in plan)
  - Layer 1: `H^(1) = ReLU(Â·X·W_0)` with hidden_dim
  - Layer 2: `Z = Â·H^(1)·W_1` with latent_dim
  - Symmetric normalization: `D^(-1/2) Â D^(-1/2)`

- ✅ **Decoder**:
  - Dot product decoder: `Ŝ_GAE = σ(Z·Z^T)`
  - Binary cross-entropy loss for reconstruction

- ✅ **Training**:
  - Samples positive edges from `A_co-occur`
  - Samples negative edges (random, same count as positives)
  - Trains with Adam optimizer
  - Configurable epochs, learning rate, dimensions

#### 2.2 Candidate Generation
- ✅ **ANN-based Expansion** (`src/structural_filter.py`):
  - Uses FAISS IndexFlatIP for approximate nearest neighbors
  - For each node, finds top-k neighbors in latent space
  - Creates `CandidatePair` objects with structural scores
  - Linear complexity: O(N·k') instead of O(N²)

**Output**: List of structurally plausible candidate pairs `C_1` (Candidate Set 1)

---

## ⚠️ **PARTIALLY IMPLEMENTED**

### 3. Section 3.3.1: ACE - Semantic Filter

**Status: ⚠️ PARTIAL (Simplified Implementation)**

#### 3.1 What's Implemented

- ✅ **Sentence Vector Store** (`src/vector_store.py`):
  - FAISS-based ANN index for sentence embeddings
  - Uses sentence-transformers (all-MiniLM-L6-v2)
  - Efficient k-NN search

- ✅ **Hypothesis Generation** (`src/semantic_filter.py`):
  - **SIMPLIFIED**: Uses template-based hypotheses instead of LLM-generated
  - Templates: "X leads to Y", "X affects Y", "relationship between X and Y", "Y depends on X"
  - ❌ **MISSING**: LLM-based hypothetical document generation (RAG-HyDE)

- ✅ **Adaptive Pooling** (`src/semantic_filter.py`):
  - Dynamic pool size: `k_pool = k_base + (1 - score) * k_expansion`
  - Uses base similarity score to adjust pool size

- ✅ **MMR Reranking** (`src/semantic_filter.py`):
  - Maximal Marginal Relevance implementation
  - Balances relevance vs diversity
  - Configurable lambda parameter

- ✅ **Reciprocal Rank Fusion (RRF)** (`src/semantic_filter.py`):
  - Combines ranked lists from multiple hypotheses
  - Uses formula: `RRFScore(d) = Σ 1/(κ + rank_h(d))`
  - Configurable kappa parameter

- ✅ **Scoring** (`src/semantic_filter.py`):
  - Support score: Mean similarity of retrieved contexts
  - Temporal score: Ratio of contexts containing temporal markers
  - Mechanistic score: Ratio of contexts containing mechanistic markers
  - Optional LLM verification via Gemini (when `--use-llm` flag is set)

#### 3.2 What's Missing

- ❌ **RAG-HyDE (Hypothetical Document Embeddings)**:
  - Plan specifies: LLM generates k hypothetical sentences describing causal relationships
  - Current: Only uses template-based hypotheses
  - **Impact**: Lower quality hypothesis generation, may miss nuanced relationships

- ❌ **Hypothesis Verification via RAG + Semantic Entropy**:
  - Plan specifies multi-step verification:
    1. Evidence retrieval via RAG for each hypothesis
    2. LLM self-check verification (`f_verify`)
    3. Semantic entropy estimation (multiple stochastic forward passes)
    4. Dual filtering: `p_support > τ_support AND H_semantic < τ_entropy`
  - Current: No verification step, hypotheses are used directly
  - **Impact**: Risk of hallucination drift, unverified hypotheses may contaminate results

- ❌ **Semantic Entropy Estimation**:
  - Plan specifies: `H_semantic(h_l) = -Σ p_r log p_r` across R independent samples
  - Temperature-scaled sampling for uncertainty estimation
  - Current: No entropy calculation
  - **Impact**: Cannot detect model uncertainty or semantic disagreement

- ❌ **CPC (Causal Plausibility Classifier)**:
  - Plan specifies: Offline training of DeBERTa-v3 cross-encoder with 3 heads:
    - Plausible head
    - Temporal head
    - Mechanistic head
  - Isotonic regression calibration for each head
  - Multi-task BCE loss
  - Current: Uses heuristic marker-based scoring instead
  - **Impact**: Less accurate filtering, no calibrated probabilities

- ❌ **CPC Dataset Construction**:
  - Plan specifies: Large-scale dataset construction with:
    - Teacher LLM labeling
    - Adversarial hard negatives
    - Balanced dataset (50% positive, 25% easy negative, 25% hard negative)
  - Current: No dataset construction pipeline
  - **Impact**: Cannot train CPC model

---

## 📊 **Implementation Summary**

### Completed Components
1. ✅ Node and sentence extraction
2. ✅ Co-occurrence graph building
3. ✅ Structural filter (GAE) - **FULLY COMPLETE**
4. ✅ Basic semantic filtering with MMR and RRF
5. ✅ Heuristic-based scoring (temporal/mechanistic markers)
6. ✅ Optional LLM verification (Gemini) for final scoring

### Missing/Incomplete Components
1. ❌ LLM-based hypothesis generation (RAG-HyDE)
2. ❌ Hypothesis verification with semantic entropy
3. ❌ CPC model training pipeline
4. ❌ CPC dataset construction
5. ❌ Isotonic regression calibration
6. ❌ Full RAV (Retrieval-Augmented Verification) pipeline

---

## 🎯 **Current Output**

The pipeline currently produces `E_prior.json` containing:
- Node pairs (i, j) with labels
- Support, temporal, and mechanistic scores
- Retrieved context snippets
- Optional LLM rationale (if `--use-llm` is enabled)

**This output is ready for CoCaD**, but with lower quality than the full plan specification due to:
- Simplified hypothesis generation
- Missing verification steps
- Heuristic scoring instead of trained CPC

---

## 📝 **Recommendations**

To fully implement the plan before CoCaD:

1. **Priority 1**: Implement RAG-HyDE hypothesis generation
   - Add LLM call to generate k hypothetical sentences per pair
   - Store generated hypotheses for verification

2. **Priority 2**: Implement hypothesis verification
   - Add RAG evidence retrieval for each hypothesis
   - Implement LLM self-check verification
   - Add semantic entropy estimation

3. **Priority 3**: Implement CPC training pipeline
   - Build dataset construction script
   - Train DeBERTa-v3 cross-encoder
   - Add isotonic regression calibration
   - Replace heuristic scoring with CPC predictions

---

## 🔍 **Code References**

- **Structural Filter**: `src/structural_filter.py` (Lines 1-150)
- **Semantic Filter**: `src/semantic_filter.py` (Lines 1-171)
- **Node Extraction**: `src/node_extractor.py` (Lines 1-89)
- **Graph Building**: `src/graph_builder.py` (Lines 1-47)
- **Vector Store**: `src/vector_store.py` (Lines 1-45)
- **LLM Verifier**: `src/llm_verifier.py` (Lines 1-168)
- **Pipeline**: `src/pipeline.py` (Lines 1-249)

---

**Last Updated**: Based on codebase analysis as of current date
**Plan Reference**: Sections 3.2 and 3.3.1 (up to line 1048, just before CoCaD)

