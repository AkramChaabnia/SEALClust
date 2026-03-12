# Agent Task Specification: K-Medoids Pre‑Clustering Optimization

## Objective

You are tasked with improving a research repository that implements an
LLM-based clustering pipeline.\
The goal is to **reduce the number of LLM API calls while preserving the
original pipeline logic** by introducing a **K‑Medoids based document
compression step**.

You must implement this change in a **clean, modular, and reproducible
way**.

------------------------------------------------------------------------

# Mandatory Git Workflow

Before making any modifications:

1.  Ensure the repository is clean.
2.  Create and switch to a new feature branch:

``` bash
git checkout -b feature/kmedoids
```

All work must be committed to this branch.

------------------------------------------------------------------------

# Identified Problems in the Current Implementation

The current implementation has three major limitations.

## 1. Excessive Number of LLM Queries

The pipeline requires a **large number of LLM API calls** because every
document is processed individually.

Consequences:

-   Very high computational cost
-   Slow runtime
-   Poor scalability on large datasets

Example:

    3000 documents → 3000 LLM calls

------------------------------------------------------------------------

## 2. Method is Not Fully Unsupervised

The approach relies on **a small set of true labels during an
initialization phase**.

Problems:

-   Cannot be fairly compared with fully unsupervised methods
-   Requires manual intervention
-   Reduces reproducibility

------------------------------------------------------------------------

## 3. Undefined Procedure for Determining Number of Clusters

The paper does not define a **clear algorithmic method to determine k**.

Instead it relies on the LLM to infer cluster structure implicitly,
which makes the method:

-   Hard to reproduce
-   Difficult to benchmark
-   Non‑systematic

------------------------------------------------------------------------

# Proposed Solution

Introduce a **document compression stage using K‑Medoids clustering**
before running the LLM pipeline.

The idea is to **reduce the number of documents that need LLM
processing** by selecting representative documents.

K‑Medoids is preferred because cluster centers are **real documents
(medoids)**.

These medoids can be safely passed to the LLM.

------------------------------------------------------------------------

# New Pipeline Architecture

## Step 1 --- Compute Document Embeddings

Generate embeddings for all documents using the same embedding model
currently used in the project.

Example:

    N = 3000 documents

Output:

    3000 embeddings

------------------------------------------------------------------------

## Step 2 --- Apply K‑Medoids Clustering

Cluster the embeddings using **K‑Medoids** with a relatively large value
of k.

Example:

    k = 100

Result:

    100 clusters
    100 medoid documents

Each medoid represents a cluster of similar documents.

------------------------------------------------------------------------

## Step 3 --- Run the Original LLM Pipeline on Medoids Only

Instead of processing all documents:

    3000 → 100 LLM inputs

The existing pipeline from the paper should run **unchanged** but only
on the **medoid documents**.

------------------------------------------------------------------------

## Step 4 --- Propagate Labels to Cluster Members

Once labels are produced for each medoid:

Assign that label to every document belonging to the same cluster.

Example:

    Cluster 17
    Medoid labeled: "Finance"

    All documents in cluster 17 → label = "Finance"

------------------------------------------------------------------------

# Expected Benefits

## Massive Reduction in LLM Calls

Example:

    Before: 3000 calls
    After: 100 calls

Reduction:

    ~30x fewer API calls

------------------------------------------------------------------------

## Improved Scalability

The system becomes capable of processing **much larger datasets**.

------------------------------------------------------------------------

## Preserves Original Methodology

The LLM reasoning pipeline remains **exactly the same**.

Only a **preprocessing compression step** is introduced.

------------------------------------------------------------------------

# Implementation Requirements

## 1. Embedding Module

Create or reuse a component to compute embeddings for documents.

Possible libraries:

-   sentence-transformers
-   OpenAI embeddings
-   existing project embedding module

------------------------------------------------------------------------

## 2. K‑Medoids Clustering Module

Implement clustering using:

-   `sklearn_extra.cluster.KMedoids`\
    or
-   another stable implementation

Inputs:

    embeddings
    k

Outputs:

    cluster assignments
    medoid indices

------------------------------------------------------------------------

## 3. Medoid Extraction

Create a function:

    get_medoid_documents(documents, medoid_indices)

Return the representative documents.

------------------------------------------------------------------------

## 4. Pipeline Integration

Modify the pipeline to support the following flow:

    documents
        ↓
    embedding generation
        ↓
    k‑medoids clustering
        ↓
    medoid document selection
        ↓
    LLM pipeline (existing logic)
        ↓
    label propagation
        ↓
    final labeled dataset

------------------------------------------------------------------------

## 5. Label Propagation Module

Implement a mapping:

    cluster_id → label

Then apply:

    document_label = cluster_label

------------------------------------------------------------------------

## 6. Configuration Parameter

Add configuration options:

    kmedoids_enabled = true
    kmedoids_k = 100

This allows toggling the feature.

------------------------------------------------------------------------

# Code Quality Requirements

-   Maintain existing architecture
-   Avoid breaking current functionality
-   Add clear docstrings
-   Ensure modular design
-   Follow repository style conventions

------------------------------------------------------------------------

# Validation

Test the pipeline using a small dataset.

Verify:

1.  Medoids are actual documents
2.  Only medoids are sent to the LLM
3.  Labels propagate correctly
4.  Output format remains unchanged

------------------------------------------------------------------------

# Deliverables

The final branch `feature/kmedoids` must include:

-   K‑Medoids preprocessing module
-   Updated pipeline integration
-   Configuration options
-   Documentation comments

Optional but recommended:

-   Benchmark showing reduction in LLM calls
-   Example run script
