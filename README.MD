# Data-centric Sensitivity Analysis (DSA) for Learned Cardinality Estimators

This repository contains the implementation of **Data-centric Sensitivity Analysis (DSA)**, a framework designed to audit the robustness limits of learned cardinality estimators.

## Table of Contents

- [Data-centric Sensitivity Analysis (DSA) for Learned Cardinality Estimators](#data-centric-sensitivity-analysis-dsa-for-learned-cardinality-estimators)
  - [Table of Contents](#table-of-contents)
  - [Directories](#directories)
  - [Abstract](#abstract)
  - [How to get the Dataset](#how-to-get-the-dataset)
- [How to Run the Audit](#how-to-run-the-audit)

## Directories

The repository is organized into several directories for clarity and ease of use:

1. **CorruptDB**: This folder stores the **perturbed database instances** generated during the stress testing process (i.e., the database after applying the worst-case data drifts).
2. **datasets**: This folder stores the original datasets used in the experiments.
3. **query**: This folder contains query workloads used for evaluation: job-light, stats-ceb, and their subqueries.
4. **WeightCache**: This folder is used to cache intermediate weights to accelerate the auditing process.

## Abstract

Learned cardinality estimators have demonstrated superior accuracy in predicting query cardinalities. However, their inherent fragility to data distribution shifts remains a primary barrier to production deployment. Existing robustness evaluations typically rely on random or simplified drift assumptions, failing to bound performance degradation under the complex logic of real-world updates.

In this work, we investigate the **robustness limits** of learned estimators by identifying how minimal, targeted data-level drifts can maximally degrade estimation accuracy. We introduce a **Data-centric Sensitivity Analysis (DSA)** framework to conduct this worst-case audit in a black-box setting. While we establish that finding the optimal worst-case drift is NP-Hard, we design a specialized approximation algorithm that identifies critical data subsets efficiently, achieving a $(1-\kappa)$ approximation ratio. Comprehensive experiments demonstrate that DSA effectively exposes structural vulnerabilities in state-of-the-art estimators, serving as a rigorous stress-testing tool for database reliability.

## How to get the Dataset

We have placed the dataset used in the experiments (including necessary files for the **datasets** and **WeightCache** folders) in the following Google Drive link:
[Download Link](https://drive.google.com/file/d/166lTTU13MlLC3zjRdryHOuV-PP2LPqID/view?usp=sharing)

You can use the following command to download and unzip it:

```bash
wget --no-check-certificate '[https://drive.google.com/uc?export=download&id=166lTTU13MlLC3zjRdryHOuV-PP2LPqID](https://drive.google.com/uc?export=download&id=166lTTU13MlLC3zjRdryHOuV-PP2LPqID)' -O dataset.zip
unzip dataset.zip
