
# Black-box Algorithmic Complexity Attack against all learned estimators

## Table of Contents

- [Black-box Algorithmic Complexity Attack against all learned estimators](#black-box-algorithmic-complexity-attack-against-all-learned-estimators)
  - [Table of Contents](#table-of-contents)
  - [Directories](#directories)
  - [Abstract](#abstract)
  - [How to get the Dataset](#how-to-get-the-dataset)
- [How to run experiment:](#how-to-run-experiment)

## Directories

The repository is organized into several directories for clarity and ease of use:

1. **CorruptDB**: This folder contains the database after it has been compromised.
2. **datasets**: This folder stores the datasets used in the experiments.
3. **query**: This folder contains four sets of queries: job-light, stats-ceb, and their subqueries.
4. **WeightCache**: This folder is used to store the cache for weights.

## Abstract

Learned cardinality estimators have great potential in predicting query cardinalities. However, these learned methods, without exception, demonstrate fragility to drifts within the training data, thereby posing significant risks for real-world deployment. In this work, we are the first to investigate how minimal data-level drifts can maximally compromise the accuracy of learned estimators. To this end, we introduce the data-centric algorithmic complexity attack against all learned estimators in a black-box setting. We establish that finding the optimal data-centric attack strategy is NP-Hard. Thus, we designed a specialized approximation algorithm that achieves near-optimal solutions within polynomial-time, offering a $(1-\kappa)$ approximation ratio. Comprehensive experiments validate the effectiveness of our approach


## How to get the Dataset

We have placed the dataset used in the experiment(**datasets**/**WeightCache** folder  ) in the following Google Drive link:
https://drive.google.com/file/d/166lTTU13MlLC3zjRdryHOuV-PP2LPqID/view?usp=sharing  

You can use ` wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=166lTTU13MlLC3zjRdryHOuV-PP2LPqID' -O dataset.zip` command to download it, and use `unzip dataset.zip` command to unzip it.

# How to run experiment:

First:

`mkdir ./CorruptDB/`

Assuming you need to launch an attack on the STATS-CEB database, targeting 20% of the budget for the 'post' table (approximately 1.2% of the entire STATS database), you can simply run the following command with one click  :)

`python ./DACA.py --dataset=stats_simplified --table=posts.csv --primaryKey=Id --query=stats_CEB.sql`


Assuming you need to launch an attack on the IMDB-JOB database, targeting 20% of the budget for the 'post' table (approximately 0.8% of the entire IMDB database), you can simply run the following command with one click  :)

`python ./DACA.py --dataset=job --table=title.csv --primaryKey=id --query=job_light_queries.sql`

Next, you can find the corresponding compromised database in the CorruptDB folder.