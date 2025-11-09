# Distributed ML Pipeline for Large-Scale Song Data
This project implements a distributed machine learning pipeline using PySpark on Databricks, built to process and model over 1 million song records from the Million Song Dataset (MSD).
The pipeline demonstrates scalable feature engineering, distributed model training, and performance optimization in a cloud-based environment.

## Overview
The goal of this project is to predict song similarity or user preference trends using large-scale audio metadata.
By leveraging AWS EMR and Databricks notebooks, the workflow handles both data preprocessing and model training efficiently across distributed Spark clusters.

Key features include:

- Scalable ETL (extract-transform-load) with PySpark DataFrames
- Feature extraction with n-gram tokenization and TF-IDF vectorization
- Dimensionality reduction to optimize model input
- Distributed model evaluation using cross-validation
- Achieved +13% AUC improvement over baseline via hyperparameter tuning


## Dataset
- Million Song Dataset 
- Source: https://labrosa.ee.columbia.edu/millionsong/
- Records: ~1 million songs
- Format: JSON / HDF5 (transformed to Parquet for Spark processing)

**Key features**:

- Track metadata (title, artist, year, duration, loudness)
- Audio features (tempo, key, mode, energy, etc.)
- Lyrics and tags (used for text-based features)


## Architecture

                ┌────────────────────┐
                │   AWS S3 Storage    │
                │ (Million Song Data) │
                └─────────┬───────────┘
                          │
                 ┌────────▼────────┐
                 │  Databricks     │
                 │ (PySpark ETL)   │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ Feature Pipeline│
                 │ TF-IDF + N-Gram │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  ML Modeling    │
                 │ (LogReg, RF)    │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │ Cross-Validation│
                 │ + Hyperparam Opt│
                 └─────────────────┘


## Pipeline Components
1. Data Loading

- Load raw song data (JSON or Parquet) into a distributed Spark DataFrame.
- Clean and normalize text fields, handle missing values.

```bash
songs_df = spark.read.parquet("s3://song-data/msd_cleaned.parquet")
songs_df = songs_df.dropna(subset=["lyrics", "tags"])
```

2. Feature Engineering

- Text preprocessing: Tokenization, stopword removal, stemming
- TF-IDF vectorization with configurable n-gram range (1–3)
- Dimensionality reduction using PCA or TruncatedSVD.
  
```bash
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2**18)
idf = IDF(inputCol="rawFeatures", outputCol="features")
```

3. Modeling

- Baseline: Logistic Regression
- Enhanced: Gradient-Boosted Trees or Random Forests with tuned hyperparameters
- Evaluation: AUC, Precision, Recall, and RMSE

```bash
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=20, regParam=0.1)
model = lr.fit(training_data)
```

4. Hyperparameter Optimization

- Conducted grid search with cross-validation across distributed executors.

```bash
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 1.0]).build()
crossval = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
```

## Results

| Model                          | AUC      | RMSE  | Improvement  |
| ------------------------------ | -------- | ----- | ------------ |
| Baseline (Logistic Regression) | 0.74     | 16.09 | —            |
| Optimized Model (TF-IDF + PCA) | **0.84** | 13.97 | **+13% AUC** |


## Technologies Used

| Category                 | Tools                                                        |
| ------------------------ | ------------------------------------------------------------ |
| Data Processing          | **PySpark**, **Databricks**, **AWS EMR**                     |
| ML / Feature Engineering | **TF-IDF**, **n-gram models**, **PCA**, **Cross-Validation** |
| Cloud Infrastructure     | **AWS S3**, **Databricks Notebooks**                         |
| Data                     | **Million Song Dataset (MSD)**                               |

## Author
Sunny Lee
Carnegie Mellon University — Distributed Systems & Data Engineering
📧 hsiangyl@andrew.cmu.edu
