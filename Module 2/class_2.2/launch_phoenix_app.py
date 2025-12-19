# monitor_with_phoenix.py
import pandas as pd
import phoenix as px

ref_df = pd.read_parquet("data/mnist_reference.parquet")
prod_df = pd.read_parquet("data/mnist_production.parquet")

# Tell Phoenix which columns contain embeddings, predictions, labels, etc.
train_ds = px.Dataset.from_pandas(
    ref_df,
    embedding_feature_column="embedding",
    prediction_label_column="pred_label",
    actual_label_column="true_label",
)

prod_ds = px.Dataset.from_pandas(
    prod_df,
    embedding_feature_column="embedding",
    prediction_label_column="pred_label",
    actual_label_column="true_label",
)

px.launch_app(train=train_ds, production=prod_ds)