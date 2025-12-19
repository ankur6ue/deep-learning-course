import pandas as pd
import phoenix as px
import os
import phoenix as px
import inspect
import time

# Pick a base port that you’re sure is free (e.g. 6100)
os.environ["PHOENIX_PORT"] = "6100"
# Optional: also set host explicitly
os.environ["PHOENIX_HOST"] = "127.0.0.1"

ref_df = pd.read_parquet("../data/mnist_reference.parquet")
prod_df = pd.read_parquet("../data/mnist_production.parquet")

feature_cols = [c for c in ref_df.columns if c.startswith("embedding")]
# 2. Define the schema (map your column names)
schema = px.Schema(
    prediction_label_column_name="pred_label",
    actual_label_column_name="true_label",
    # If using embeddings, specify them here:
    embedding_feature_column_names={"embedding": px.EmbeddingColumnNames(vector_column_name="embedding")},
    prediction_score_column_name="pred_conf"
)

# 3. Wrap the dataframe in an Inferences object
primary_inferences = px.Inferences(dataframe=prod_df, schema=schema, name="my_dataset")
ref_inferences =  px.Inferences(dataframe=ref_df, schema=schema, name="my_dataset")


# 4. Launch the application
session = px.launch_app(
    primary=primary_inferences,
    reference=ref_inferences,
)

while True:
    time.sleep(100) # Keeps the server running



