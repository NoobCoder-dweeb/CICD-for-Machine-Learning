import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import skops.io as sio
import os 
import sys
from typing import List, Any, Callable

def evaluate(pipe, X_test, y_test, folder: str) -> None:
    predictions = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)   
    f1 = f1_score(y_test, predictions, average="weighted")

    with open(os.path.join(folder, "metrics.txt"), "w") as f:
        f.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

    cm = confusion_matrix(y_true=y_test, y_pred=predictions, labels=pipe.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)

    disp.plot()
    plt.savefig(os.path.join(folder, "confusion_matrix.png"), dpi=120)
    plt.close()
    

def save_model(pipe, folder) -> None:
    sio.dump(pipe, os.path.join(folder, "drug_pipeline.skops"))

def create_pipeline(cat_cols: List[int], num_cols: List[int]) -> Any:
    transform = ColumnTransformer(transformers=[
        ("encode", OrdinalEncoder(), cat_cols),
        ("impute", SimpleImputer(strategy="median"), num_cols),
        ("scale", StandardScaler(), num_cols),
    ])

    return Pipeline(steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125))
    ])

def train():
    data_folder = os.path.join(os.path.dirname(__file__), "Data")
    model_folder = os.path.join(os.path.dirname(__file__), "Models")
    result_folder = os.path.join(os.path.dirname(__file__), "Results")

    assert os.path.exists(data_folder), f"Data folder does not exist at {data_folder}"
    assert os.path.exists(model_folder), f"Model folder does not exist at {model_folder}"
    assert os.path.exists(result_folder), f"Result folder does not exist at {result_folder}"

    file_name = sys.argv[1]
    file_path = os.path.join(data_folder, file_name)
    assert os.path.exists(file_path), f"Data file does not exist at {file_path}"

    drug_df = pd.read_csv(file_path)

    X = drug_df.drop(columns=["Drug"]).values
    y = drug_df.Drug.values

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=125)
    
    cat_cols = [1,2,3]
    num_cols = [0,4]

    pipe = create_pipeline(cat_cols, num_cols)

    # train the model
    pipe.fit(X_train, y_train)

    evaluate(pipe, X_test, y_test, result_folder)

    save_model(pipe, model_folder)

if __name__ == "__main__":
    train()
    