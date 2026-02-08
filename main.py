import pandas as pd
from sklearn import datasets

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset


def main():
    adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
    adult = adult_data.frame

    adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

    # Prod data will include people with education levels unseen in the reference dataset:
    adult_prod = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

    schema = DataDefinition(
        numerical_columns=["education-num", "age", "capital-gain", "hours-per-week", "capital-loss", "fnlwgt"],
        categorical_columns=["education", "occupation", "native-country", "workclass", "marital-status", "relationship", "race", "sex", "class"],
    )

    eval_data_1 = Dataset.from_pandas(
        pd.DataFrame(adult_prod),
        data_definition=schema
    )

    eval_data_2 = Dataset.from_pandas(
        pd.DataFrame(adult_ref),
        data_definition=schema
    )

    report = Report([
        DataDriftPreset() 
    ])

    my_eval = report.run(eval_data_1, eval_data_2)
    my_eval.save_html("evidently_drift_report.html")


if __name__ == "__main__":
    main()
