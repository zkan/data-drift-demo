import pandas as pd
from sklearn import datasets

# from evidently import Dataset
# from evidently import DataDefinition
# from evidently import Report
# from evidently.presets import DataDriftPreset, DataSummaryPreset


def main():
    print("Hello from data-drift-demo!")

    adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
    adult = adult_data.frame
    print(adult)


if __name__ == "__main__":
    main()
