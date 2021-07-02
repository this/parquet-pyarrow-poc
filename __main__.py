import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal


def create_dataset() -> pd.DataFrame:
    data = {
        "course_id": ["001", "002", "003", "004", "005"],
        "course_name": ["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology"],
        "students": [
            ["Alice", "Bob", "Sam"],
            ["Harry", "Ricky"],
            ["George", "Kathy", "Megan"],
            ["Will", "Jhon", "Peter", "Ryan"],
            ["Bill"]
        ]
    }
    return pd.DataFrame(data)


def save_dataset(df, output_file_name) -> None:
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file_name)


def read_dataset(file_name) -> pd.DataFrame:
    table = pq.ParquetFile(file_name).read()
    return table.to_pandas()


def main() -> None:
    original_df = create_dataset()
    save_dataset(original_df, "output.parquet")
    new_df = read_dataset("output.parquet")
    print("Original Dataset\n", original_df)
    print("New Dataset\n", new_df)
    assert_frame_equal(original_df, new_df)


if __name__ == '__main__':
    main()
