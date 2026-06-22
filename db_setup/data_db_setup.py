import os
import sqlite3
import pandas as pd

DB_NAME = "data.db"
FILES = os.listdir("./data")  # or your directory listing
directory = "./data"


def pandas_to_sqlite_dtype(dtype):
    dtype_str = str(dtype)

    if "int" in dtype_str:
        return "INTEGER"
    elif "float" in dtype_str:
        return "REAL"
    elif "bool" in dtype_str:
        return "INTEGER"
    elif "datetime" in dtype_str:
        return "TEXT"
    else:
        return "TEXT"


def initialize_database():
    connection = None

    try:
        connection = sqlite3.connect(DB_NAME)
        cursor = connection.cursor()

        files = [
            f
            for f in FILES
            if os.path.isfile(os.path.join(directory, f))
        ]

        for item in files:

            if not item.endswith(".csv"):
                continue

            print(f"Processing {item}")

            csv_path = os.path.join(directory, item)

            # Read CSV
            df = pd.read_csv(csv_path)

            # Table name = filename without extension
            table_name = os.path.splitext(item)[0]

            # Build CREATE TABLE statement
            column_defs = []

            for col in df.columns:
                sqlite_type = pandas_to_sqlite_dtype(df[col].dtype)

                # Escape column names
                column_defs.append(
                    f'"{col}" {sqlite_type}'
                )

            create_sql = f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                {", ".join(column_defs)}
            );
            """

            cursor.execute(create_sql)

            # Load data
            df.to_sql(
                table_name,
                connection,
                if_exists="append",
                index=False
            )

            print(f"Created table {table_name}")

        connection.commit()

        print(
            f"Database '{DB_NAME}' initialized successfully!"
        )

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if connection:
            connection.close()

if __name__ == '__main__':
    initialize_database()