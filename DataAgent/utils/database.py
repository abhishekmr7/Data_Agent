import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


class DatabaseUtil:

    def __init__(self, db_config):
        self.db_config = db_config

        try:
            self.connection = psycopg2.connect(**db_config)
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            self.connection = None

    def schema_details(self, schema_name):
        connection = self.connection
        if connection is None:
            return "Error fetching schema details: no database connection"

        cursor = None
        schema_info_context = f"Database Schema: {schema_name}\n"

        try:
            cursor = connection.cursor()

            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s;",
                (schema_name,),
            )
            tables_list = cursor.fetchall()

            for table in tables_list:
                table_name = table[0]
                schema_info_context += f"\nTable: {table_name}\n"

                # Columns & data types
                cursor.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s;",
                    (schema_name, table_name),
                )
                for column_name, data_type in cursor.fetchall():
                    schema_info_context += f"  Column: {column_name}, Data Type: {data_type}\n"

                # Sample data
                cursor.execute(f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT 5;')
                sample_data = cursor.fetchall()
                schema_info_context += "  Sample Data:\n"
                for row in sample_data:
                    schema_info_context += f"    {row}\n"

        except Exception as e:
            print(f"Error fetching schema details: {e}")
            schema_info_context = f"Error fetching schema details: {e}"

        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

        return schema_info_context

    def execute_sql(self, query):
        connection = self.connection
        if connection is None:
            return "Error executing query: no database connection"

        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            connection.commit()
            return str(result)
        except Exception as e:
            print(f"Error executing query: {e}")
            connection.rollback()
            return f"Error executing query: {e}"
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()


if __name__ == "__main__":
    obj = DatabaseUtil(
        {
            "host": os.environ["host"],
            "port": os.environ.get("port", "5432"),
            "user": os.environ["user"],
            "password": os.environ["password"],
            "dbname": os.environ["database"],
        }
    )

    result = obj.schema_details("public")

    with open("test_schema_details.txt", "w", encoding="utf-8") as f:
        f.write(result)