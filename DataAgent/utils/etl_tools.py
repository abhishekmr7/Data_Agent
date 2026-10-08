import os
import requests
import pandas as pd
class ETLTools:
    def __init__(self):
        pass

    def extract_load(self,url:str,output_folder:str,format:str):
        """This tool Extracts the data from API(URL) and loads it into desired location(output_folder).
        Args:
            url (str): API URL to extract data from
            output_folder (str): Location to load the extracted
        returns:
            str: Message indicating the success of the operation or failure
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_folder = os.path.join(project_root, output_folder)

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()
            filename = os.path.join(output_folder, f"Extracted_data.{format}")
            os.makedirs(output_folder, exist_ok=True)

            df = pd.json_normalize(data['results'])
            if format == 'csv':
                df.to_csv(filename, index=False)
            elif format == 'json':
                df.to_json(filename, orient='records', lines=True)
            elif format == 'parquet':
                df.to_parquet(filename, index=False)
            else:
                return f"Unsupported format:{format}"

            return f"Data Successfully extracted and saved to {filename}"
        except requests.exceptions.RequestException as e:
            return f"Failed to extract data {e}"


    def transform_Load(self,file_path:str):
        """This tool Transforms the data from file_path and loads it into desired location(output_folder).
        Args:
            file_path (str): Location of the file to transform
            output_folder (str): Location to load the transformed data
            format (str): Format to save the transformed data (csv, json, parquet)
        Returns:
            str: Message indicating the success of the operation or failure
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)   
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file format: {file_extension}"

        top3_rows = str(df.head(3))

        return f"Data Successfully transformed. Top 3 rows:\n{top3_rows}"


    def execute_function(self,code:str):
        """This tool executes the function provided in the code string.
        Args:
            code (str): Code string containing the function to execute  
        Returns:
            str: Message indicating the success of the operation or failure
        """
        try:
            exec(code)
            return "Function executed successfully."
        except Exception as e:
            return f"Failed to execute function: {e}"

if __name__ == "__main__":
    obj = ETLTools()
    path = "C:\\Ai-Projects\\DataAgent\\data\\extract\\Extracted_data.csv"

    print(obj.transform_Load(path))