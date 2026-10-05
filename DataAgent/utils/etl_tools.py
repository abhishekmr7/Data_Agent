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


    s



if __name__ == "__main__":
    obj = ETLTools()
    print(obj.extract_load("https://pokeapi.co/api/v2/pokemon/","data/extract","csv"))

            
       

