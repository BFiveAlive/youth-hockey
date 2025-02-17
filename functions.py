import json
import requests
import pandas as pd


def parse_rest_api_to_pandas(url):
  """
  Fetches data from a REST API and converts it into a Pandas DataFrame.

  Args:
    url: The URL of the REST API endpoint.

  Returns:
    A Pandas DataFrame containing the data from the API, or None if an error occurs.
  """
  try:
    response = requests.get(url)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    
    # Attempt to parse the JSON response
    try:
      data = response.json()
    except json.JSONDecodeError:
      print("Error: Invalid JSON response received from the API.")
      return None
    
    df = pd.json_normalize(data)
    return df
  
  except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")
    return None