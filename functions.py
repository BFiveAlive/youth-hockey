import json
import requests
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt


# Function to fetch data from a REST API and convert it into a Pandas DataFrame
def parse_rest_api_to_pandas(url):
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
  

def model_charts(df):
  # PREDICTED GOALS
  out_chart = df[(df['keep'] == '1')]
  slope, intercept, rvalue, pvalue, stderr = stats.linregress(out_chart['pred_diff'], out_chart['goal_diff'])

  fig1, ax1 = plt.subplots()
  ax1.scatter(out_chart['pred_diff'], out_chart['goal_diff'], s=10, alpha=1)
  ax1.plot(out_chart['pred_diff'], intercept + slope*out_chart['pred_diff'], 'r')
  ax1.plot(out_chart['pred_diff'], out_chart['pred_diff'], color='black', linestyle='--')
  ax1.axvline(x=0, color='black', linestyle='--') 
  ax1.axhline(y=0, color='black', linestyle='--')
  ax1.axvspan(-2, 2, facecolor='green', alpha=0.3) 
  ax1.set_xlabel('Predicted Goal Difference')
  ax1.set_ylabel('Goal Difference')
  ax1.set_title('Predicted vs Actual Goal Difference')

  

  # FILTERED METRICS
  r2_1 = round(r2_score(df['goal_diff'], df['pred_diff']), 4)
  mae_1 = round(mae(df['goal_diff'], df['pred_diff']), 4)


  # RESIDUALS
  residuals = out_chart['pred_diff'] - out_chart['goal_diff']

  fig2, ax2 = plt.subplots()
  ax2.scatter(out_chart['pred_diff'], residuals, s=6, alpha=0.8)
  ax2.set_xlabel('Predicted Difference')
  ax2.set_ylabel('Residuals')
  ax2.set_title('Residual Plot')
  ax2.axhline(y=0, color='r', linestyle='-')
  

  # PREDICTED POINTS
  out_chart = df.groupby('hometeam_name', observed=True).agg({'home_points':'sum', 'pred_points':'sum'}).reset_index()
  slope, intercept, rvalue, pvalue, stderr = stats.linregress(out_chart['pred_points'], out_chart['home_points'])

  fig3, ax3 = plt.subplots()
  ax3.scatter(out_chart['pred_points'], out_chart['home_points'], s=6, alpha=0.8)
  ax3.plot(out_chart['pred_points'], intercept + slope*out_chart['pred_points'], 'r')
  ax3.set_xlabel('Predicted Points')
  ax3.set_ylabel('Actual Points')
  ax3.set_title('Predicted vs Actual Season Points')
  

  # FILTERED METRICS
  r2_3 = round(r2_score(out_chart['home_points'], out_chart['pred_points']), 4)
  mae_3 = round(mae(out_chart['home_points'], out_chart['pred_points']), 4)

  return fig1, r2_1, mae_1, fig2, fig3, r2_3, mae_3