# Sales Pipeline and Market Share Analysis Project # 🏆
=============

## Overview ##

This project analyzes sales pipeline calculations and market share for the Water Heater Division, with a focus on residential (RWH-5M) and commercial (CWH-5-M) segments. By integrating Oracle sales data with industry market share data, we develop predictive models to optimize pipeline targets and understand market positioning across different territories.

=============

## Project Structure ##

├── data/
│   ├── Sales Performance and Pipeline Analysis for Water Heater Division (1).csv
│   └── Copy of WHD Market Share Data_2024.xlsx
├── docs/
│   ├── README.md
│   └── requirement.txt
├── flowchart/
│   ├── Required new Business Flow Chart
│   └── Technical Required New Business Pipeline
├── notebooks/
│   ├── 01_data_preparation_EDA.ipynb
│   ├── 02_Data_Feature_Engineering.ipynb
│   ├── 03_data_modeling_hyparameter_tuning.ipynb
│   └── 04_data_evaluation_deployment.ipynb
├── trained pickled model/
│   ├── ???
│   └── ???

=============

## Data Sources ##

### Oracle Sales Data ###
Timeframe: 2022-01-01 to 2024-10-31
994,467 records with 47 features
Contains sales, order, and customer information
### Industry Market Share Data ###
RWH-5M (Residential): 2011-2024
CWH-5-M (Commercial): 2011-2024
Territory and state-level market insights

=============

## Key Features ##

Market Share Analysis 📈
Pipeline Calculation 🧮
Sales Gap Analysis 📉
Territory Performance Metrics 📊
Predictive Modeling 🔮

=============

## Calculations ##

### Market Share ###
Market Share (%) = (Rheem Oracle Units Sold / Industry Total Units Sold) * 100

### Pipeline Needs ###
Total Pipeline Needed = New Business Pipeline / Expected Win Rate
New Business Pipeline = New Total Units Target - Account Management Residuals

### Adjusted Pipeline ###
Sales Gap = New Target Units - Model 1 Predicted Units
Adjusted Pipeline Need = Base Pipeline Need * (1 + (Sales Gap / Target Units))

=============

## Setup and Installation ##

### Clone this repository ###

Bash

git clone [repository-url]
Create and activate virtual environment

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install required packages

Bash

pip install -r requirements.txt

=============

## Usage ##

### Data Preparation and EDA ###

Bash

jupyter notebook notebooks/01_data_preparation_EDA.ipynb
Feature Engineering

Bash

jupyter notebook notebooks/02_Data_Feature_Engineering.ipynb
Model Development

Bash

jupyter notebook notebooks/03_data_modeling_hyparameter_tuning.ipynb
Evaluation and Deployment

Bash

jupyter notebook notebooks/04_data_evaluation_deployment.ipynb

=============

## Dependencies ##

pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
jupyter

## Contributing ##

Fork the repository
Create your feature branch
Commit your changes
Push to the branch
Create a new Pull Request1   
1.
github.com
github.com

=============

## License ##

[Insert License Information]

=============

## Contact ##

[Insert Contact Information]

## Acknowledgments ##

Water Heater Division Team
Data Science Team
[Other acknowledgments]







