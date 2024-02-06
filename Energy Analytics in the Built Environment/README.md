# Project Title: Energy Consumption Analysis - The Impact of Residential Building Characteristics on Building Energy Efficiency in Newham

## Description

The UK's net-zero strategy aims to significantly reduce greenhouse gas (GHG) emissions by targeting not only industrial but also residential sectors. The strategy includes improving energy efficiency in buildings, particularly through the Amended Buildings Regulations 2021, which are expected to cut carbon emissions from buildings by 30% compared to 2013 standards. This essay focuses on the energy efficiency of residential buildings, exploring how built forms, insulation, and household energy services like lighting, heating, and hot water influence it. It specifically examines the London borough of Newham using data analysis techniques such as Decision Trees and K-means clustering to understand building characteristics and Ordinary Least Squares regression to investigate the relationship between these characteristics and energy efficiency.

The literature review highlights residential activities as significant contributors to the UK's GHG emissions, exacerbated by the Covid-19 lockdowns and colder temperatures. Energy efficiency, crucial for reducing GHG emissions, is typically assessed through Energy Performance Certificates (EPCs), though the assessment's reliability has been questioned. Factors influencing energy efficiency include the building's form, insulation quality, and the efficiency of household service systems, with effective insulation and service systems significantly reducing energy needs.

The methodology section details the use of a dataset from the Energy Performance of Building Data England and Wales, focusing on 123,802 residential buildings in Newham. The analysis explores the correlation between building characteristics and EPC ratings, employing Decision Trees to classify buildings by form and insulation, K-means clustering to group buildings by service system efficiency, and OLS Regression to quantify the relationship between building characteristics and energy efficiency.

This project aims to analyze energy consumption patterns in various sectors and regions, focusing on factors such as fuel type, sector, and geographic area. The analysis involves preprocessing data, visualizing trends, and applying machine learning models to understand and predict energy consumption behaviors. The project utilizes Python libraries like Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn for data manipulation, analysis, and visualization.

## Installation

To get started with this project, you'll need to set up a Python environment. Make sure you have Python installed, and then install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Usage

Here's a simple example to demonstrate how to load and preprocess the dataset:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('total_final_energy_consumption.csv')

# Preprocess the data (example)
df['Value'] = pd.to_numeric(df['Value'], errors='coerce').fillna(0)

# Basic data analysis example
print(df.head())
```

For detailed analysis and visualization, refer to the Jupyter notebooks or Python scripts provided in the project repository.

## Data

The project uses several datasets, including:
- `total_final_energy_consumption.csv`: Main dataset containing energy consumption data.
- Other related datasets for in-depth analysis.


## Contributing

We welcome contributions to this project! If you have suggestions or improvements, please fork the repository and create a pull request.

## Acknowledgments

- Data Source: The Energy Performance of Building Data England and Wales published by the Department for Levelling Up, Housing & Communities.
- Libraries: This project makes extensive use of open-source Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

