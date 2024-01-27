# README for Electric Vehicle (EV) Charging Station Analysis

This project focuses on analyzing electric vehicle (EV) charging stations in London, their distribution, growth, and impact on the adoption of EVs. The datasets include total charging stations, rapid charging stations, and the number of ultra-low emission vehicles (ULEVs) registered in different boroughs of London. By integrating these datasets, the project aims to uncover trends, correlations, and potential forecasts regarding the future of EV infrastructure and adoption rates.

## Installation

To run this project, you will need to install several Python libraries. You can install them using `pip`:

```bash
pip install pandas numpy seaborn matplotlib tensorflow
```

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `seaborn` and `matplotlib`: For data visualization.
- `tensorflow`: For building machine learning models.

## Project Structure

- **Data Processing and Cleaning**: The datasets are cleaned by replacing missing values, converting data types, and renaming columns for better readability.
- **Data Analysis**: Exploratory data analysis is performed to understand the distribution of charging stations across different boroughs and their growth over time.
- **Data Visualization**: Various plots are generated to visualize trends in the number of charging stations and registered ULEVs.
- **Machine Learning Modeling**: TensorFlow is used to create regression models to predict the future number of EVs based on the number of charging stations.

## Usage

1. **Data Loading**: Load the datasets into pandas DataFrames.
    ```python
    total_df = pd.read_csv('total_charging.csv')
    rapid_df = pd.read_csv('rapid_charging.csv')
    ulevs = pd.read_csv('ulevs.csv')
    ```

2. **Data Cleaning and Preparation**: Clean the data by handling missing values and converting data types.
    ```python
    total_df['Jan-23 \n(Total Charging Devices) [Note 2]'] = total_df['Jan-23 \n(Total Charging Devices) [Note 2]'].str.replace(',','').fillna(0).astype(int)
    ```

3. **Exploratory Data Analysis (EDA)**: Analyze the datasets to understand the distribution and trends.
    ```python
    sns.pairplot(total_df[['Oct-22\n(Total Charging Devices)', '2022 Q3']])
    ```

4. **Visualization**: Create visualizations to display the analysis results.
    ```python
    sns.barplot(x='Local Authority / Region Name', y='Total Charging Devices', data=total_df)
    ```

5. **Machine Learning Modeling**: Use TensorFlow to build models for predicting future trends.
    ```python
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[2])
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x_data, y_data, epochs=10)
    ```

## Results

The analysis reveals the growth in EV charging infrastructure in London, highlighting areas with significant increases in charging stations and identifying patterns in ULEV registrations. The machine learning models provide insights into potential future trends in EV adoption.

## Contributing

Contributions to this project are welcome. Please open an issue first to discuss your ideas or improvements before making any changes.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.