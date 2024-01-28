# README for Analysis of Volcanic Unrest at Campi Flegrei Caldera, Italy

This repository contains a comprehensive analysis of the volcanic unrest at Campi Flegrei Caldera, Italy, focusing on deformation, seismicity, and geochemical data trends over time. The analysis utilizes data collected by the Vesuvius Observatory and provided by the UCL Hazard Centre.

## Installation

Before running the analysis, ensure that R and the following packages are installed:

- `tidyverse` for data manipulation and visualization
- `gridExtra` for arranging multiple grid-based plots
- `ggplot2` for creating elegant data visualizations
- `dplyr` for data manipulation
- `stats` for statistical functions
- `lubridate` for date-time manipulation
- `plotly` for interactive plots
- `patchwork` for combining multiple ggplot2 plots
- `dygraphs` for interactive time series visualizations
- `ggmap` for spatial visualization
- `latticeExtra` for enhanced lattice plots

You can install these packages using the R command:

```R
install.packages(c('tidyverse', 'gridExtra', 'ggplot2', 'dplyr', 'stats', 'lubridate', 'plotly', 'patchwork', 'dygraphs', 'ggmap', 'latticeExtra'))
```

## Data Overview

The analysis includes several datasets related to the Campi Flegrei area:

- Deformation data (`CF_VT_DEF.csv`): Contains information on ground deformation measurements.
- Seismicity data: Includes the number of volcanic-tectonic (VT) earthquakes.
- Geochemical data from various fumaroles (e.g., `CF_Gas_BG.csv`, `CF_Gas_BN.csv`, `CF_Gas_Pisc.csv`): Contains concentrations of various gases like CO2, H2O, CH4, and others over time.

## Analysis Highlights

The analysis focuses on:

- Temporal trends in gas concentrations and ratios (e.g., CO2/H2O, CO2/CH4) at different fumaroles (Bocca Grande, Bocca Nuova, Pisciarelli).
- Temporal trends in the number of VT earthquakes and ground deformation, and their relationship with geochemical changes.
- Visualization of the data through various plots, including time series, scatter plots, and interactive graphs.
- Comparison of trends across different fumaroles to understand the spatial variability of volcanic unrest indicators.

## Usage

To replicate the analysis:

1. Ensure all required packages are installed and the data files are placed in your working directory.
2. Load the R scripts provided in the repository.
3. Execute the scripts to perform the analysis and generate plots.

## Contributing

Contributions to this project are welcome. Suggestions for further analyses, improvements to existing methods, or additional data sources are highly appreciated.

## License

This project is open source and available under standard open source licenses (please specify the license here, e.g., MIT License).

## Acknowledgements

This work is based on data collected and shared by the Vesuvius Observatory. We extend our gratitude to the UCL Hazard Centre for providing access to this valuable dataset.

For any questions or further information, please contact Mungyeom Kim at mungyeom.kim.20@ucl.ac.uk.
