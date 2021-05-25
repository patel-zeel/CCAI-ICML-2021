```r
python == 3.8.5
```

## Data

* ### Delhi
    * #### AQI
        * IAQI for one/several pollutants, taken from monitoring stations. [OpenAQ](https://openaq-fetches.s3.amazonaws.com/index.html) - [Script to download](https://patel-zeel.github.io/blog/data/openaq/2021/04/30/Programatically_download_OpenAQ_data.html)
            * Data till July 2019 is problamatic, it mostly has data from a single station
        * Location coordinates (lat, lon).
        * Meteorological features: weather, temperature, pressure, humidity, wind speed and direction. Hourly meteorological data
            * [ERA5 hourly data-single levels](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form) resolution is too low for Delhi data. 37 stations share only 4 ERA5 locations - [notebook link](https://github.com/patel-zeel/CCAI-ICML-2021/blob/main/data/delhi/notebooks/Combine_AQ_ERA5.ipynb) 
            * [ERA5 hourly data-Land](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form) - 37 stations share 13 ERA5 locations. "0.1" degree resolution in lat-long.
            * Checking datasets on CCAI website resources - [link](https://www.climatechange.ai/resources)
            
* ### Beijing
    * #### AQI & Meteorological
        * [U-Air](https://dl.acm.org/doi/10.1145/2487575.2488188) - [Author profile](http://urban-computing.com/yuzheng) - [Dataset Link](https://www.microsoft.com/en-us/research/publication/u-air-when-urban-air-quality-inference-meets-big-data/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F193973%2Fair%2520quality%2520data.zip) - [Download](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Air20Quality20Data.zip)
        * [ADAIN](https://ojs.aaai.org/index.php/AAAI/article/view/11871) - Same author as above - [Dataset download](http://urban-computing.com/data/Data-1.zip)
        * [KDD Cup 2018](https://www.kdd.org/kdd2018/kdd-cup) - [Dataset page](https://www.biendata.xyz/competition/kdd_2018/data/)

## Results
### Bejing dataset from U-Air : spatial interpolation for each time-stamp.
* rf - RandomForest
* svr - Support Vector Regression
* gp_m32/12 - Gaussian Process regression with Matern32/12 kernel
* elst - ElasticNet
* dt - Decision Tree
* dkl - Deep Kernel Learning
* gp_sm_gpytorch - Spectral Mixture kernel known for extrapolation in GPyTorch
* gp_rbf_torch - My own implementation of GP in PyTorch
* mlp - neural network trained on each time-stamp
* mlp_gen - neural network trained on all time-stamps
#### RMSE

|                 | fold_0     | fold_1     | fold_2     | fold_3     | fold_4     | fold_5     | average    |
|:----------------|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|
| rf              | 39.11  | 24.69  | 28.87  | 33.87  | 29.10  | 23.60  | 29.87  |
| svr             | 43.37  | 26.48  | 28.52  | 33.61  | 29.76  | 25.96  | 31.28  |
| gp_m32          | 42.15  | 27.69  | 30.74  | 53.32  | 28.65  | 22.54  | 34.18  |
| gp_m12          | 41.39  | 27.01  | 29.62  | 35.31  | 28.98  | 22.59  | 30.82  |
| gp_linear       | 40.04  | 30.85  | 28.60  | 35.38  | 32.98  | 27.62  | 32.58  |
| elst            | 42.24  | 30.01  | 28.92  | 34.99  | 31.43  | 26.53  | 32.35  |
| dt              | 42.18  | 31.61  | 33.15  | 37.65  | 34.14  | 28.61  | 34.56  |
| dkl             | 45.06  | 28.40  | 31.41  | 33.16  | 29.94  | 27.75  | 32.62  |
| gp_rbf_gpytorch | 41.58  | 24.03  | 29.14  | 34.49  | 27.48  | 22.02  | 29.79  |
| gp_sm_gpytorch  | 43.55  | 29.69  | 31.38  | 38.27  | 30.87  | 25.95  | 33.28  |
| gp_rbf_gpy      | 43.32  | 28.10  | 30.97  | 36.65  | 29.39  | 23.26  | 31.95  |
| gp_rbf_torch    | 53.94  | 31.26  | 35.00  | 37.57  | 31.79  | 24.73  | 35.72  |
| gp_rbf          | 43.18  | 23.67  | 27.98  | 33.41  | 26.83  | 21.69  | 29.46  |
| nsgp_rbf        | 50.15  | 29.27  | 31.13  | 35.92  | 32.61  | 25.89  | 34.16  |
| mlp             | 42.54  | 30.23  | 30.43  | 36.61  | 31.20  | 27.41  | 33.07  |
| mlp_gen | 40.00 | -- | -- | -- | -- | -- | -- |

#### R^2 Score

|                 | fold_0    | fold_1    | fold_2    | fold_3    | fold_4    | fold_5    | average   |
|:----------------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|
| rf              | 0.79  | 0.92  | 0.87  | 0.84  | 0.88  | 0.91  | 0.87  |
| svr             | 0.75  | 0.91  | 0.88  | 0.84  | 0.88  | 0.89  | 0.86  |
| gp_m32          | 0.76  | 0.90  | 0.86  | 0.59  | 0.89  | 0.91  | 0.82  |
| gp_m12          | 0.77  | 0.91  | 0.87  | 0.82  | 0.88  | 0.91  | 0.86  |
| gp_linear       | 0.78  | 0.88  | 0.88  | 0.82  | 0.85  | 0.87  | 0.85  |
| elst            | 0.76  | 0.88  | 0.87  | 0.83  | 0.86  | 0.88  | 0.85  |
| dt              | 0.76  | 0.87  | 0.83  | 0.80  | 0.84  | 0.86  | 0.83  |
| dkl             | 0.73  | 0.90  | 0.85  | 0.84  | 0.88  | 0.87  | 0.84  |
| gp_rbf_gpytorch | 0.77  | 0.93  | 0.87  | 0.83  | 0.90  | 0.92  | 0.87  |
| gp_sm_gpytorch  | 0.75  | 0.89  | 0.85  | 0.79  | 0.87  | 0.89  | 0.84  |
| mlp             | 0.76  | 0.88  | 0.86  | 0.81  | 0.87  | 0.87  | 0.84  |
| mlp_gen | 0.78 | -- | -- | -- | -- | -- | -- |
