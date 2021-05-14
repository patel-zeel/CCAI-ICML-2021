# CCAI-ICML-2021
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
        * [U-Air](https://dl.acm.org/doi/10.1145/2487575.2488188) - [Author profile](http://urban-computing.com/yuzheng) - [Dataset Link](https://www.microsoft.com/en-us/research/publication/u-air-when-urban-air-quality-inference-meets-big-data/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F193973%2Fair%2520quality%2520data.zip) - [Donwload](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Air20Quality20Data.zip)
        * [ADAIN](https://ojs.aaai.org/index.php/AAAI/article/view/11871) - Same author as above - [Dataset download](http://urban-computing.com/data/Data-1.zip)
        * [KDD Cup 2018](https://www.kdd.org/kdd2018/kdd-cup) - [Dataset page](https://www.biendata.xyz/competition/kdd_2018/data/)
