# TimeSeriesProjects2
## Summary of This Project
1. Time Series Dataset
    * Notation
    * Assumptions
    * Data exploration
2. Time Series Data Visualization
    * QQ plot
    * ACF
    * PACF
    * Seasonal Decomposition (ETS)
3. Time Series Data Preprocessing
    * Missing values
    * time stamps
4. Other topics
    * White Noise
    * Random Walk
    * Stationarity
    * Autocorrelation
5. Time Series data modelling
    * AR:
        * Predict movements based on correlations
    * MA:
        * Account for unexpected shocks in sequencial data
    * ARMA and Stationarity
    * ARIMA
    * ARIMAX
    * ARCH, GARCH
----
## Time Series Data
Time Series is a sequence of information which attaches a time period to each value. To be able to analyze time series data, all time-periods must be equal and clearly defined, which would result in a constant frequency. 

**Features of Time Series Data:**

* Data associate with a timestamp or time periods
* Intervals between timestamps are constent
* pattern observed in time-series are expected to persist in the future
* Time-dependent: values for every period are affected by outside factors and by the values of past periods
* Affected by seasonality

**Applications of Time Series Data:**

* Determining stability of financial markets and efficiency portfolios

### Notations
* Frequency: how often values of the dataset are recorded. (ms, s, monthly, yearly, etc.)
* Variables: X, Y
* entire period: T
* interval/single period: t

### Missing values vs. constent intervals
Miss values in Time Sereis is more complex because values between consecutive periods affect each other
* Adjust frequency
    * increase frequency: approximate missing values
    * decrease frequency by aggregation

### QQ Plot
* What is QQ Plot?

    Quantile-Quantile Plot is used to dertmine wheter a dataset is distributed in a certain way. It usually show how the data fits a normal distribution.
* How does QQ plot work?

    It takes all values in data series and arrange them in accending order. Y axis is the values of variable. X axis is the theoritical quantitle, that is how many standard deviations away from the mean values.
----
## Time Series Object in Python
1. Transforming String 'date' column into a 'date type'
    ```python
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    ```
2. Set attribue of a dataframe as an index
    ```python
    df.set_index('date', inplace=True)
    ```
3. Set Desired Date Frequency

    [pandas.DataFrame.asfreq(self, freq, method=None, how: Union[str, NoneType] = None, normalize: bool = False, fill_value=None)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.asfreq.html) Convert TimeSeries to specified frequency.
    
    [Frequency Strings](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)

        Alias   Description
        B       Businessday 
        D       Calendar day
        W       Weekly 
        M       Monthly
        ...
    ```python
    df.asfreq('d')
    ```
    Missing values in the frequency is filled with Null or assigned fill_value argument
4. Fill Mission Values

    df.fillna()
    1) front filling:ffill
    2) back filling: bfill
    3) assign the same values
    ```python
    df['col1'] = df['col1'].fillna(method='ffill')
    df['col2'] = df['col2'].fillna(method='bfill')
    df['col3'] = df['col3'].fillna(value=df['col3'].mean())
    ```
5. Adjust columns

    **Why remove columns?**
    * Less data loaded, faster calculations
    * clarity. Easier to keep track of dataset
    ```python
    del df['col4'], df['col5']
    ```
6. Spliting time series data

    For ML modelling, data need to be splited into train and test data. For common ML algorithms, data will be shuffled before splitting. However, time series data relies on chornological order. Shuffling is impossible. For TS data, training set is from the beginning to cut off point. Testing set is from cut off point to end.

    ```python
    # split 80%-20%
    size = int(len(df)*0.8)
    train = df.iloc[:size]
    test = df.iloc[size:]
    ```
----
## Working with Time Series in Python
### White Noise
* **What is white noise**?
    * a special type of time series, where data doesn't follow a pattern
    * unpredictable
* Features of white noise
    * constant mean
    * ocnstant variance
    * no autocorrelation between periods
* Autocorrelation

    * ![equation](https://latex.codecogs.com/gif.latex?\rho&space;=&space;corr(x_{t},&space;x_{t-1}))
    * NO autocorrelation means no clear relationship between past and present values
* Create white noise in python
    ```python
    wn = np.random.normal(loc = df['col1'].mean(), scale=df['col1'].std(), size=len(df))
    df['wn'] = wn

    ```
### Random Walk
* What is **Randome Walk**?

    * A special type of time series, where values tend to persist over time and the differences between periods are simply white noise.
* Features of Randome Walk
    * ![equation](https://latex.codecogs.com/gif.latex?P_{t}&space;=&space;P_{t-1}&space;&plus;\varepsilon&space;_{t})
        * ![euqation](https://latex.codecogs.com/gif.latex?\varepsilon&space;_{t}\sim&space;WN(\mu&space;,&space;\sigma&space;^{2})) : residue
        * ![equation](https://latex.codecogs.com/gif.latex?P_{t}): price at t
    * difference between periods are simply White Noise
    * Past value ~ present value ~ Future Value
    * Small variations between consecutive time periods
    * Cyclical increases and decreases in short periods of time
* Market efficiency
    * Measures the level of difficulty in forecasting correct future values
    * If price is random walk, future price cannot be predicted precisely
    * If price can be predicted with accuracy, arbitrage opportunity
### Stationarity
* What is Stationarity?

    Data seires from Consecutive time periods of the same length should have identical covariance. 
* **Covariance stationary** assumptions: weak form stationarity
    * Constant mean ![equation](https://latex.codecogs.com/gif.latex?\mu)
    * Constant variance ![equation](https://latex.codecogs.com/gif.latex?\sigma&space;^{2})
    * Constant covariance ![equation](https://latex.codecogs.com/gif.latex?Cov(x_{n},x_{n&plus;k})=Cov(x_{m},x_{m&plus;k}))

    example: white noise
* **Strict Stationary**
    * Samples of identical size have identical distributions
    * ![equation](https://latex.codecogs.com/gif.latex?(x_{t},&space;x_{t&plus;k})\sim&space;Dist(\mu&space;,\sigma&space;^{2})&space;\rightarrow&space;(x_{t&plus;\tau&space;},&space;x_{t&plus;\tau&space;&plus;k})\sim&space;Dist(\mu&space;,\sigma&space;^{2}))
    * Rarely observed in nature
* How to determine Stationarity? →**Dickey-Fuller  test** 
    * Non-stationary statistical test
        * ![equation](https://latex.codecogs.com/gif.latex?H_{0}:&space;\varphi&space;_{1}&space;<&space;1)   data is non-stationary
        * ![equation](https://latex.codecogs.com/gif.latex?H_{1}:&space;\varphi&space;_{1}&space;=&space;1)
        * If  test statistics < critical value, rejuect Null → data is stationary
        * use statsmodels.tsa.stattools.adfuller() to test if a series if stationary

        ```python
        import statsmodels.tsa.stattools as sts 
        sts.adfuller(df['col1'])
        help(sts.adfuller())
        # p values shows the how strongly the evidence support Null hypothesis (non-stationary), which indicates how likely the series data is non-stationary
        ```
### Seasonality
* What is Seasonality?
    * Trends will appear on a cyclical basis
* How to test Seasonality?
    * Split into 3 effects: 
        * trend: pattern
        * seasonal: cyclical effects
        * residual: error of predictions
    * Method1: naive decomposition
        * expect a linear relationship between three parts and the series
        * Two approaches of naive decomposition
            * Additive: observed = trend + seasonal + residual
            * multiplicative: observed = trend x seasonal x residual
        * use statsmodels.tsa.seasonal.seasonal_decompose()

        ```python
        from statsmodel.tsa.seasonal import seasonal_decompose
        s_dec_multiplicative = seasonal_decompose(df['col2'], model = "multiplicative") # model could be model = 'additive'
        s_dec_multiplicative.plot()
        ```
### Correlation
* shuffling not desired for time series data
    * keep the chronological order the series
    * links between past and present values in the series
* What is correlation?
    * correlation ![equation](https://latex.codecogs.com/gif.latex?\rho&space;(x,y)) measure the similiarity in the change of values of two series
* What is Autocorrelation?
    * correlation between a sequence and it self
    * auto correlation between different lags ![equation](https://latex.codecogs.com/gif.latex?\rho&space;(x_{t},x_{t-1}))
### Autocorrelation: ACF vs. PACF
* **ACF**
    * what is ACF?
        * calculate autocorrelation values for whatever lags interested in simutaneously
        * time series itself and lagged version of itself
        * ![equation](https://latex.codecogs.com/gif.latex?\rho&space;(x_{t},x_{t-1}))
        * ![equation](https://latex.codecogs.com/gif.latex?\rho&space;(x_{t},x_{t-2}))
    * use statsmodels.graphics.tsaplots.sgt.plot_acf()
        * line indicates auto correlation between periods
        * blue area indication significance. Larger significance, lower likelihood of autocorrelation

    ```python
        from statsmodels.graphics.tsaplots import sgt
        sgt.plot_acf(df['col1'], lags=40, zero=False)
    ```
* **PACF**
    * PACF calculates only for  ![equation](https://latex.codecogs.com/gif.latex?x_{t-2}\rightarrow&space;x_{t}), cancels out ![equation](https://latex.codecogs.com/gif.latex?x_{t-2}\rightarrow&space;x_{t-1}\rightarrow&space;x_{t})
    * ACF ![equation](https://latex.codecogs.com/gif.latex?x_{t-2}\rightarrow&space;x_{t-1}\rightarrow&space;x_{t})
    * cancles all additional channels a previous period value affects the present one
    * use statsmodels.graphics.tsaplots.sgt.plot_pacf()
----
## How to Time Series Models in Python
* Prediction Power
* Parsimonious (as simple as possible)
    * log-likelihood ratio test
        * can only be applied to models with different degress of freedom
        * same total number of lags → NO LLR
    * Information criteria
        * AIC
        * BIC
    * residual should be white noise
## AR Model: AutoRegression
