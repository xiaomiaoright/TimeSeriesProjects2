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
* What is AR Model?
    * a linear model, where current period values are a sum of past outcomes multiplied by a numeric factor
    * ![equation](https://latex.codecogs.com/gif.latex?x_{t}&space;=&space;C&space;&plus;&space;\varphi&space;x_{t-1}&space;&plus;&space;\varepsilon&space;_{t})
        * ![equation](https://latex.codecogs.com/gif.latex?x_{t-1}) value of x during previous period
        * ![equation](https://latex.codecogs.com/gif.latex?\varphi) any numeric constant by which multiply the lagged variable
        * ![equation](https://latex.codecogs.com/gif.latex?\varepsilon&space;_{t-1}) unpredictable shocks
* AR(p)
* How to determine p? 
    * ACF: 
        * captures both direct and indirect effect of previous values has on current value
        * model should include past lags which have direct significant effect on present → how? PACF
        ```python
        from statsmodels.graphics.tsaplots import sgt
        sgt.plot_acf(df['col1'], lags=40, zero=False)
        ```
    * PACF


        ```python
        from statsmodels.graphics.tsaplots import sgt
        sgt.plot_pacf(df['col1'], lags=40, zero=False, method='ols)
        ```
* AR(1) model with Python

    ```python
    import statsmodels.graphics.tsaplots as sgt
    from statsmodels.tsa.arima_model import ARMA
    # AR(1) model
    model_ar = ARMA(df.market_value, order=(1,0)) 
    # 1 indicates taking lagged 1 for auto regression. 
    # 0 indicates not taking residuals into consideration

    # Fit the model
    model_ar.fit()

    # View model summary
    model_ar.summary()

    # Keys summary parameters:
    # Statistical H0: constant or parameter = 0 
    # z statistics
    # p value: how strong evidence support H0
    ```
* AR(p) model with Python Log-likelihood ratio test
    * What is log likelihood of data?
        * the logarithm of the probability of the observed data coming from the estimated model. For given number of (p, d, q), maximize log likelihood to find paratmer estimates
        * Information Criteria
    * Statistical test: H0: AR(1) and AR(2) are the same

        ```python
        def LLR_test(model1, model2, DF=1):
            from scipy.stats.distributions import chi2
            L1 = model1.fit().llf
            L2 = model2.fit().llf
            LR = (L2-L1)*2
            p = chi2.sf(LR, DF).round(3)
            return p
        # returns the p-values of LLR test
        # p value shows how strong the evidence support the H0 (two models are the same)
        # small p value (<0.05) shows there is a significant difference between the two models
        ```
* Iimiations of AR model:
    * AR model assumes the time series is stationary
    * AR model predict poorly if data is non-staionary
    * How to determine if time series is stationary? → Dickey Fuller test
    * Dickey Fuller Test H0: data is non-stionary, compare p-value with significance level
    * If p value of DF test is high, then data is non stationary. Cannot use AR model
    * Transform non-stationary to stationary
    * Examples

        ```python
        # pandas provides a method .pct_change() to calculate percentage change between two consecutive values
        df['returns']= df['col1'].pct_change(1).mul(100)
        # 1 is the distance in time between the periods to compare
        # Returns 0.02 for 2%
        # to update the output to percentage format, use .mul(100), result will be 0.002 →2
        df = df.iloc[1:] # remove the first data, since it has no return (null)
        # run Dicky Fuller test on returns
        sts.adfuller(df['returns])
        ```
* Take stock prices as example
    * returns (percentage change between lagged period) is preferred than prices because of non stationary nature of prices.
    * normalized returns is preferred
        * normalzied returns account for absolute profitability of investment in contrast to prices
        * normalized returns allow to compare the relative profitability as opposed to non-normalized returns
* Normalization in AR model
    * ![equation](https://latex.codecogs.com/gif.latex?x_{t}&space;\rightarrow&space;%&space;of&space;x_{1})
    * Steps
        1. Set benchmark, usually the first value in series
        2. Divide the series by benchmark

        ```python
        benchmark = df['col1'].iloc[0]
        df['norm'] = df['col1'].div(benchmark).mul(100)
        sts.adfuller(df['norm']) # Test the generated series is stationary or not. H0: non-stationary 
        ```
    * **useing normalized values has no effect on model selection**
* Residuals of AR model
    * Why?
        * If the residual are non random(white noise), then there is a pattern that needs to be accounted for
    * Summary of residuals
    * test stationarity
        * ideally should be random walk process, should be stationary
    
        ```python
        import statsmodels.graphics.tsaplots as sgt
        from statsmodels.tsa.arima_model import ARMA
        # AR(1) model
        model_ar = ARMA(df.market_value, order=(7,0)) 
        model_ar.fit()
        model_ar.summary()
        df['residual'] = model_ar.resid

        # view the mean and variance of residual
        df['residual'].mean()
        df['residual'].var()

        # Test if residual is stationary
        sts.adfuller(df['residual'])
        ```
* MA vs. AR
    * AR Auto regression models rely on past data
    * MA self - correcting model learn from past residuals, adjust for past shocks quickly
    * MA good at prediction random walk dataset
----
## Moving Average MA Model
* What is MA model?
    * ![equation](https://latex.codecogs.com/gif.latex?r_{t}&space;=&space;c&space;&plus;&space;\theta&space;_{1}\varepsilon&space;_{t-1}&space;&plus;\varepsilon&space;_{t})
    * ![equation](https://latex.codecogs.com/gif.latex?r_{t}) the value of 'r' in current period
    * ![equation](https://latex.codecogs.com/gif.latex?\theta&space;_{1}) numeric coefficient for value associated with 1st lag
    * ![equation](https://latex.codecogs.com/gif.latex?\varepsilon&space;_{t}) residual for the current period
    * ![equation](https://latex.codecogs.com/gif.latex?\varepsilon&space;_{t-1}) residual for the past period
* MA(1) ~ AR(∞) with certain restriction
* ACF → MA(q) and PACF → AR(p)
    * Determine which lagged values have a significant direct effect on the present day ones is not relevant
* MA(1) model
    * 

    ```python
    from statsmodels.tsa.arima_model import ARMA
    import statsmodel.graphics.tsaplots as sgt
    model_ret_MA_1 = RMA(df['col1'].iloc[1:], order=(0,1))
    result_ret_MA_1 = model_ret_MA_1.fit()
    result_ret_MA_1.summary()
    ```
* MA(q) model
    * How to determin q order? Log likelihood test
        * LLR test H0: two models are same
        * LLR test returns p value
        ```python
        def LLR_test(model1, model2, DF=1):
            from scipy.stats.distributions import chi2
            L1 = model1.fit().llf
            L2 = model2.fit().llf
            LR = (L2-L1)*2
            p = chi2.sf(LR, DF).round(3)
            return p
        # returns the p-values of LLR test
        # p value shows how strong the evidence support the H0 (two models are the same)
        # small p value (<0.05) shows there is a significant difference between the two models
        ```
    * ACF plot
        * X axis is lagging numbers
        * Y axis correlation: how significant the lagged values affect present values
* MA model residuals
* Normalized values
    * Normalization won't affect the order of model
* Features
    * MA model do not perform well for non stationary data
    * MA models are great in modelling Random Walks because they take into account errors
----
## ARMA Model (p, q)
* What is ARMA(p, q)?
    * p: AR(p) lagged values
    * q: MA(q) lagged errors
    * Allow AR models to calibrate faster and adjust to some huge shocks
    * Give MA terms a much better foundation for predictions
    * ![equation](https://latex.codecogs.com/gif.latex?y_{t}&space;=&space;c&plus;&space;\varphi&space;_{1}y_{t-1}&plus;\theta&space;_{1}\varepsilon&space;_{t-1}&plus;\theta&space;_{t})
    * ![euqation](https://latex.codecogs.com/gif.latex?y_{t},&space;y_{t-1}):
        * Values in the current and 1 period ago respectively
    * ![euqation](https://latex.codecogs.com/gif.latex?\varepsilon&space;_{t},&space;\varepsilon&space;_{t-1})
        * Error tems for the same two periods
    * ![euqation](https://latex.codecogs.com/gif.latex?c)
        * baseline constant factor
    * ![euqation](https://latex.codecogs.com/gif.latex?\varphi&space;_{1})
        * What part of the value last period is relevant in explaining the current one
    * ![euqation](https://latex.codecogs.com/gif.latex?\theta&space;_{1})
        * What part of the errorlast period is relevant in explaining the current value
* ARMA(p,q) model with Python
    ```python
    from statsmodels.tsa.arima_model import ARMA
    import statsmodel.graphics.tsaplots as sgt
    model_ret_AR_1_MA_1 = RMA(df['col1'].iloc[1:], order=(1,1))
    result_ret_AR_1_MA_1 = model_ret_AR_1_MA_1.fit()
    result_ret_AR_1_MA_1.summary()
    ```
    * determine which model is better → LLR test
    ```python
    LLR_test(model_ret_AR_1, model_ret_AR_1_MA_1, DF=1)
    LLR_test(model_ret_MA_1, model_ret_AR_1_MA_1, DF=1)
    # LLR test can only compare nested models
    # p1+q1 > p2+q2 and p1>=p2 and q1>=q2
    # If nested not satisfied, need to compare the LogLikelihood and AIC of two models
    # Higher LogLikelihood and Lower AIC is desired
    ```    
* How to find best ARMA model?
    1. All coefficients significant
    2. High LogLikelihood
    3. Low information criteria
* Residual of ARMA model
    * steps
        * analyzing residuals of predictor
        * extract values and add to dataframe
        * plot and examine ACF

    ```python
    from statsmodels.tsa.arima_model import ARMA
    import statsmodel.graphics.tsaplots as sgt
    df['res_ret_ar_3_ma_2'] = results_ret_ar_3_ma_2.resid[1:]
    df.res_ret_ar_3_ma_2.plot(figsize = (20,5))
    plt.title("Residuals of Returns", size=24)
    plt.show()

    # Plot ACF and see if the residual is random
    sgt.plot_acf(df.res_ret_ar_3_ma_2[2:], zero = False, lags = 40)
    plt.title("ACF Of Residuals for Returns",size=24)
    plt.show()
    # ACF plot might show more lagged numbers are relevant/significant than the model used. 
    # if that is the case, update the model with higher lagged orders
    ```
----
## ARIMA(p, d, q) Model for non-stationary data
* What is ARIMA(p,d,q)?
    * d: integration
    * accounting for non-seasonal difference between periods
* ARIMA(1,1,1)
    * ![equation](https://latex.codecogs.com/gif.latex?\Delta&space;P_{t}=c&plus;\varphi_{1}\Delta&space;P_{t-1}&plus;\theta&space;_{1}\varepsilon&space;_{t-1}&plus;\varepsilon&space;_{t})
    * ![equation](https://latex.codecogs.com/gif.latex?P_{t},P_{t-1})
        * values in current and 1 lagged periods respectively
    * ![equation](https://latex.codecogs.com/gif.latex?\varepsilon&space;_{t},\varepsilon&space;_{t-1})
        * Errors terms for the same two periods
    * ![equation](https://latex.codecogs.com/gif.latex?c)
        * Baseline constant factor
    * ![equation](https://latex.codecogs.com/gif.latex?\varphi_{1})
        * What part of the value last period is relevant in explaining the current one
    * ![equation](https://latex.codecogs.com/gif.latex?\theta&space;_{1})
        * What part of the error last period is relevant in explaining the current value
    * ![equation](https://latex.codecogs.com/gif.latex?\Delta&space;P&space;_{t})
        * = ![equation](https://latex.codecogs.com/gif.latex?P&space;_{t}-P&space;_{t-1})
* ARMIA model is an ARMA(p,q) model for a newly generated time-series
* INtergration:
    * loss observations is unavoidable
* ARIMA Model with Python
    ```python
    import statsmodels.graphics.tsaplots as sgt
    import statsmodels.tsa.stattools as sts
    from statsmodels.tsa.arima_model import ARIMA    
    model_ar_1_i_1_ma_1 = ARIMA(df.market_value, order=(1,1,1))
    results_ar_1_i_1_ma_1 = model_ar_1_i_1_ma_1.fit()
    results_ar_1_i_1_ma_1.summary()


    df['res_ar_1_i_1_ma_1'] = results_ar_1_i_1_ma_1.resid.iloc[:]
    sgt.plot_acf(df['res_ar_1_i_1_ma_1'].iloc[1:], zero = False, lags = 40)
    # ACF plot cannot plot missing values. So first value in the df series need to be dismissed
    plt.title("ACF Of Residuals for ARIMA(1,1,1)",size=20)
    plt.show()
    ```

* Test intergrated data stationary
    ```python
    df['delta_col'] = df['col1'].diff(1) # take order 1 integration

    sts.adfuller(df['delta_col'].iloc[1:]) # test stationarity, avoid the first missing value

    # create ARIMA model: ARIMA(1,0,1) of integrated col  ~ ARIMA(1,1,1) of Col
    model_delta_ar_1_i_1_ma_1 = ARIMA(df['delta_col'][1:], order=(1,0,1))
    results_delta_ar_1_i_1_ma_1 = model_delta_ar_1_i_1_ma_1.fit()
    results_delta_ar_1_i_1_ma_1.summary()    ```

    * ARMA(p,q) for integrated prices and ARIMA(p,1,q) for prices
* ARIMA vs ARMA
    * ARIMA with high order integration
        * More and more computationally expensive
        * Transform data serveral times
        * Differentiate values from zero
        * Possible to converage
        * Numberical instability
        * More layers added, harder to interpret the result
* ARIMAX model: outside factors
    * ![equation](https://latex.codecogs.com/gif.latex?\Delta&space;P_{t}&space;=&space;c&space;&plus;&space;\beta&space;X&space;&plus;&space;\varphi&space;_{1}\Delta&space;P_{t-1}&plus;\theta&space;_{1}\varepsilon&space;_{t-1}&space;&plus;\varepsilon&space;_{t})
    * X: any variables intersted in, exogeneous variables
    * in ARIMA model, pass argument exog = array_type

    ```python
    model_ar_1_i_1_ma_1_Xspx = ARIMA(df.market_value, exog = df.spx, order=(1,1,1))
    results_ar_1_i_1_ma_1_Xspx = model_ar_1_i_1_ma_1_Xspx.fit()
    results_ar_1_i_1_ma_1_Xspx.summary()
    ```
* SARIMAX Model: count for seasonality
    * SARIMAX(p,d,q)(P,D,Q,s) 
        * 4 addtiontional orders
        * first three seasonal variations of arima orders, seasonal(P,D,Q)
        * the fourth is the length of cycle, if s=1, no seasonality
    ```python
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model_sarimax = SARIMAX(df.market_value, exog = df.spx, order=(1,0,1), seasonal_order = (2,0,1,5))
    results_sarimax = model_sarimax.fit()
    results_sarimax.summary()
    ```
* Volatility and Prediction Stability
    * What is volatility?
        * magnitude of residuals
        * ~ variance → stability → low risk → safety
    * How?
        * Square the residuals
            * solves positive negative conundrum
            * penalizes high differences between true values and predictions more
            * increases the importance of big unpredicted shocks
----
## ARCH Model
* Autoregression Conditional Heteroscedasticity model
    * Heteroscedasticity: different dispersion (variance)
    * Conditional: a value dependent on others
    * autoregression: using past values to measure variance which is conditional of the variance of past periods
* Simple ARCH model
    * ![equation](https://latex.codecogs.com/gif.latex?\sigma&space;^{2}) euqation
    * ![equation](https://latex.codecogs.com/gif.latex?Var(y_{t}|y_{t-1})&space;=&space;\alpha&space;_{0}&space;&plus;\alpha_{1}&space;\varepsilon_{t-1}^{2})
    * ![equation](https://latex.codecogs.com/gif.latex?Var(y_{t}|y_{t-1})): conditional variance
    * ![equation](https://latex.codecogs.com/gif.latex?\alpha&space;_{0}): constant factor, ~c
    * ![equation](https://latex.codecogs.com/gif.latex?\alpha_{1}) : coefficient for first term
    * ![equation](https://latex.codecogs.com/gif.latex?\varepsilon_{t-1}^{2}): Squared value of the residual epsilon for the previous period
    * ARCH(q): q orders
* Volatility
    * Numeric measurement of uncertainty
    * not directly observable
* Mean and Variance equation
    * ![equation](https://latex.codecogs.com/gif.latex?r_{t}&space;=&space;\mu&space;_{t}&space;&plus;&space;\varepsilon&space;_{t})
    * ![equation](https://latex.codecogs.com/gif.latex?\sigma&space;{_t^2}&space;=&space;\alpha&space;_{0}&space;&plus;\alpha&space;_{1}\varepsilon&space;_{t-1}^{2})
    * ![equation](https://latex.codecogs.com/gif.latex?\mu&space;_{t}&space;=&space;C_{0}&space;&plus;\varphi&space;_{1}\mu&space;_{t-1})
    * ![equation](https://latex.codecogs.com/gif.latex?\varepsilon&space;_{t}): residual values left after estimating the coefficients
    * ![equation](https://latex.codecogs.com/gif.latex?\mu&space;_{t}): mean, a function of past values and past erros. ARMAX model or constant value depending on the dataset.
        * assume μ is serially uncorrleated → no time-dependent pattern
    ```python
    df['returns'] = df.market_value.pct_change(1)*100
    df['sq_returns'] = df.returns.mul(df.returns)
    # Volatility based on sqaured returns
    df.sq_returns.plot(figsize=(20,5))
    plt.title("Volatility", size = 24)
    plt.show()

    # PACF  of return
    sgt.plot_pacf(df.returns[1:], lags = 40, alpha = 0.05, zero = False , method = ('ols'))
    plt.title("PACF of Returns", size = 20)
    plt.show()
    # PACF of squared returns
    sgt.plot_pacf(df.sq_returns[1:], lags = 40, alpha = 0.05, zero = False , method = ('ols'))
    plt.title("PACF of Squared Returns", size = 20)
    plt.show()
    ```
* arch_model() method
    ```python
    from arch import arch_model
    model_arch_1 = arch_model(df.returns[1:], mean = "Constant", vol = "ARCH", p = 1, dist='t') 
    # setting mean equation: 'Constant' or 'Zero' or "AR" (if "AR", set lags argument = [2,4,6])
    # dist: probability distributions for error terms
    # mean of this series is not serially correlated, time-invariant, not related to past values or past residuals

    # setting volatility model: vol = 'ARCH'
    # specify order: p=1
    results_arch_1 = model_arch_1.fit(update_freq = 5)
    results_arch_1.summary()
    ```
    * arch model summary
        * Constant Mean 
            * r-squared: 
                * R-squared is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.
                * R-squared = Explained variation / Total variation
                * for constant mean, r-sqaure ~0, because no variance to explain
            * LogLikelihood:
                * compare LLR of ARCH Model with ARIMA model, even the simplest ARCH model yields a better estimate than the complex multi-lag ARIMA model
                * **ARCH can only be used to predict future variance, rather than future returns** ARCH can be used to determine the stability of market, but can not predict if prices will go up or down
        * Mean Model
        * Volatility Model
* GARCH: Generalized Autoregressive Conditional Heteroscedasticiity model
