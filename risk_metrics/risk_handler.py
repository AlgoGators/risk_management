from risk_config import GeneralConfig, VarianceConfig, CovarianceConfig, ReturnsConfig, VARConfig

from risk_functions import PercentReturns, LongTermVariance, GARCHVariance, LongTermCovariance, GARCHCovariance, HistoricalVAR
import pandas as pd

class RiskHandler:
    def __init__(self, trend_tables : dict[str, pd.DataFrame]):
        self.trend_tables : dict[str, pd.DataFrame] = trend_tables

    @property
    def instruments(self):
        if not hasattr(self, '_instruments'):
            self._instruments = list(self.trend_tables.keys())
            self._instruments.sort()
        return self._instruments
    
    @property
    def daily_returns(self):
        if not hasattr(self, '_daily_returns'):
            for i, instrument_name in enumerate(self.instruments):

                # calculate the percent returns
                instrument_returns : pd.DataFrame = PercentReturns(
                    self.trend_tables[instrument_name],
                    return_column=ReturnsConfig.RETURN_COLUMN_FORMAT % (instrument_name),
                    unadj_column=GeneralConfig.UNADJ_COLUMN,
                    expiration_column=GeneralConfig.EXPIRATION_COLUMN,
                    date_column=GeneralConfig.DATE_COLUMN).daily_returns
                
                if (i == 0):
                    daily_returns = instrument_returns
                else:
                    daily_returns = daily_returns.join(instrument_returns, how=GeneralConfig.JOIN_TYPE)
            
            self._daily_returns = daily_returns
        return self._daily_returns
    
    @daily_returns.setter
    def daily_returns(self, daily_returns : pd.DataFrame):
        self._daily_returns = daily_returns
    
    @property
    def weekly_returns(self):
        if not hasattr(self, '_weekly_returns'):
            self._weekly_returns : pd.DataFrame = PercentReturns.set_weekly_returns(self.daily_returns)
        return self._weekly_returns
    
    @weekly_returns.setter
    def weekly_returns(self, weekly_returns : pd.DataFrame):
        self._weekly_returns = weekly_returns

    @property
    def weekly_product_returns(self):
        if not hasattr(self, '_weekly_product_returns'):
            product_dictionary : dict = {}

            for i, instrument_X in enumerate(self.instruments):
                for j, instrument_Y in enumerate(self.instruments):
                    if j < i:
                        continue
                    product_dictionary[CovarianceConfig.PRODUCT_FORMAT % (instrument_X, instrument_Y)] = self.weekly_returns[ReturnsConfig.RETURN_COLUMN_FORMAT % (instrument_X)] * self.weekly_returns[ReturnsConfig.RETURN_COLUMN_FORMAT % (instrument_Y)]

            self._weekly_product_returns = pd.DataFrame(product_dictionary, index=self.weekly_returns.index)
        return self._weekly_product_returns

    @property
    def long_term_variances(self):
        if not hasattr(self, '_long_term_variances'):
            variances_dictionary : dict = {}

            for i, instrument in enumerate(self.instruments):
                instrument_long_term_variance : LongTermVariance = LongTermVariance.LongTermVariance(
                    self.daily_returns.loc[:, [ReturnsConfig.RETURN_COLUMN_FORMAT % (instrument)]],
                    percent_return_column=ReturnsConfig.RETURN_COLUMN_FORMAT % (instrument),
                    long_term_variance_column=VarianceConfig.LONG_TERM_VARIANCE_FORMAT % (instrument),
                    date_column=GeneralConfig.DATE_COLUMN,
                    minimum_observations=VarianceConfig.MINIMUM_DAILY_OBSERVATIONS,
                    maximum_observations=VarianceConfig.MAXIMUM_DAILY_OBSERVATIONS)
            
                variances_dictionary[VarianceConfig.LONG_TERM_VARIANCE_FORMAT % (instrument)] = instrument_long_term_variance.df()

            long_term_variances : pd.DataFrame
            for i, instrument in enumerate(self.instruments):
                instrument_long_term_variances = variances_dictionary[VarianceConfig.LONG_TERM_VARIANCE_FORMAT % (instrument)]
                if (i == 0):
                    long_term_variances = instrument_long_term_variances
                    continue

                long_term_variances = long_term_variances.join(
                    instrument_long_term_variances, how=GeneralConfig.JOIN_TYPE)

            self._long_term_variances = long_term_variances
        return self._long_term_variances

    @long_term_variances.setter
    def long_term_variances(self, long_term_variances : pd.DataFrame):
        self._long_term_variances = long_term_variances

    @property
    def GARCH_variances(self):
        if not hasattr(self, '_GARCH_variances'):
            variances_dictionary : dict = {}

            for i, instrument in enumerate(self.instruments):
                instrument_GARCH_variance : GARCHVariance = GARCHVariance().GARCHVariance(
                    self.long_term_variances.loc[:, [VarianceConfig.LONG_TERM_VARIANCE_FORMAT % (instrument)]],
                    self.daily_returns.loc[:, [ReturnsConfig.RETURN_COLUMN_FORMAT % (instrument)]],
                    weights=VarianceConfig.GARCH_VARIANCE_WEIGHTS,
                    percent_return_column=ReturnsConfig.RETURN_COLUMN_FORMAT % (instrument),
                    long_term_variance_column=VarianceConfig.LONG_TERM_VARIANCE_FORMAT % (instrument),
                    GARCH_variance_column=VarianceConfig.GARCH_VARIANCE_FORMAT % (instrument),
                    date_column=GeneralConfig.DATE_COLUMN,
                    minimum_observations=VarianceConfig.MINIMUM_DAILY_OBSERVATIONS)

                variances_dictionary[VarianceConfig.GARCH_VARIANCE_FORMAT % (instrument)] = instrument_GARCH_variance.df()

            GARCH_variances : pd.DataFrame
            for i, instrument in enumerate(self.instruments):
                instrument_GARCH_variances : GARCHVariance = variances_dictionary[VarianceConfig.GARCH_VARIANCE_FORMAT % (instrument)]
                if (i == 0):
                    GARCH_variances = instrument_GARCH_variances
                    continue

                GARCH_variances = GARCH_variances.join(
                    instrument_GARCH_variances, how=GeneralConfig.JOIN_TYPE)

            self._GARCH_variances = GARCH_variances
        return self._GARCH_variances
    
    @GARCH_variances.setter
    def GARCH_variances(self, GARCH_variances : pd.DataFrame):
        self._GARCH_variances = GARCH_variances
    
    @property
    def long_term_covariances(self):
        if not hasattr(self, '_long_term_covariances'):
            covariances_dictionary : dict = {}

            for i, instrument_X in enumerate(self.instruments):
                for j, instrument_Y in enumerate(self.instruments):
                    if (j < i):
                        # Skip the lower triangle of the covariance matrix (also could just steal this value from the upper triangle)
                        continue
                    
                    instrument_long_term_covariance : LongTermCovariance = LongTermCovariance(
                        product_returns=self.weekly_product_returns.loc[:, [CovarianceConfig.PRODUCT_FORMAT % (instrument_X, instrument_Y)]],
                        product_column=CovarianceConfig.PRODUCT_FORMAT % (instrument_X, instrument_Y),
                        column_X=instrument_X,
                        column_Y=instrument_Y,
                        long_term_covariance_format=CovarianceConfig.LONG_TERM_COVARIANCE_FORMAT).LongTermCovariance(
                        minimum_observations=CovarianceConfig.MINIMUM_WEEKLY_OBSERVATIONS,
                        maximum_observations=CovarianceConfig.MAXIMUM_WEEKLY_OBSERVATIONS,
                        date_column=GeneralConfig.DATE_COLUMN)

                    covariances_dictionary[CovarianceConfig.LONG_TERM_COVARIANCE_FORMAT % (instrument_X, instrument_Y)] = instrument_long_term_covariance.df()

            long_term_covariances : pd.DataFrame
            for i, column_X in enumerate(self.instruments):
                for j, column_Y in enumerate(self.instruments):
                    if (j < i):
                        # Skip the lower triangle of the covariance matrix (also could just steal this value from the upper triangle)
                        continue

                    instrument_long_term_covariances = covariances_dictionary[CovarianceConfig.LONG_TERM_COVARIANCE_FORMAT % (column_X, column_Y)]
                    
                    if (i == 0 and j == 0):
                        long_term_covariances = instrument_long_term_covariances
                        continue
                    
                    long_term_covariances = long_term_covariances.join(
                        instrument_long_term_covariances, how=GeneralConfig.JOIN_TYPE)

            self._long_term_covariances = long_term_covariances
        return self._long_term_covariances
    
    @long_term_covariances.setter
    def long_term_covariances(self, long_term_covariances : pd.DataFrame):
        self._long_term_covariances = long_term_covariances
    
    @property
    def GARCH_covariances(self):
        if not hasattr(self, '_GARCH_covariances'):
            covariances_dictionary : dict = {}

            for i, instrument_X in enumerate(self.instruments):
                for j, instrument_Y in enumerate(self.instruments):
                    if (j < i):
                        # Skip the lower triangle of the covariance matrix (also could just steal this value from the upper triangle)
                        continue
                    instrument_GARCH_covariance : GARCHCovariance = GARCHCovariance(
                        long_term_covariances=self.long_term_covariances.loc[:, [CovarianceConfig.LONG_TERM_COVARIANCE_FORMAT % (instrument_X, instrument_Y)]],
                        long_term_covariance_format=CovarianceConfig.LONG_TERM_COVARIANCE_FORMAT,
                        product_returns=self.weekly_product_returns.loc[:, [CovarianceConfig.PRODUCT_FORMAT % (instrument_X, instrument_Y)]],
                        product_column=CovarianceConfig.PRODUCT_FORMAT % (instrument_X, instrument_Y),
                        column_X=instrument_X,
                        column_Y=instrument_Y,
                        weights=CovarianceConfig.GARCH_COVARIANCE_WEIGHTS,
                        GARCH_covariance_format=CovarianceConfig.GARCH_COVARIANCE_FORMAT).GARCHCovariance(
                        minimum_observations=CovarianceConfig.MINIMUM_WEEKLY_OBSERVATIONS,
                        date_column=GeneralConfig.DATE_COLUMN)

                    covariances_dictionary[CovarianceConfig.GARCH_COVARIANCE_FORMAT % (instrument_X, instrument_Y)] = instrument_GARCH_covariance.df()

            GARCH_covariances : pd.DataFrame
            for i, instrument_X in enumerate(self.instruments):
                for j, instrument_Y in enumerate(self.instruments):
                    if (j < i):
                        # Skip the lower triangle of the covariance matrix (also could just steal this value from the upper triangle)
                        continue

                    instrument_GARCH_covariances = covariances_dictionary[CovarianceConfig.GARCH_COVARIANCE_FORMAT % (instrument_X, instrument_Y)]
                    
                    if (i == 0 and j == 0):
                        GARCH_covariances = instrument_GARCH_covariances
                        continue
                    
                    GARCH_covariances = GARCH_covariances.join(
                        instrument_GARCH_covariances, how=GeneralConfig.JOIN_TYPE)

            self._GARCH_covariances = GARCH_covariances
        return self._GARCH_covariances
    
    @GARCH_covariances.setter
    def GARCH_covariances(self, GARCH_covariances : pd.DataFrame):
        self._GARCH_covariances = GARCH_covariances

    @property
    def one_day_VAR(self):
        if not hasattr(self, '_one_day_VAR'):
            for instrument in self.instruments:
                historical_VAR : HistoricalVAR = HistoricalVAR.HistoricalVAR(
                    daily_returns=self.daily_returns.loc[:, [ReturnsConfig.RETURN_COLUMN_FORMAT % (instrument)]],
                    VAR_column=VARConfig.VAR_COLUMN_FORMAT % (instrument),
                    observation_window=VARConfig.OBSERVATION_WINDOW,
                    return_column=ReturnsConfig.RETURN_COLUMN_FORMAT % (instrument),
                    quantile=VARConfig.QUANTILE,
                    date_column=GeneralConfig.DATE_COLUMN)

                if (instrument == self.instruments[0]):
                    one_day_VAR = historical_VAR.df()
                else:
                    one_day_VAR = one_day_VAR.join(historical_VAR.df(), how=GeneralConfig.JOIN_TYPE)
            
            self._one_day_VAR = one_day_VAR

        return self._one_day_VAR

    @one_day_VAR.setter
    def one_day_VAR(self, one_day_VAR : pd.DataFrame):
        self._one_day_VAR = one_day_VAR

    def conform_data(self):
        # Get the indexes from all the dataframes
        #* this should also initialize each property incase they havent been initialized yet
        indexes = [self.daily_returns.index, self.long_term_variances.index, self.GARCH_variances.index, 
                   self.long_term_covariances.index, self.GARCH_covariances.index, self.one_day_VAR.index]

        # Convert the indexes into pandas Series objects
        indexes = [pd.Series(index) for index in indexes]

        # Perform an outer join on the indexes
        merged_index = pd.concat(indexes).drop_duplicates()

        # Reindex all the dataframes with the merged index
        self.daily_returns = self.daily_returns.reindex(merged_index)
        self.weekly_returns = self.weekly_returns.reindex(merged_index)
        self.long_term_variances = self.long_term_variances.reindex(merged_index)
        self.GARCH_variances = self.GARCH_variances.reindex(merged_index)
        self.long_term_covariances = self.long_term_covariances.reindex(merged_index)
        self.GARCH_covariances = self.GARCH_covariances.reindex(merged_index)
        self.one_day_VAR = self.one_day_VAR.reindex(merged_index)


        # We assume any gaps in percent returns at this point are because the market was closed that day,
        # but only fill forward; 
        #* apologies for the complexity but it was the only way i could find
        # Find the index of the first non-NaN value in each column
        first_non_nan_index = self.daily_returns.apply(lambda x: x.first_valid_index())

        # Iterate over each column and replace NaN values below the first non-NaN index with 0
        for column in self.daily_returns.columns:
            first_index = first_non_nan_index[column]
            self.daily_returns.loc[first_index:, column] = self.daily_returns.loc[first_index:, column].fillna(0)


        self.weekly_returns = self.weekly_returns.fillna(method='ffill')

        self.long_term_variances = self.long_term_variances.fillna(method='ffill')
        self.GARCH_variances = self.GARCH_variances.interpolate(
            method='linear', limit_direction='forward')

        self.long_term_covariances = self.long_term_covariances.fillna(method='ffill')
        self.GARCH_covariances = self.GARCH_covariances.interpolate(
            method='linear', limit_direction='forward')
        
        self.one_day_VAR = self.one_day_VAR.fillna(method='ffill')

    def save_data(self):
        self.daily_returns.to_csv('daily_returns.csv')
        self.weekly_returns.to_csv('weekly_returns.csv')
        self.long_term_variances.to_csv('long_term_variances.csv')
        self.GARCH_variances.to_csv('GARCH_variances.csv')
        self.long_term_covariances.to_csv('long_term_covariances.csv')
        self.GARCH_covariances.to_csv('GARCH_covariances.csv')
        self.one_day_VAR.to_csv('one_day_VAR.csv')

if __name__ == '__main__':
    trend_tables = {}
    
    instruments = ['ES_data', 'ZN_data', 'RB_data']

    for instrument in instruments:
        trend_tables[instrument] = pd.read_csv(f'unittesting/test_data/{instrument}.csv')

    risk_handler = RiskHandler(trend_tables)
    risk_handler.conform_data()
    risk_handler.save_data()
