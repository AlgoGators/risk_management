class GeneralConfig:
    JOIN_TYPE : str = 'outer'
    DATE_COLUMN : str = 'Date'
    EXPIRATION_COLUMN : str = 'Delivery Month' # 'Expiration'
    UNADJ_COLUMN : str = 'Unadj_Close' # 'Close'
    DAYS_IN_A_WEEK : int = 5

class VARConfig:
    QUANTILE : float = 0.05
    VAR_COLUMN_FORMAT : str = '%s_VAR'
    OBSERVATION_WINDOW : int = 512

class ReturnsConfig:
    RETURN_COLUMN_FORMAT : str = '%s_return'

class VarianceConfig:
    #@ Shared Settings
    MINIMUM_DAILY_OBSERVATIONS : int = 128

    #@ Long Term Variance Settings
    LONG_TERM_VARIANCE_FORMAT : str = '%s_long_term_variance'
    MAXIMUM_DAILY_OBSERVATIONS : int = 256
    
    #@ GARCH Settings
    GARCH_VARIANCE_FORMAT : str = '%s_garch_variance'
    GARCH_VARIANCE_WEIGHTS : tuple = (0.031, 0.669, 0.300)

class CovarianceConfig:
    #@ Shared Settings
    PRODUCT_FORMAT : str = '%sx%s'
    MINIMUM_WEEKLY_OBSERVATIONS : int = 52

    #@ Long Term Variance Settings
    LONG_TERM_COVARIANCE_FORMAT : str = 'LT_%s_%s'
    MAXIMUM_WEEKLY_OBSERVATIONS : int = 52

    #@ GARCH Settings
    GARCH_COVARIANCE_FORMAT : str = 'GARCH_%s_%s'
    GARCH_COVARIANCE_WEIGHTS : tuple = (0.031, 0.669, 0.300)
