import logging
import csv
import io
import datetime
from enum import Enum

class StringEnum(str, Enum):
    def __repr__(self): return self.value

class LogType(StringEnum):
    POSITION_LIMIT = "POSITION LIMIT"
    PORTFOLIO_MULTIPLIER = "PORTFOLIO MULTIPLIER"

class LogSubType(StringEnum):
    MAX_LEVERAGE = "MAX LEVERAGE"
    MAX_FORECAST = "MAX FORECAST"
    MAX_OPEN_INTEREST = "MAX OPEN INTEREST"
    LEVERAGE_MULTIPLIER = "LEVERAGE_MULTIPLIER"
    CORRELATION_MULTIPLIER = "CORRELATION_MULTIPLIER"
    VOLATILITY_MULTIPLIER = "VOLATILITY_MULTIPLIER"
    JUMP_MULTIPLIER = "JUMP_MULTIPLIER"

class LogMessage():
    _date : str | datetime.datetime
    _type : LogType
    _subtype : LogSubType
    _info : str
    _additional_info : str
    def __init__(
            self,
            DATE : datetime.datetime,
            TYPE : LogType,
            SUBTYPE : LogSubType = None,
            INFO : str = None,
            ADDITIONAL_INFO : str = None):
        self._date = DATE
        self._type = TYPE
        self._subtype = SUBTYPE
        self._info = INFO
        self._additional_info = ADDITIONAL_INFO
        self.message = [self._date, self._type, self._subtype, self._info, self._additional_info]

    @classmethod
    def attrs(cls):
        keys = cls.__annotations__.keys()
        return [x.strip('_') for x in list(keys)]

    def __str__(self):
        return str(self.message)
    
    def __repr__(self):
        return self.message

class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)
        self.write_header()
    
    def format(self, record):
        if not isinstance(record.msg, LogMessage):
            return super().format(record)

        row = [record.levelname]
        row.extend(record.msg.message)
        self.writer.writerow(row)
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()

    def write_header(self):
        header = ['Level']
        header.extend(LogMessage.attrs())
        self.output.write(','.join(map(str, header)))
        self.output.write('\n')
