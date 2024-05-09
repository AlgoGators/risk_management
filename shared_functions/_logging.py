import logging
import csv
import io
import datetime


class LogMessage():
    _date : str | datetime.datetime
    _type : str
    _subtype : str
    _info : str
    _additional_info : str
    def __init__(
            self,
            DATE : str | datetime.datetime,
            TYPE : str,
            SUBTYPE : str = None,
            INFO : str = None,
            ADDITIONAL_INFO : str = None):
        _date = DATE
        _type = TYPE
        _subtype = SUBTYPE
        _info = INFO
        _additional_info = ADDITIONAL_INFO
        self.message = [_date, _type, _subtype, _info, _additional_info]

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

