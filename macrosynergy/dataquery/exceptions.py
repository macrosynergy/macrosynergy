"""Authentication classes for DataQuery of OAuth and CertAuth."""

import datetime
import json

class DQException(Exception):
    """DataQuery Exception class.

    param <str> message: message to be displayed.

    param <base_exception> base_exception: an exception that caused this exception, 
        or would be caused by this exception.
    
    param <dict> kwargs: additional information to be stored in the exception as E.info.

    """
    def __init__(self, message, base_exception=None ,**kwargs):
        super().__init__(message)
        self.message = message
        self.info = {   'message': message, 
                        **kwargs }
        if base_exception:
            self.base_exception = base_exception
            self.info['base_exception'] = base_exception

        
        if "timestamp" not in self.info:
            if ("header" in kwargs) and ("Date" in kwargs["header"]):
                self.info['timestamp'] = kwargs["header"]["Date"]
            else:
                self.info['timestamp'] = datetime.datetime.utcnow().isoformat()

    def __str__(self):
        r = f"{self.message}; Additional Info {json.dumps({str(k):str(v) for k,v in self.info.items() if k != 'message'})}"
        if "base_exception" in self.__dict__:
            r += f" caused by {self.base_exception}"
        return r
    
    def __repr__(self):
        return super().__repr__()
