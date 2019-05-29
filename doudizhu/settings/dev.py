from .base import *
import logging

SECRET_KEY = 'fiDSpuZ7QFe8fm0XP9Jb7ZIPNsOegkHYtgKSd4I83Hs='

DATABASE = {
    'host': 'localhost',
    'database': 'ddz',
    'user': 'root',
    'password': '123456',
}
class WordFilter(logging.Filter):
    def __init__(self, param=None):
        self.param = param

    def filter(self, record):
        if self.param is None:
            allow = True
        else:
            allow = self.param in record.msg
        '''
        if allow:
            record.msg = 'changed: ' + record.msg
        '''
        return allow

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'result'],
    },
    'filters': {
        'result_filter': {
            '()': WordFilter,
            'param': 'RESULT',
        }
    },
    'formatters': {
        'simple': {
            'format': '%(asctime).16s %(message)s'
        },
        'precise': {
            'format': '%(asctime).16s %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'result': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'formatter': 'simple',
            'filename': 'result.log',
            'filters': ['result_filter'],
        },
    },
    'loggers': {
        'doudizhu': {
            'level': 'INFO',
            'handlers': ['console', 'result'],
            'propagate': False,
        },
    },
}
