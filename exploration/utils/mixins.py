from abc import ABC, abstractmethod
import pickle
import os
from logging import INFO, Logger, DEBUG
from typing import Optional


class PickleableMixin(ABC):
    @property
    @abstractmethod
    def filename(self) -> str:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def dump(self, path: str):
        filename = f'{self.filename}.pkl' if not path.endswith('.pkl') else self.filename
        full_path = os.path.join(path, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            pickle.dump(self, f)


class LoggableMixin:
    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger

    @property
    def logger(self) -> Logger:
        return self._logger

    @logger.setter
    def logger(self, value: Optional[Logger]):
        assert isinstance(value, Logger) if value is not None else True
        self._logger = value

    def _log(self, msg: str, level: int = INFO):
        if self.logger is not None:
            self.logger.log(level=level, msg=msg)
        else:
            print(msg)

    def log_info(self, msg: str):
        self._log(msg, level=INFO)

    def log_debug(self, msg: str):
        self._log(msg, level=DEBUG)

    def log_error(self, msg: str):
        self._log(msg, level=INFO)

    def log_warning(self, msg: str):
        self._log(msg, level=INFO)
