from dataclasses import dataclass
from typing import Callable, Type
import pickle
import os
import traceback
import functools
from uuid import uuid4


@dataclass
class ExceptionsMetadata:
    func_name: str
    kwargs: dict
    traceback_string: str

    def dump(self, exceptions_dir: str):
        os.makedirs(exceptions_dir, exist_ok=True)
        exceptions_path = os.path.join(exceptions_dir, f'{self.func_name}_{uuid4().hex}.pkl')
        with open(exceptions_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, exceptions_path: str) -> 'ExceptionsMetadata':
        with open(exceptions_path, 'rb') as f:
            obj = pickle.load(f)
        return obj


class ExceptionsMonitor:
    """
    Context manager for logging exceptions.
    """
    def __init__(self, exceptions_dir: str, func_name: str):
        """
        Args:
            func: function to be wrapped
            exceptions_dir: directory to store exceptions
        """
        self.exceptions_dir = exceptions_dir
        self.func_name = func_name
        os.makedirs(exceptions_dir, exist_ok=True)

    def __enter__(self):
        """ """
        return self

    def __exit__(self, exc_type: Type[Exception], exc_val: Exception, exc_tb: traceback):
        """
            If exception is raised, dumps exception metadata to pickle file.
        Args:
            exc_type: The exception type raised.
            exc_val: The exception value, which should be an instance of exc_type.
            exc_tb: The exception’s traceback.
        """
        if exc_type is not None:
            func_name = None
            while func_name is None:
                if exc_tb.tb_frame.f_code.co_name == self.func_name:
                    func_name = exc_tb.tb_frame.f_code.co_name
                    break
                if not exc_tb.tb_next:
                    break
                else:
                    exc_tb = exc_tb.tb_next
            assert func_name is not None

            exception_metadata = ExceptionsMetadata(
                func_name=func_name,
                kwargs=exc_tb.tb_frame.f_locals,
                traceback_string=str(traceback.format_exception(etype=exc_type, value=exc_val, tb=exc_tb))
            )
            exception_metadata.dump(exceptions_dir=self.exceptions_dir)
            raise exc_val


def log_kwargs_on_exception(exceptions_dir: str):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                traceback_string = str(traceback.format_exception(type(e), e, e.__traceback__))
                exception_metadata = ExceptionsMetadata(
                    func_name=func.__name__,
                    kwargs=kwargs,
                    traceback_string=traceback_string
                )
                exception_metadata.dump(exceptions_dir=exceptions_dir)
                raise e
        return wrapper
    return decorator


def log_kwargs_on_method_exception(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert len(args) == 1
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback_string = str(traceback.format_exception(type(e), e, e.__traceback__))
            self = args[0]
            exception_metadata = ExceptionsMetadata(
                func_name=func.__name__,
                kwargs=dict(**kwargs, **{'self': self}),
                traceback_string=traceback_string
            )
            exception_metadata.dump(exceptions_dir=self.exceptions_dir)
            raise e
    return wrapper
