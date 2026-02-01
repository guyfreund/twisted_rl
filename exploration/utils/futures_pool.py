from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Iterable, List, Callable, Dict, Any, Optional

import numpy as np
from tqdm import tqdm
import traceback
from cachetools import Cache


class CachePool:
    """
    A singleton to maintain all lru caches so that they can be cleared at once
    """
    _cache_list: List[Cache] = []

    @classmethod
    def register_cache(cls, cache: Cache):
        cls._cache_list.append(cache)

    @classmethod
    def clear(cls):
        for cache in cls._cache_list:
            cache.clear()


def clear_cache():
    CachePool.clear()


class ReturnValue:
    def __init__(self, result, e: Optional[Exception] = None, trace=None):
        self.result = result
        self.e = e
        self.trace = trace

    def is_exception(self) -> bool:
        return self.e is not None


class Caller:
    def __init__(self, func: Callable, clear_cache_activate: bool, catch_exceptions: bool, verbose: bool):
        self.func = func
        self.clear_cache_activate = clear_cache_activate
        self.catch_exceptions = catch_exceptions
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        self._manage_cache()
        try:
            return ReturnValue(result=self.func(*args, **kwargs))
        except Exception as e:
            if self.catch_exceptions:
                return ReturnValue(result=None, e=e, trace=traceback.format_exception(etype=(type(e)), value=e, tb=e.__traceback__))
            else:
                if self.verbose:
                    traceback.print_exception(type(e), e, e.__traceback__)
                raise e
        finally:
            if self.clear_cache_activate:
                clear_cache()

    def _manage_cache(self):
        import warnings

        warnings.filterwarnings('ignore', category=UserWarning)

        if self.clear_cache_activate:
            clear_cache()
            import gc
            gc.collect()


class KWArgsMap:
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, lst):
        return self.func(**lst)


class StarMap:
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, lst):
        return self.func(**lst)


class FuturesMultiprocessingPool:
    def __init__(self, processes: int = None, clear_cache_activate: bool = False, catch_exceptions: bool = True,
                 verbose: bool = True, start_method: str = 'spawn'):
        self.processes = processes
        self.clear_cache_activate = clear_cache_activate
        self.catch_exceptions = catch_exceptions
        self.verbose = verbose
        self.executor = None
        self.start_method = start_method
        self.init()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def init(self):
        mp_context = get_context(self.start_method)
        self.executor = ProcessPoolExecutor(max_workers=self.processes, mp_context=mp_context)

    def _process_results_generator(self, results: Iterable[ReturnValue]) -> Iterable:
        for result in results:
            if result.is_exception():
                if self.catch_exceptions:
                    if self.verbose:
                        print(result.trace)
                else:
                    raise result.e
            else:
                yield result.result

    def _process_results(self, results: Iterable[ReturnValue]) -> List:
        return list(self._process_results_generator(results=results))

    def map(self, func: Callable, iterable: Iterable, chunksize: int = 1) -> List:
        wrapped_func = Caller(func=func, catch_exceptions=self.catch_exceptions, verbose=self.verbose,
                              clear_cache_activate=self.clear_cache_activate)
        items = [item for item in iterable]
        total = len(items)
        if self.verbose:
            desc = f'{self.__class__.__name__} processing {total} items with {self.processes} workers'
            results = list(tqdm(self.executor.map(wrapped_func, items, chunksize=chunksize), total=total, desc=desc))
        else:
            results = self.executor.map(wrapped_func, items, chunksize=chunksize)
        return self._process_results(results=results)

    def starmap(self, func: Callable, iterable: Iterable, chunksize: int = 1) -> List:
        return self.map(StarMap(func=func), iterable=iterable, chunksize=chunksize)

    def kwargs_map(self, func: Callable, iterable: Iterable, chunksize: int = 1) -> List:
        return self.map(KWArgsMap(func), iterable=iterable, chunksize=chunksize)

    def close(self):
        self.executor.shutdown(wait=True)

    def terminate(self):
        self.executor.shutdown(wait=False)


def perform_tasks(pool: FuturesMultiprocessingPool, func: Callable, kwargs_list: List[Dict[str, Any]], total: int,
                  init_pool: bool = True, close_pool: bool = True, parallel: bool = True, batch_size: int = 1,
                  chunksize: Optional[int] = None) -> List:
    results = []
    chunksize = int((batch_size if batch_size > 1 else total) / pool.processes) if (pool is not None and chunksize is None) else chunksize

    if parallel:
        try:
            dataloader = CustomDataLoader(dataset=kwargs_list, batch_size=batch_size if batch_size > 1 else total)
            total_batches = len(dataloader)
            if init_pool:
                pool.init()
            for i, batch in enumerate(dataloader):
                print(f'Batch {i}/{total_batches}')
                for result in pool.kwargs_map(func=func, iterable=batch, chunksize=chunksize):
                    results.append(result)
            if close_pool:
                pool.close()

        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            pool.terminate()
            raise e

    else:
        for item in tqdm(iterable=kwargs_list, total=total, desc='Performing Task'):
            result = func(**item)
            results.append(result)

    return results


class CustomDataLoader:
    def __init__(self, dataset: List, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current_index = 0

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.dataset):
            raise StopIteration

        batch = self.dataset[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size

        return batch
