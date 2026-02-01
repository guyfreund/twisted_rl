import inspect
import pkgutil
import os
from pydoc import locate
from typing import Callable, Dict, List


def locate_directories_recursively_in_path(root_path: str, use_abs_path: bool = False) -> List[str]:
    root_path = os.path.abspath(root_path) if use_abs_path else root_path
    paths = [root_path.replace('/', '.')]

    for root, subdirs, _ in os.walk(root_path):
        for subdir in subdirs:
            raw_path = os.path.join(root, subdir)
            path = raw_path.replace('/', '.')
            if not (path.endswith('__pycache__') or path.endswith('.vscode')):
                paths.append(path)

    return paths


def dynamic_import_from_package(package_path: str, predicate: Callable[[type], bool] = None) -> Dict[str, type]:
    """
    This function import dynamically all concrete classes that are located in a specific package (only one level is
    currently supported)
    :param package_path: package path that contains all the classes that should be imported
    :param predicate: predicate that return True for all classes that should be imported
    :return: Dictionary with classes' names as keys and actual classes as values
    """
    modules = [package_path + "." + module.name for module in pkgutil.iter_modules(locate(package_path).__path__)]

    if predicate is None:
        predicate = lambda x: True

    # Goes over all modules and load all concrete classes that fit the input predicate
    classes = {}
    for module in modules:
        members = inspect.getmembers((locate(module)))

        # Go over all members of this nodule to find relevant members
        for name, cls in members:

            # If the current member is a concrete class and derives from the class we add it to the dictionary
            if inspect.isclass(cls) and not inspect.isabstract(cls) and predicate(cls) and name not in classes:
                classes[name] = cls

    return classes


def dynamic_import_from_packages(package_paths: List[str], predicate: Callable[[type], bool] = None) -> Dict[str, type]:
    """
    This function import dynamically all concrete classes that are located in a list of packages (only one level is
    currently supported for each package)
    :param package_paths: list of package paths that contain all the classes that should be imported
    :param predicate: predicate that return True for all classes that should be imported
    :return: Dictionary with classes' names as keys and actual classes as values
    """
    classes = {}
    for package_path in package_paths:
        classes.update(dynamic_import_from_package(package_path, predicate))
    return classes
