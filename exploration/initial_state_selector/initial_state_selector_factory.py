from typing import Optional
from logging import Logger

from exploration.initial_state_selector.initial_state_selector import InitialStateSelector
from exploration.utils.import_utils import dynamic_import_from_packages, locate_directories_recursively_in_path


class InitialStateSelectorFactory:
    classes = dynamic_import_from_packages(
        package_paths=locate_directories_recursively_in_path('exploration/initial_state_selector'),
        predicate=lambda x: issubclass(x, InitialStateSelector)
    )

    @classmethod
    def create(cls, class_name: str, logger: Optional[Logger] = None, **kwargs) -> InitialStateSelector:
        return cls.classes[class_name](logger=logger, **kwargs)
