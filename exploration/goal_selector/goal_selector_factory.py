from typing import Optional
from logging import Logger

from exploration.goal_selector.goal_selector import GoalSelector
from exploration.utils.import_utils import dynamic_import_from_packages, locate_directories_recursively_in_path


class GoalSelectorFactory:
    classes = dynamic_import_from_packages(
        package_paths=locate_directories_recursively_in_path('exploration/goal_selector'),
        predicate=lambda x: issubclass(x, GoalSelector)
    )

    @classmethod
    def create(cls, class_name: str, logger: Optional[Logger] = None, **kwargs) -> GoalSelector:
        return cls.classes[class_name](logger=logger, **kwargs)
