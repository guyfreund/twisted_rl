from easydict import EasyDict as edict

from exploration.utils.schedule.schedule import Schedule
from exploration.utils.import_utils import dynamic_import_from_packages, locate_directories_recursively_in_path


class ScheduleFactory:
    classes = dynamic_import_from_packages(
        package_paths=locate_directories_recursively_in_path('exploration/utils/schedule'),
        predicate=lambda x: issubclass(x, Schedule)
    )

    @classmethod
    def create(cls, class_name: str, **kwargs) -> Schedule:
        return cls.classes[class_name](**kwargs)

    @classmethod
    def create_from_cfg(cls, cfg: edict) -> Schedule:
        name, kwargs = cfg.name, cfg.kwargs
        kwargs = {} if kwargs is None else kwargs
        return cls.create(class_name=name, **kwargs)
