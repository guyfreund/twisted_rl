from exploration.reachable_configurations.reachable_configurations import IReachableConfigurations
from exploration.utils.import_utils import dynamic_import_from_packages, locate_directories_recursively_in_path


class ReachableConfigurationsFactory:
    classes = dynamic_import_from_packages(
        package_paths=locate_directories_recursively_in_path('exploration/reachable_configurations'),
        predicate=lambda x: issubclass(x, IReachableConfigurations)
    )

    @classmethod
    def create(cls, class_name: str, **kwargs) -> IReachableConfigurations:
        return cls.classes[class_name](**kwargs)

    @classmethod
    def get_cls(cls, class_name: str):
        return cls.classes[class_name]
