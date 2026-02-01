from exploration.utils.import_utils import dynamic_import_from_packages, locate_directories_recursively_in_path
from exploration.utils.criteria.critera import BaseCriteria


class CriteriaFactory:
    classes = dynamic_import_from_packages(
        package_paths=locate_directories_recursively_in_path('exploration/utils/criteria'),
        predicate=lambda x: issubclass(x, BaseCriteria)
    )

    @classmethod
    def create(cls, class_name: str, **kwargs) -> BaseCriteria:
        return cls.classes[class_name](**kwargs)
