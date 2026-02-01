from easydict import EasyDict as edict

from exploration.preprocessing.preprocessor import IPreprocessor
from exploration.utils.import_utils import dynamic_import_from_packages, locate_directories_recursively_in_path


class PreprocessorFactory:
    classes = dynamic_import_from_packages(
        package_paths=locate_directories_recursively_in_path('exploration/preprocessing'),
        predicate=lambda x: issubclass(x, IPreprocessor)
    )

    @classmethod
    def create(cls, class_name: str, **kwargs) -> IPreprocessor:
        return cls.classes[class_name](**kwargs)

    @classmethod
    def create_from_cfg(cls, cfg: edict) -> IPreprocessor:
        name, kwargs = cfg.preprocessor.name, cfg.preprocessor.kwargs
        kwargs = {} if kwargs is None else kwargs
        return cls.create(class_name=name, **kwargs)
