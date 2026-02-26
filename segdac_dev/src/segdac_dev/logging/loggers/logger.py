from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def get_job_id(self) -> str:
        pass

    @abstractmethod
    def set_name(self, name: str):
        pass

    @abstractmethod
    def add_tag(self, tag: str):
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = None, epoch: int = None):
        pass

    @abstractmethod
    def log_metric(self, name: str, value: float, step: int = None, epoch: int = None):
        pass

    @abstractmethod
    def log_parameters(self, parameters: dict):
        pass

    @abstractmethod
    def log_video(self, file: str, step: int):
        pass

    @abstractmethod
    def log_image(self, image_data, name: str, step: int):
        pass

    @abstractmethod
    def log_code(self, folder: str):
        pass

    @abstractmethod
    def log_model(self, model_name, model_file_path):
        pass

    @abstractmethod
    def log_asset(self, asset_path: str, step: int):
        pass

    @abstractmethod
    def log_other(self, key: str, value):
        pass

    @abstractmethod
    def log_html(self, html: str, clear: bool):
        pass
