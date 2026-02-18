from os import PathLike
from typing import Self
from pathlib import Path
import joblib


class SaveLoadMixin:
    """Save and load functionality for classes of FederatedRandomSurvivalForest Package."""

    def save(self, path: PathLike, create_folders: bool=False):
        """
        Preferred method to save the object to a file.

        Parameters
        ----------
        path : PathLike
            The file path to save the object to.
        create_folders: Bool
            Whether to create parent folders if they do not exist already.
        """

        if create_folders:
            Path(path).mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: PathLike) -> Self:
        """
        Preferred method to load the object from a file.

        Parameters
        ----------
        path : PathLike
            The file path to load the object from.
        """
        return joblib.load(path)
