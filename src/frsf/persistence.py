from os import PathLike
from typing import Self
import joblib


class SaveLoadMixin:
    """Save and load functionality for classes of FederatedRandomSurvivalForest Package."""

    def save(self, path: PathLike):
        """
        Preferred method to save the model to a file.

        Parameters
        ----------
        path : PathLike
            The file path to save the model to.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: PathLike) -> Self:
        """
        Preferred method to load the model from a file.

        Parameters
        ----------
        path : PathLike
            The file path to load the model from.
        """
        return joblib.load(path)
