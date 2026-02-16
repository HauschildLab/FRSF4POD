# %%
import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from .persistence import SaveLoadMixin
from typing import Self


class SchemaAligner(BaseEstimator, TransformerMixin, SaveLoadMixin):
    """
    Aligns local data schema to a federated or global schema.

    The input to this transformer should be a pandas DataFrame containing
    the local data and a list of column names representing the full schema.
    The transformer will rename the columns of the local data according to
    an optional column map and then reindex the DataFrame to match the full schema.
    Any missing columns in the local data will be filled with NaN values.
    """

    def fit(
        self, full_schema: list[str], column_map: Optional[dict[str, str]] = None
    ) -> Self:
        """
        Fits the SchemaAligner to the provided full schema and column map.

        Parameters
        ----------
        full_schema : list of strings
            A list of column names representing the full schema to align to.

        column_map : dictionary of string -> string pairs, optional
            A dictionary mapping local column names to the corresponding column names in the full schema.
            If None, it is assumed that the local column names are the same as the full schema column names.
        """

        self.full_schema = full_schema
        self.column_map = column_map
        return self

    def transform(self, Data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by aligning its schema to the full schema.

        Parameters
        ----------
        Data : pd.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with aligned schema.
        """

        if self.column_map is None:
            local_columns = Data.columns.tolist()
            column_map = {column: column for column in local_columns}
        else:
            column_map = self.column_map

        Data_renamed = Data.rename(columns=column_map)
        Data_reindexed = Data_renamed.reindex(columns=self.full_schema)
        # Data_reindexed.attrs["aligned"] = True
        return Data_reindexed

    def fit_transform(
        self,
        Data: pd.DataFrame,
        full_schema: list[str],
        column_map: Optional[dict[str, str]] = None,
    ):
        """
        Fits the SchemaAligner to the provided full schema and column map,
        and then transforms the input DataFrame by aligning its schema to the full schema.

        Parameters
        ----------
        Data : pd.DataFrame
            The input DataFrame to be transformed.

        full_schema : list of strings
            A list of column names representing the full schema to align to.

        column_map : dictionary of string -> string pairs, optional
            A dictionary mapping local column names to the corresponding column names in the full schema.
            If None, it is assumed that the local column names are the same as the full schema column names.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with aligned schema.
        """

        self.fit(full_schema, column_map)
        return self.transform(Data)


class SchemaCreator(BaseEstimator, TransformerMixin, SaveLoadMixin):
    """
    Creates a unified schema for federated learning from local feature sets.

    The SchemaCreator takes in the local feature sets and optional column maps
    from multiple clients and creates a unified schema that can be used for
    federated learning. It also provides functionality to add new clients with
    their own feature sets and update the schema accordingly.

    Parameters
    ----------
    anonymize : bool, default=False
        Whether to anonymize the feature names in the schema.

    extra_columns : int, default=0
        The number of extra columns to reserve for accommodating new clients.

    extra_column_prefix : str, default="extra_"
        The prefix to use for naming the extra columns.

    random_state : int, default=None
        The random state to use for anonymization.
    """

    def __init__(
        self,
        anonymize: bool = False,
        extra_columns: int = 0,
        extra_column_prefix: str = "extra_",
        random_state: Optional[int] = None,
    ):
        """
        Initializes the SchemaCreator with the specified parameters.

        Parameters
        ----------
        anonymize : bool, default=False
            Whether to anonymize the feature names in the schema.

        extra_columns : int, default=0
            The number of extra columns to reserve for accommodating new clients.

        extra_column_prefix : str, default="extra_"
            The prefix to use for naming the extra columns.

        random_state : int, default=None
            The random state to use for anonymization.
        """
        super().__init__()
        self.anonymize = anonymize
        self.extra_columns = extra_columns
        self.extra_column_prefix = extra_column_prefix
        self.random_state = random_state
        self.extra_columns_used = 0

    def fit_transform(
        self,
        local_features: list[list[str]],
        local_column_maps: Optional[list[dict]] = None,
    ):
        """
        Creates a unified schema from the provided local feature sets and optional column maps.

        Parameters
        ----------
        local_features : list of lists of strings
            A list of local feature sets, where each feature set is a list of strings.

        local_column_maps : list of dictionaries, optional
            A list of local column maps, where each column map is a dictionary mapping local feature names to full schema feature names.

        Returns
        -------
        tuple of (list of strings, list of dictionaries)
            The unified schema and the updated local column maps.
        """
        local_features = [set(features) for features in local_features]
        if local_column_maps is None:
            local_column_maps = [dict() for _ in range(len(local_features))]

        local_column_maps_closure = [
            {feature: feature for feature in features} for features in local_features
        ]

        local_column_maps_full: list[dict[str, str]] = []
        for column_map, column_map_closure in zip(
            local_column_maps, local_column_maps_closure
        ):
            local_column_maps_full.append(column_map_closure | column_map)

        schema = set()
        for local_map in local_column_maps_full:
            schema.update(local_map.values())
        schema = sorted(schema)

        extra_columns = [
            f"{self.extra_column_prefix}{i}" for i in range(self.extra_columns)
        ]
        schema.extend(extra_columns)

        if self.anonymize:
            new_schema = [f"feature_{i}" for i in range(len(schema))]
            schema = (
                np.random.RandomState(self.random_state).permutation(schema).tolist()
            )
        else:
            new_schema = schema

        self.schema = new_schema
        self.schema_column_map = dict(zip(schema, new_schema))

        for local_map in local_column_maps_full:
            for key, value in local_map.items():

                local_map[key] = self.schema_column_map[value]

        return self.schema, local_column_maps_full

    def add_client(self, features: list[str], column_map: Optional[dict] = None):
        """
        Adds a new client with the provided features and optional column map to the existing schema.

        Parameters
        ----------
        features : list of strings
            The list of features for the new client.
        column_map : dictionary of string -> string pairs, optional
            A dictionary mapping the new client's local feature names to the corresponding column names in the full schema.
            If None, it is assumed that the local feature names are the same as the full schema column names.

        Returns
        -------
        tuple of (list of strings, dictionary)
            The updated schema and the column map for the new client.
        """
        if column_map is None:
            column_map = dict()

        column_map_closure = {feature: feature for feature in features}

        local_column_map_full: dict[str, str] = column_map_closure | column_map

        needed_columns = set(local_column_map_full.values()) - set(
            self.schema_column_map.keys()
        )
        if len(needed_columns) > self.extra_columns - self.extra_columns_used:
            raise ValueError(
                "Not enough extra columns to accommodate new client features."
            )

        keys_to_remove = []

        for key, value in local_column_map_full.items():
            if value in self.schema_column_map.keys():
                continue
            extra_column_name = f"{self.extra_column_prefix}{self.extra_columns_used}"
            keys_to_remove.append(extra_column_name)
            self.schema_column_map[value] = self.schema_column_map[extra_column_name]
            self.extra_columns_used += 1
            local_column_map_full[key] = extra_column_name

        for key, value in local_column_map_full.items():
            local_column_map_full[key] = self.schema_column_map[value]

        for key in keys_to_remove:
            del self.schema_column_map[key]

        return self.schema, local_column_map_full
