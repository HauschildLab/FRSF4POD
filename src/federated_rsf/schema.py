# %%
import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from .persistence import SaveLoadMixin
from typing import Self
import dataclasses

@dataclasses.dataclass
class DatasetSchema(SaveLoadMixin):
    columns: list[str]
    column_map: Optional[dict[str, str]] = None

    def __post_init__(self):
        # If no column_map is passed, create identity map
        if self.column_map is None:
            identity_map = {col: col for col in self.columns}
            self.column_map = identity_map


class SchemaAligner(BaseEstimator, TransformerMixin, SaveLoadMixin):
    """
    Aligns local data schema to a federated or global schema.

    The input to this transformer should be a pandas DataFrame containing
    the local data and a list of column names representing the full schema.
    The transformer will rename the columns of the local data according to
    an optional column map and then reindex the DataFrame to match the full schema.
    Any missing columns in the local data will be filled with NaN values.
    """

    def fit(self, dataset_schema: DatasetSchema) -> Self:
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

        self.full_schema = dataset_schema.columns
        self.column_map = dataset_schema.column_map
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
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

        data_renamed = data.rename(columns=self.column_map)
        data_reindexed = data_renamed.reindex(columns=self.full_schema)
        return data_reindexed

    def fit_transform(
        self,
        data: pd.DataFrame,
        dataset_schema: DatasetSchema,
    ):
        """
        Fits the SchemaAligner to the provided full schema and column map,
        and then transforms the input DataFrame by aligning its schema to the full schema.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame to be transformed.

        dataset_schema: DatasetSchema
            A dataset schema containing:    
            - a list of column names representing the full schema to align to.
            - a dictionary of string -> string pairs, mapping the local column names to the corresponding column names in the full schema
        

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with aligned schema.
        """

        self.fit(dataset_schema)
        return self.transform(data)


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
        local_schemas: list[DatasetSchema],
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
        list of DataSchemas
            The transformed dataset schemas
        """

        global_columns = set()
        for schema in local_schemas:
            global_columns.update(schema.column_map.values())
        global_columns = sorted(global_columns)

        extra_columns = [
            f"{self.extra_column_prefix}{i}" for i in range(self.extra_columns)
        ]
        global_columns.extend(extra_columns)

        if self.anonymize:
            new_global_columns = [f"feature_{i}" for i in range(len(global_columns))]
            global_columns = (
                np.random.RandomState(self.random_state).permutation(global_columns).tolist()
            )
        else:
            new_global_columns = global_columns

        self.global_columns = new_global_columns
        self.schema_column_map = dict(zip(global_columns, new_global_columns))

        updated_schemas: list[DatasetSchema] = []

        for schema in local_schemas:
            updated_column_map = dict()
            for key, value in schema.column_map.items():

                updated_column_map[key] = self.schema_column_map[value]

            updated_schemas.append(DatasetSchema(self.global_columns, updated_column_map))

        return updated_schemas

    def add_client(self, dataset_schema: DatasetSchema):
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
        needed_columns = set(dataset_schema.column_map.values()) - set(
            self.schema_column_map.keys()
        )
        if len(needed_columns) > self.extra_columns - self.extra_columns_used:
            raise ValueError(
                "Not enough extra columns to accommodate new client features."
            )

        keys_to_remove = []
        updated_column_map = dict()

        for key, value in dataset_schema.column_map.items():
            if value in self.schema_column_map.keys():
                updated_column_map[key] = value
                continue
            extra_column_name = f"{self.extra_column_prefix}{self.extra_columns_used}"
            keys_to_remove.append(extra_column_name)
            self.schema_column_map[value] = self.schema_column_map[extra_column_name]
            self.extra_columns_used += 1
            updated_column_map[key] = extra_column_name

        for key, value in updated_column_map.items():
            updated_column_map[key] = self.schema_column_map[value]

        for key in keys_to_remove:
            del self.schema_column_map[key]


        return DatasetSchema(self.global_columns, updated_column_map)
