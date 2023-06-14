import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            numerical_features = ['Inches', 'Memory', 'height', 'width', 'Clock', 'Weight_in_kg']
            categorical_features = ['Company', 'Product', 'Gpu', 'OpSys']
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy = "median")),
                    ("scaler",StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaling",StandardScaler())
                ]
            )
#14:06
        except:
            pass
