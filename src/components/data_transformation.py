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
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
     
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            numerical_features = ['Inches', 'Memory', 'height', 'width', 'Clock', 'Weight_in_kg']
            categorical_features = ['Company',  'OpSys']
        
            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy = "median")),
                    ("scaler",StandardScaler(with_mean = False))
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaling",StandardScaler(with_mean = False))
                ]
            )
            logging.info(f"categorical columns:{categorical_features} ")
            logging.info(f"numerical columns: {numerical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_features),
                    ("categorical_pipeline",categorical_pipeline,categorical_features)
                ]
            )
            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self,train_path,test_path):
        
        
        try:
             
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data compleated")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_obj()
            target_column_name = "Price_euros"
            numerical_column = ['Inches', 'Memory', 'height', 'width', 'Clock', 'Weight_in_kg']
            categorical_column = ['Company',  'OpSys']
            
            input_feature_train_df = train_df.drop(columns = [target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on train and test data")
             
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        
            
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            target_feature_train_df = np.array(target_feature_train_df).reshape((1, -1))


            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]


            target_feature_test_df = np.array(target_feature_test_df).reshape((1, -1))

            test_arr = np.c_[input_feature_test_arr,target_feature_test_df]

            
             

            logging.info("Saved preprocessing object")


            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,obj = preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e, sys)
