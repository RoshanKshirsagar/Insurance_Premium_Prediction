from insurance.entity import config_entity, artifact_entity
from insurance.exception import InsuranceException
from insurance.logger import logging
from typing import Optional
import os , sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from insurance.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder
from insurance import utils



## Missing value impute
## Outliers Handling
## Imbalanced Data Handling
## Convert Categorical data into NUmerical data

class DataTransformation:
    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys)
        


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            ## By using SimpleImputer , we will handle Missing Data
            simple_imputer = SimpleImputer(strategy= 'constant', fill_value= 0)

            ## By using RobustScaler , we will handle Outliers
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps = [
                ('Imputer', simple_imputer),
                ('RobustScaler', robust_scaler),
            ])

            return pipeline

        except Exception as e:
            raise InsuranceException(e, sys)





    def initiate_data_transformation(self,) ->artifact_entity.DataTransformationArtifact:
        try:
            # Reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Selecting input features for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis =1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis =1)

            # Selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()

            target_feature_train_arr = target_feature_train_df.squeeze()
            target_feature_test_arr = target_feature_test_df.squeeze()

            # Transforming input features by applying label encoder
            for col in input_feature_train_df.columns:
                if input_feature_test_df[col].dtypes == 'O' :
                    input_feature_train_df[col] = label_encoder.fit_transform(input_feature_train_df[col])
                    input_feature_test_df[col] = label_encoder.fit_transform(input_feature_test_df[col])

                else:
                    input_feature_train_df[col] = input_feature_train_df[col]
                    input_feature_test_df[col] = input_feature_test_df[col]


            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_feature_train_df)

            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)     

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_train_path,
                                        array = train_arr)
            
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_test_path,
                                        array = test_arr)
           
           
            utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj = transformation_pipeline)

            utils.save_object(file_path=self.data_transformation_config.feature_encoder_path, obj = label_encoder)


            data_transformation_artifacts = artifact_entity.DataTransformationArtifact(
                transform_object_path = self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transform_train_path,
                transformed_test_path = self.data_transformation_config.transform_test_path,
                feature_encoder_path = self.data_transformation_config.feature_encoder_path
            )

            return data_transformation_artifacts

                         
        except Exception as e:
            raise InsuranceException(e, sys)
 


  