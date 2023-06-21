import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")

            print("before loading")

            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            print("After Loading")

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)
            

class CustomData:
    def __init__(self,Inches: float,Memory:int,height:float, width:float, Clock:float, Weight_in_kg: float,Company:str,  OpSys:str):
        self.Inches = Inches,
        self.Memory = Memory,
        self.height = height,
        self.width = width,
        self.Clock = Clock,
        self.Weight_in_kg = Weight_in_kg,
        self.Company = Company,
        self.OpSys = OpSys
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Inches" : [self.Inches],
                "Memory" : [self.Memory],
                "height" :[self.height],
                "width" : [self.width],
                "Clock" : [self.Clock],
                "Weight_in_kg" : [self.Weight_in_kg],
                "Company" : [self.Company],
                "OpSys" : [self.OpSys]
            }

            return pd.DataFrame(custom_data_input_dict)
            


        except Exception as e:
            raise CustomException(e, sys)

krishna = 1
laptop = pd.read_csv("/config/workspace/practice_project/research/cleaned.csv")       
pred_dfs = laptop.tail(1) 
pred_df = pred_dfs.drop("Price_euros",axis = 1)



print(pred_df)
 

predict_pipeline=PredictPipeline()
 
results=predict_pipeline.predict(pred_df)
 
print("predicted :",results)
print("Actual :",pred_dfs["Price_euros"].values)
 

    






