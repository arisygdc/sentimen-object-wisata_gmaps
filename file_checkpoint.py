import pandas as pd
import numpy as np

class _FileCheckpoint:
    def __init__(self):
        self.dataframe = None

    def SetDataframe(self, data: pd.DataFrame):
        self.dataframe = data
        
    def CheckDataframe(self) -> bool:
        DataframeInitialized = self.dataframe is not None
        if not DataframeInitialized:
            return DataframeInitialized
        IsNotEmptyDataframe = not self.dataframe.empty
        return IsNotEmptyDataframe
    
    def GetDataframe(self) -> pd.DataFrame:
        return self.dataframe

class _BestModel:
    def __init__(self) -> None:
        self.curr_acc=float(0)
        self.model=None
        self.method=None
        self.split=None
        self.vector=None

    def compare_current_acuracy(self, acc_compare: float):
        if self.curr_acc < acc_compare:
            return True
        return False

    def set_bestmodel(self, model, method, split):
        self.model = model
        self.method = method
        self.split = split
    
    def compare_or_save(self, acc_compare: float, model, vector, method, split):
        comp_result = self.compare_current_acuracy(acc_compare)
        if comp_result:
            self.set_bestmodel(model, method, split)
            self.curr_acc=acc_compare
            self.vector=vector
        return comp_result
    
    def get_model(self):
        return self.model
    
    def meta(self):
        return {
            'Model': self.method, 
            'Split': self.split, 
            'Accuracy': self.curr_acc
        }

best_model = _BestModel()
checkpoint = _FileCheckpoint()

