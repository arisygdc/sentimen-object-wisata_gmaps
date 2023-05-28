import pandas as pd
import os

CHECKPOINT_SAVE_DIR = os.path.join(os.getcwd(), ".hasil")
PREPROCESSING_CHECKPOINT = "preprocessing.xlsx"

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
    
    def Cache(self, filename: str):
        if not self.CheckDataframe():
            raise Exception("Dataframe is not initialized")
        file = os.path.join(CHECKPOINT_SAVE_DIR, filename)
        self.dataframe.to_excel(file, index=False)

    def GetCache(self, filename: str):
        file = os.path.join(CHECKPOINT_SAVE_DIR, filename)
        self.dataframe = pd.read_excel(file)
        
        
            


checkpoint = _FileCheckpoint()