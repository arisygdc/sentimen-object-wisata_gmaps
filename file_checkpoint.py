import pandas as pd

class _FileCheckpoint:
    def __init__(self):
        self.dataframe = None

    def SetDataframe(self, data: pd.DataFrame):
        self.dataframe = data
        
    def CheckDataframe(self) -> bool:
        return self.dataframe is not None
    
    def GetDataframe(self) -> pd.DataFrame:
        return self.dataframe

checkpoint = _FileCheckpoint()