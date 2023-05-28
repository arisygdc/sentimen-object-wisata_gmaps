import pandas as pd

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

checkpoint = _FileCheckpoint()