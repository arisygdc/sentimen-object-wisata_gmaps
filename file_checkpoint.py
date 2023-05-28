import pandas as pd

class _FileCheckpoint:
    def __init__(self):
        self.dataframe = None

    def SetDataframe(self, data: pd.DataFrame):
        self.dataframe = data
    
    def GetDataframe(self) -> pd.DataFrame | None:
        return self.dataframe

checkpoint = _FileCheckpoint()