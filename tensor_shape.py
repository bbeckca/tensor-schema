from typing import Union

class TensorShape:
    def __init__(self, *dims):
        self.dims = dims
    
    def __repr__(self):
        return f"TensorShape{self.dims}"
