import inspect
import torch
from typing import get_type_hints, get_args, get_origin
from tensor_shape import TensorShape

class TensorSchema:
    def __init__(self, **kwargs):
        """Initialize the schema with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def validate(self):
        type_hints = get_type_hints(self.__class__, include_extras=True)
        shape_env = {}  # optional, used later for symbolic matching

        for field_name, field_type in type_hints.items():
            if get_origin(field_type) is not None:
                args = get_args(field_type)

                for arg in args:
                    if isinstance(arg, TensorShape):
                        expected_shape = arg.dims
                        value = getattr(self, field_name)
                        if isinstance(value, list) or isinstance(value, tuple):
                            if not value:
                                raise ValueError(f"{field_name} is an empty list")

                            # Ensure all tensors in the list have the same shape
                            first = value[0]
                            for i, v in enumerate(value):
                                if not isinstance(v, torch.Tensor):
                                    raise ValueError(f"{field_name}[{i}] is not a torch.Tensor")
                                if v.shape != first.shape:
                                    raise ValueError(
                                        f"{field_name} contains inconsistent shapes: "
                                        f"{first.shape} vs {v.shape} at index {i}"
                                    )

                            # Treat the list as a stacked tensor: shape = (len(list), *tensor.shape)
                            actual_shape = (len(value),) + first.shape

                        elif isinstance(value, torch.Tensor):
                            actual_shape = value.shape

                        else:
                            raise ValueError(f"{field_name} is neither a Tensor nor a List[Tensor]")
                            
                        if len(actual_shape) != len(expected_shape):
                            raise ValueError(f"{field_name} has rank {len(actual_shape)} but expected {len(expected_shape)}")

                        for i, dim in enumerate(expected_shape):
                            if isinstance(dim, int):
                                if actual_shape[i] != dim:
                                    raise ValueError(
                                        f"{field_name} dim[{i}] expected {dim}, got {actual_shape[i]}"
                                    )
                            # TODO: Symbolic dim binding will be handled later
        
        print("Validation passed")

    
    def print_shapes(self):
        """Print TensorShape annotations for debugging."""
        print(f"Shapes in {self.__class__.__name__}:")
        type_hints = get_type_hints(self.__class__, include_extras=True)
        
        for field_name, field_type in type_hints.items():
            if get_origin(field_type) is not None:
                args = get_args(field_type)
                for arg in args:
                    if isinstance(arg, TensorShape):
                        print(f"  {field_name}: {arg}")
