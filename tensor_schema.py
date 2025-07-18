import inspect
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
                        actual_shape = value.shape

                        # TODO: handle list/tuple case

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
