from typing import TypedDict, Literal, Union, Annotated
from torch import Tensor, randint
from tensor_schema import TensorSchema
from tensor_shape import TensorShape

class Phi3VImagePixelInputs(TensorSchema):
    image_sizes: Annotated[Union[Tensor, list[Tensor]], TensorShape("bn", 2)]


if __name__ == "__main__":
    inputs = Phi3VImagePixelInputs(
        image_sizes=randint(0, 256, (16, 2))
    )
    
    # Simple debug - just print the shapes
    inputs.print_shapes()

    inputs.validate()
    
    # Test TensorShape printing
    # shape = TensorShape("bn", "p", 3, "h", "w")
    # print(f"\nTest shape: {shape}")