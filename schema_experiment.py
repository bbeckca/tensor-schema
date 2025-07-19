from typing import TypedDict, Literal, Union, Annotated
from torch import Tensor, randint
from tensor_schema import TensorSchema
from tensor_shape import TensorShape

class Phi3VImagePixelInputs(TensorSchema):
    image_sizes: Annotated[Union[Tensor, list[Tensor]], TensorShape("bn", 2)]


if __name__ == "__main__":
    # Test single tensor
    inputs = Phi3VImagePixelInputs(
        image_sizes=randint(0, 256, (16, 2))
    )
    
    inputs.print_shapes()

    inputs.validate()

    # Test list of tensors
    inputs = Phi3VImagePixelInputs(
        image_sizes=[randint(0, 256, (2,)) for _ in range(16)]
    )

    inputs.print_shapes()

    inputs.validate()