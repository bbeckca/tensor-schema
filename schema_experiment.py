from typing import TypedDict, Literal, Union, Annotated, Optional
from torch import Tensor, randint
from tensor_schema import TensorSchema
from tensor_shape import TensorShape

class Phi3VImagePixelInputs(TensorSchema):
    aspect_ratios: Optional[Annotated[Union[Tensor, None], TensorShape("bn",)]]
    image_sizes: Annotated[Union[Tensor, list[Tensor], tuple[Tensor]], TensorShape("bn", 2)]


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

    # Test tuple of tensors
    inputs = Phi3VImagePixelInputs(
        image_sizes=tuple((randint(0, 256, (2,)) for _ in range(16)))
    )

    inputs.print_shapes()

    inputs.validate()

    # Test symbolic dims across multiple fields
    inputs = Phi3VImagePixelInputs(
        aspect_ratios=randint(0, 256, (16,)),
        image_sizes=randint(0, 256, (16, 2))
    )

    inputs.print_shapes()

    inputs.validate()