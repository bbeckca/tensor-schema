from typing import TypedDict, Literal, Union, Annotated, Optional, List
from torch import Tensor, randint, randn
from tensor_schema import TensorSchema
from tensor_shape import TensorShape

class Phi3VImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size (number of prompts)
        - n: Number of images
        - p: Number of patches
        - h: Height of each patch
        - w: Width of each patch
    """
    type: Literal["pixel_values"] = "pixel_values"
    data: Annotated[Union[Tensor, List[Tensor]], TensorShape("bn", "p", 3, "h", "w")]
    image_sizes: Annotated[Union[Tensor, List[Tensor]], TensorShape("bn", 2)]


def test_validation(inputs, expected_to_fail=False):
    """Helper method to test validation with clear pass/fail indicators."""
    try:
        inputs.validate()
        if expected_to_fail:
            print("❌ Unexpectedly passed (should have failed)")
        else:
            print("✅ Passed")
    except ValueError as e:
        if expected_to_fail:
            print(f"❌ Failed (expected): {e}")
        else:
            print(f"❌ Failed (unexpected): {e}")


if __name__ == "__main__":
    print("Testing Phi3VImagePixelInputs schema validation...")
    print()

    print("1. Single tensor:")
    inputs = Phi3VImagePixelInputs(
        data=randn(16, 64, 3, 32, 32),
        image_sizes=randint(0, 256, (16, 2))
    )
    test_validation(inputs)

    print("2. List of tensors:")
    inputs = Phi3VImagePixelInputs(
        data=[randn(64, 3, 32, 32) for _ in range(16)],
        image_sizes=[randint(0, 256, (2,)) for _ in range(16)]
    )
    test_validation(inputs)

    print("3. Tuple of tensors:")
    inputs = Phi3VImagePixelInputs(
        data=tuple((randn(64, 3, 32, 32) for _ in range(16))),
        image_sizes=tuple((randint(0, 256, (2,)) for _ in range(16)))
    )
    test_validation(inputs)

    print("4. Symbolic dimension matching:")
    inputs = Phi3VImagePixelInputs(
        data=randn(16, 64, 3, 32, 32),
        image_sizes=randint(0, 256, (16, 2))
    )
    test_validation(inputs)

    print("\nError cases:")
    
    print("5. Mismatched batch sizes:")
    inputs = Phi3VImagePixelInputs(
        data=randn(12, 64, 3, 32, 32),
        image_sizes=randint(0, 256, (16, 2))
    )
    test_validation(inputs, expected_to_fail=True)

    print("6. Missing data field:")
    inputs = Phi3VImagePixelInputs(
        image_sizes=randint(0, 256, (16, 2))
    )
    test_validation(inputs, expected_to_fail=True)

    print("7. Missing image_sizes field:")
    inputs = Phi3VImagePixelInputs(
        data=randn(16, 64, 3, 32, 32)
    )
    test_validation(inputs, expected_to_fail=True)

    print("8. Wrong data shape:")
    inputs = Phi3VImagePixelInputs(
        data=randn(16, 64, 3, 32),  # Missing last dimension
        image_sizes=randint(0, 256, (16, 2))
    )
    test_validation(inputs, expected_to_fail=True)

    print("9. Wrong image_sizes shape:")
    inputs = Phi3VImagePixelInputs(
        data=randn(16, 64, 3, 32, 32),
        image_sizes=randint(0, 256, (16, 3))  # Wrong number of columns
    )
    test_validation(inputs, expected_to_fail=True)

    print("10. Empty lists:")
    inputs = Phi3VImagePixelInputs(
        data=[],
        image_sizes=[]
    )
    test_validation(inputs, expected_to_fail=True)

    print("11. Inconsistent shapes in list:")
    inputs = Phi3VImagePixelInputs(
        data=[randn(64, 3, 32, 32), randn(64, 3, 16, 16)],
        image_sizes=[randint(0, 256, (2,)), randint(0, 256, (2,))]
    )
    test_validation(inputs, expected_to_fail=True)

    print("12. Symbolic + constant dimension mismatch:")
    inputs = Phi3VImagePixelInputs(
        data=randn(16, 64, 4, 32, 32),  # dim[2] = 4, should be 3
        image_sizes=randint(0, 256, (16, 2))
    )
    test_validation(inputs, expected_to_fail=True)

    print("\nAll tests completed.")
