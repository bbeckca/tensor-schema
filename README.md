# Tensor Schema Prototype

A schema system for validating input shapes in multi-modal models.

## 🚀 Overview

Currently, multi-modal inputs are validated using `_parse_and_validate_*_input` functions, but these only perform minimal checks. Some models only validate input types, making it easy for actual tensor shapes to mismatch documented expectations in classes like `*ImagePixelInputs`. This confuses model developers and maintainers.

## 💡 Solution

I propose adding a base class `TensorSchema` to validate model inputs with automatic shape checking.

### Current Approach

```python
class Phi3VImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """Shape: `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`"""

    image_sizes: torch.Tensor
    """Shape: `(batch_size * num_images, 2)`"""
```

### Proposed Solution

```python
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
    data: Annotated[Union[torch.Tensor, List[torch.Tensor]], TensorShape("bn", "p", 3, "h", "w")]
    image_sizes: Annotated[Union[torch.Tensor, List[torch.Tensor]], TensorShape("bn", 2)]
```

## ✨ Key Features

- **Automatic Validation**: Similar to Pydantic models, validation happens automatically
- **Performance Control**: Ability to disable validation using a flag to avoid performance issues
- **Type Annotations**: Uses `typing_extensions.Annotated` with additional metadata for validation
- **Future-Proof**: Can switch to `typing.Annotated` once Python 3.9 support is dropped

## 🔧 How It Works

### Dimension Validation

- **Constant Dimensions**: Direct checking of constant values
  - Example: Validates that `data.shape[2] == 3`
- **Named Dimensions**: Consistent validation across fields
  - Example: Since `data.shape[0]` and `image_sizes.shape[0]` share the name `bn`, validates that `data.shape[0] == image_sizes[0]`

### List/Tuple Support

- For list/tuple fields instead of tensors:
  - Uses `len()` to check the leading dimension
  - Recursively validates each element for remaining dimensions

## 📦 Package Considerations

This solution can benefit projects outside of vLLM as well, so we should consider developing this as a separate package for broader adoption.

