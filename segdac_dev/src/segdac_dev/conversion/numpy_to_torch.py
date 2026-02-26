import torch
import numpy as np

numpy_to_torch_dtype_dict = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}


def convert_dtype(input_dtype):
    if input_dtype in numpy_to_torch_dtype_dict.values():
        return input_dtype
    return numpy_to_torch_dtype_dict[input_dtype.type]
