"""
Shared protobuf <-> Python (de)serialization helpers used by the gRPC leader
client (`GRPCLeader`). Keep this module free of client/server logic so a change
only ever needs to be made in one place.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _grpc_available

if TYPE_CHECKING or _grpc_available:
    from lerobot_robot_grpc.protos import device_pb2
else:
    device_pb2 = None

logger = logging.getLogger(__name__)


# Keys are `device.proto` DataType enum values: FLOAT32=0, UINT8=1, UINT16=2, INT32=3.
# Kept as plain ints so this module stays importable without the optional `grpcio` dependency.
_FEATURE_META_BY_PROTO: dict[int, tuple[str, type[np.generic]]] = {
    # Explicit little-endian (e.g. "<f") so that encoding is deterministic across platforms.
    0: ("<f", np.float32),
    3: ("<i", np.int32),
    1: ("<B", np.uint8),
    2: ("<H", np.uint16),
}

# Python scalar types are exposed in `feedback_features`/`action_features`: values parsed
# by `parse_feature` are Python scalars (e.g. via `struct.unpack`), and LeRobot dataset
# features expect Python types (see `hw_to_dataset_features`).
_PYTHON_SCALAR_BY_PROTO: dict[int, type] = {
    0: float,
    3: int,
    1: int,
    2: int,
}


def feature_meta_for(data_type: device_pb2.DataType) -> tuple[str, type[np.generic]]:
    """Maps a protobuf DataType to its struct format character and numpy scalar type."""
    if data_type not in _FEATURE_META_BY_PROTO:
        raise ValueError(f"Unsupported data type '{data_type}'.")
    return _FEATURE_META_BY_PROTO[data_type]


def python_scalar_type_for(data_type: device_pb2.DataType) -> type:
    """Maps a protobuf DataType to the Python scalar type returned by `parse_feature`."""
    if data_type not in _PYTHON_SCALAR_BY_PROTO:
        raise ValueError(f"Unsupported data type '{data_type}'.")
    return _PYTHON_SCALAR_BY_PROTO[data_type]


def encode_image(feature_info: device_pb2.OneFeatureInfo, image: np.ndarray) -> bytes:
    """Encodes an HWC numpy array (RGB(A), uint8) into the encoding declared by the feature info."""
    encoding = feature_info.encoding
    if encoding == device_pb2.Encoding.JPEG:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, feature_info.img_quality])
        elif image.ndim == 3 and image.shape[-1] == 3:
            ok, buf = cv2.imencode(
                ".jpg",
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, feature_info.img_quality],
            )
        else:
            raise ValueError(f"Feature '{feature_info.key}': JPEG encoding only supports 1 or 3 channels.")
    elif encoding == device_pb2.Encoding.PNG:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            ok, buf = cv2.imencode(".png", image)
        elif image.ndim == 3 and image.shape[-1] == 3:
            ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        elif image.ndim == 3 and image.shape[-1] == 4:
            ok, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
        else:
            raise ValueError(f"Feature '{feature_info.key}': PNG encoding only supports 1, 3 or 4 channels.")
    else:
        raise ValueError(f"Unsupported encoding '{encoding}' for feature '{feature_info.key}'.")
    if not ok:
        raise ValueError(f"Failed to encode feature '{feature_info.key}'.")
    return buf.tobytes()


def encode_feature(
    ft_info: dict[str, device_pb2.OneFeatureInfo],
    source: RobotObservation | RobotAction | dict[str, Any],
) -> Iterator[device_pb2.OneFeature]:
    for key in source.keys():
        if key not in ft_info:
            raise KeyError(f"Unknown action key {key}")
        feature_info = ft_info[key]
        shape = (feature_info.shape.H, feature_info.shape.W, feature_info.shape.C)
        if shape == (1, 1, 1):
            if feature_info.encoding != device_pb2.Encoding.RAW:
                raise ValueError(f"Encoding '{feature_info.encoding}' is not supported for scalar feature '{key}'.")
            fmt_char, _ = feature_meta_for(feature_info.type)
            yield device_pb2.OneFeature(key=key, data=struct.pack(fmt_char, source[key]))
        elif feature_info.encoding == device_pb2.Encoding.RAW:
            yield device_pb2.OneFeature(key=key, data=source[key].tobytes())
        else:
            yield device_pb2.OneFeature(key=key, data=encode_image(feature_info, source[key]))


def parse_feature(feature: device_pb2.OneFeature, ft_info: dict[str, device_pb2.OneFeatureInfo]) -> Any:
    """Parses a OneFeature protobuf message into its corresponding value."""
    key = feature.key
    if key not in ft_info:
        raise KeyError(f"Feature '{key}' not found in feature info dictionary.")
    feature_info = ft_info[key]
    data_type = feature_info.type
    shape = feature_info.shape

    fmt_char, np_dtype = feature_meta_for(data_type)

    if not (shape.H == 1 and shape.W == 1 and shape.C == 1):
        if feature_info.encoding == device_pb2.Encoding.RAW:
            return np.frombuffer(feature.data, np_dtype).reshape(shape.H, shape.W, shape.C).copy()
        return decode_encoded_image(feature, feature_info, np_dtype)

    if feature_info.encoding != device_pb2.Encoding.RAW:
        raise ValueError(f"Encoding '{feature_info.encoding}' is not supported for scalar feature '{key}'.")
    expected_size = struct.calcsize(fmt_char)
    if len(feature.data) != expected_size:
        raise ValueError(
            f"Feature '{key}' payload size {len(feature.data)} does not match expected size {expected_size}."
        )
    return struct.unpack(fmt_char, feature.data)[0]


def decode_encoded_image(
    feature: device_pb2.OneFeature,
    feature_info: device_pb2.OneFeatureInfo,
    np_dtype: type[np.generic],
) -> np.ndarray:
    """Decodes a JPEG/PNG-encoded image feature into an HWC numpy array (RGB(A))."""
    encoding = feature_info.encoding
    expected_shape = (feature_info.shape.H, feature_info.shape.W, feature_info.shape.C)
    channels = expected_shape[2]
    encoded = np.frombuffer(feature.data, np.uint8)

    if encoding == device_pb2.Encoding.JPEG:
        if channels == 1:
            flags = cv2.IMREAD_GRAYSCALE
        elif channels == 3:
            flags = cv2.IMREAD_COLOR
        else:
            raise ValueError(f"Feature '{feature.key}': JPEG encoding only supports 1 or 3 channels.")
        image = cv2.imdecode(encoded, flags)
        if image is None:
            raise ValueError(f"Failed to decode JPEG feature '{feature.key}'.")
        if image.ndim == 2:
            image = image[..., None]
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding == device_pb2.Encoding.PNG:
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to decode PNG feature '{feature.key}'.")
        if image.ndim == 2:
            image = image[..., None]
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unsupported encoding '{encoding}' for feature '{feature.key}'.")

    if image.shape != expected_shape:
        raise ValueError(
            f"Feature '{feature.key}' decoded shape {image.shape} does not match expected shape {expected_shape}."
        )
    return image.astype(np_dtype)


def load_feature(
    feature: device_pb2.OneFeature,
    ft_info: dict[str, device_pb2.OneFeatureInfo],
    target: RobotObservation | RobotAction | dict[str, Any],
    *,
    aux_behavior: Literal["keep_latest", "ignore"] = "keep_latest",
) -> None:
    """Loads a OneFeature into `target`.

    `aux_behavior` controls what happens when an AUXILIARY feature fails to parse:
    - "keep_latest" (client side): log and keep the previously received value.
    - "ignore" (server side): silently skip the feature.
    CRITICAL features always raise `DeviceNotConnectedError` on parse failure.
    """
    key = feature.key
    if key not in ft_info:
        raise KeyError(f"Feature '{key}' not found in feature info dictionary.")
    feature_info = ft_info[key]
    try:
        target[key] = parse_feature(feature, ft_info)
    except (struct.error, ValueError) as e:
        if feature_info.criticality == device_pb2.Criticality.CRITICAL:
            logger.error(f"Failed to retrieve critical feature {key}: {e}")
            raise DeviceNotConnectedError(f"Failed to find critical feature '{key}'") from e
        if feature_info.criticality == device_pb2.Criticality.AUXILIARY:
            if aux_behavior == "keep_latest":
                logger.info(f"AUXILIARY feature '{key}' failed to parse; keeping latest value.")
            elif aux_behavior != "ignore":
                raise ValueError(f"Unsupported aux_behavior '{aux_behavior}'.") from e
            return
        raise ValueError(f"Unsupported criticality '{feature_info.criticality}' for feature '{key}'") from e
    except Exception as e:
        logger.error(f"Failed to load feature {key}: {e}")
        raise DeviceNotConnectedError(
            f"Failed to load feature '{key}'. Likely mismatched type and/or shape, or unknown keys."
        ) from e
