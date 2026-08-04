from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FLOAT32: _ClassVar[DataType]
    UINT8: _ClassVar[DataType]
    UINT16: _ClassVar[DataType]
    INT32: _ClassVar[DataType]

class Criticality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CRITICAL: _ClassVar[Criticality]
    AUXILIARY: _ClassVar[Criticality]

class WatchDogLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    A: _ClassVar[WatchDogLevel]
    B: _ClassVar[WatchDogLevel]
    C: _ClassVar[WatchDogLevel]

class Encoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RAW: _ClassVar[Encoding]
    JPEG: _ClassVar[Encoding]
    PNG: _ClassVar[Encoding]

class CalibrationStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CALIBRATED: _ClassVar[CalibrationStatus]
    NEED_TO_CALIBRATE: _ClassVar[CalibrationStatus]
    CALIBRATING: _ClassVar[CalibrationStatus]

class DeviceStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FATAL: _ClassVar[DeviceStatus]
    IDLE: _ClassVar[DeviceStatus]
    COLLECTION: _ClassVar[DeviceStatus]
    CONTROL: _ClassVar[DeviceStatus]
FLOAT32: DataType
UINT8: DataType
UINT16: DataType
INT32: DataType
CRITICAL: Criticality
AUXILIARY: Criticality
A: WatchDogLevel
B: WatchDogLevel
C: WatchDogLevel
RAW: Encoding
JPEG: Encoding
PNG: Encoding
CALIBRATED: CalibrationStatus
NEED_TO_CALIBRATE: CalibrationStatus
CALIBRATING: CalibrationStatus
FATAL: DeviceStatus
IDLE: DeviceStatus
COLLECTION: DeviceStatus
CONTROL: DeviceStatus

class ImageShape(_message.Message):
    __slots__ = ("H", "W", "C")
    H_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    C_FIELD_NUMBER: _ClassVar[int]
    H: int
    W: int
    C: int
    def __init__(self, H: _Optional[int] = ..., W: _Optional[int] = ..., C: _Optional[int] = ...) -> None: ...

class OneFeatureInfo(_message.Message):
    __slots__ = ("key", "criticality", "watchdog", "type", "shape", "encoding", "img_quality")
    KEY_FIELD_NUMBER: _ClassVar[int]
    CRITICALITY_FIELD_NUMBER: _ClassVar[int]
    WATCHDOG_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    IMG_QUALITY_FIELD_NUMBER: _ClassVar[int]
    key: str
    criticality: Criticality
    watchdog: WatchDogLevel
    type: DataType
    shape: ImageShape
    encoding: Encoding
    img_quality: int
    def __init__(self, key: _Optional[str] = ..., criticality: _Optional[_Union[Criticality, str]] = ..., watchdog: _Optional[_Union[WatchDogLevel, str]] = ..., type: _Optional[_Union[DataType, str]] = ..., shape: _Optional[_Union[ImageShape, _Mapping]] = ..., encoding: _Optional[_Union[Encoding, str]] = ..., img_quality: _Optional[int] = ...) -> None: ...

class OneFeature(_message.Message):
    __slots__ = ("key", "data")
    KEY_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    key: str
    data: bytes
    def __init__(self, key: _Optional[str] = ..., data: _Optional[bytes] = ...) -> None: ...

class CalibrationInfo(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: CalibrationStatus
    def __init__(self, status: _Optional[_Union[CalibrationStatus, str]] = ...) -> None: ...

class DeviceInfo(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: DeviceStatus
    def __init__(self, status: _Optional[_Union[DeviceStatus, str]] = ...) -> None: ...
