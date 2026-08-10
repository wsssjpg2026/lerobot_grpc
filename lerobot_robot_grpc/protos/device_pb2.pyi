import datetime

from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DATA_TYPE_UNSPECIFIED: _ClassVar[DataType]
    FLOAT32: _ClassVar[DataType]
    UINT8: _ClassVar[DataType]
    UINT16: _ClassVar[DataType]
    INT32: _ClassVar[DataType]

class Criticality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CRITICALITY_UNSPECIFIED: _ClassVar[Criticality]
    CRITICALITY_CRITICAL: _ClassVar[Criticality]
    CRITICALITY_AUXILIARY: _ClassVar[Criticality]

class WatchDogLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    WATCH_DOG_LEVEL_UNSPECIFIED: _ClassVar[WatchDogLevel]
    WATCH_DOG_LEVEL_A: _ClassVar[WatchDogLevel]
    WATCH_DOG_LEVEL_B: _ClassVar[WatchDogLevel]
    WATCH_DOG_LEVEL_C: _ClassVar[WatchDogLevel]

class Encoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENCODING_UNSPECIFIED: _ClassVar[Encoding]
    RAW: _ClassVar[Encoding]
    JPEG: _ClassVar[Encoding]
    PNG: _ClassVar[Encoding]
    H264: _ClassVar[Encoding]

class FrameQuality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FRAME_QUALITY_UNSPECIFIED: _ClassVar[FrameQuality]
    FRAME_QUALITY_GOOD: _ClassVar[FrameQuality]
    FRAME_QUALITY_DEGRADED: _ClassVar[FrameQuality]
    FRAME_QUALITY_STALE: _ClassVar[FrameQuality]

class CalibrationStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CALIBRATION_STATUS_UNSPECIFIED: _ClassVar[CalibrationStatus]
    CALIBRATED: _ClassVar[CalibrationStatus]
    NEED_TO_CALIBRATE: _ClassVar[CalibrationStatus]
    CALIBRATING: _ClassVar[CalibrationStatus]

class DeviceStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEVICE_STATUS_UNSPECIFIED: _ClassVar[DeviceStatus]
    FATAL: _ClassVar[DeviceStatus]
    IDLE: _ClassVar[DeviceStatus]
    COLLECTION: _ClassVar[DeviceStatus]
    CONTROL: _ClassVar[DeviceStatus]
DATA_TYPE_UNSPECIFIED: DataType
FLOAT32: DataType
UINT8: DataType
UINT16: DataType
INT32: DataType
CRITICALITY_UNSPECIFIED: Criticality
CRITICALITY_CRITICAL: Criticality
CRITICALITY_AUXILIARY: Criticality
WATCH_DOG_LEVEL_UNSPECIFIED: WatchDogLevel
A: WatchDogLevel
B: WatchDogLevel
C: WatchDogLevel
ENCODING_UNSPECIFIED: Encoding
RAW: Encoding
JPEG: Encoding
PNG: Encoding
H264: Encoding
FRAME_QUALITY_UNSPECIFIED: FrameQuality
FRAME_QUALITY_GOOD: FrameQuality
FRAME_QUALITY_DEGRADED: FrameQuality
FRAME_QUALITY_STALE: FrameQuality
CALIBRATION_STATUS_UNSPECIFIED: CalibrationStatus
CALIBRATED: CalibrationStatus
NEED_TO_CALIBRATE: CalibrationStatus
CALIBRATING: CalibrationStatus
DEVICE_STATUS_UNSPECIFIED: DeviceStatus
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
    __slots__ = ("key", "data", "produce_ts", "quality")
    KEY_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    PRODUCE_TS_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    key: str
    data: bytes
    produce_ts: _timestamp_pb2.Timestamp
    quality: FrameQuality
    def __init__(self, key: _Optional[str] = ..., data: _Optional[bytes] = ..., produce_ts: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., quality: _Optional[_Union[FrameQuality, str]] = ...) -> None: ...

class CalibrationInfo(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: CalibrationStatus
    def __init__(self, status: _Optional[_Union[CalibrationStatus, str]] = ...) -> None: ...

class CalibrateRequest(_message.Message):
    __slots__ = ("force",)
    FORCE_FIELD_NUMBER: _ClassVar[int]
    force: bool
    def __init__(self, force: _Optional[bool] = ...) -> None: ...

class CalibrationFrame(_message.Message):
    __slots__ = ("readings",)
    class MotorReading(_message.Message):
        __slots__ = ("name", "position", "range_min", "range_max")
        NAME_FIELD_NUMBER: _ClassVar[int]
        POSITION_FIELD_NUMBER: _ClassVar[int]
        RANGE_MIN_FIELD_NUMBER: _ClassVar[int]
        RANGE_MAX_FIELD_NUMBER: _ClassVar[int]
        name: str
        position: int
        range_min: int
        range_max: int
        def __init__(self, name: _Optional[str] = ..., position: _Optional[int] = ..., range_min: _Optional[int] = ..., range_max: _Optional[int] = ...) -> None: ...
    READINGS_FIELD_NUMBER: _ClassVar[int]
    readings: _containers.RepeatedCompositeFieldContainer[CalibrationFrame.MotorReading]
    def __init__(self, readings: _Optional[_Iterable[_Union[CalibrationFrame.MotorReading, _Mapping]]] = ...) -> None: ...

class DeviceInfo(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: DeviceStatus
    def __init__(self, status: _Optional[_Union[DeviceStatus, str]] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("features",)
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    features: _containers.RepeatedCompositeFieldContainer[OneFeature]
    def __init__(self, features: _Optional[_Iterable[_Union[OneFeature, _Mapping]]] = ...) -> None: ...

class GetInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetInfoResponse(_message.Message):
    __slots__ = ("observation_features", "action_features", "feedback_features")
    OBSERVATION_FEATURES_FIELD_NUMBER: _ClassVar[int]
    ACTION_FEATURES_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_FEATURES_FIELD_NUMBER: _ClassVar[int]
    observation_features: _containers.RepeatedCompositeFieldContainer[OneFeatureInfo]
    action_features: _containers.RepeatedCompositeFieldContainer[OneFeatureInfo]
    feedback_features: _containers.RepeatedCompositeFieldContainer[OneFeatureInfo]
    def __init__(self, observation_features: _Optional[_Iterable[_Union[OneFeatureInfo, _Mapping]]] = ..., action_features: _Optional[_Iterable[_Union[OneFeatureInfo, _Mapping]]] = ..., feedback_features: _Optional[_Iterable[_Union[OneFeatureInfo, _Mapping]]] = ...) -> None: ...
