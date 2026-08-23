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

class TrackingState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRACKING_STATE_UNSPECIFIED: _ClassVar[TrackingState]
    TRACKING_STATE_READY: _ClassVar[TrackingState]
    TRACKING_STATE_HELD: _ClassVar[TrackingState]
    TRACKING_STATE_LOST: _ClassVar[TrackingState]
    TRACKING_STATE_RECOVERING: _ClassVar[TrackingState]
    TRACKING_STATE_CONFIRM_REQUIRED: _ClassVar[TrackingState]
    TRACKING_STATE_TRANSIENT_LOSS: _ClassVar[TrackingState]
    TRACKING_STATE_REFERENCE_PENDING: _ClassVar[TrackingState]
    TRACKING_STATE_POSE_DISCONTINUITY: _ClassVar[TrackingState]
    TRACKING_STATE_POSE_CONFIRM_REQUIRED: _ClassVar[TrackingState]
    TRACKING_STATE_MAP_CHANGED: _ClassVar[TrackingState]

class TrackingReadinessState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRACKING_READINESS_STATE_UNSPECIFIED: _ClassVar[TrackingReadinessState]
    TRACKING_READINESS_STATE_NOT_APPLICABLE: _ClassVar[TrackingReadinessState]
    TRACKING_READINESS_STATE_STARTING: _ClassVar[TrackingReadinessState]
    TRACKING_READINESS_STATE_WAITING_LIGHTHOUSE: _ClassVar[TrackingReadinessState]
    TRACKING_READINESS_STATE_SOLVING_GLOBAL_SCENE: _ClassVar[TrackingReadinessState]
    TRACKING_READINESS_STATE_VERIFYING_STABILITY: _ClassVar[TrackingReadinessState]
    TRACKING_READINESS_STATE_READY: _ClassVar[TrackingReadinessState]
    TRACKING_READINESS_STATE_LOST: _ClassVar[TrackingReadinessState]
    TRACKING_READINESS_STATE_MAP_CHANGED: _ClassVar[TrackingReadinessState]
    TRACKING_READINESS_STATE_ERROR: _ClassVar[TrackingReadinessState]

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
WATCH_DOG_LEVEL_A: WatchDogLevel
WATCH_DOG_LEVEL_B: WatchDogLevel
WATCH_DOG_LEVEL_C: WatchDogLevel
ENCODING_UNSPECIFIED: Encoding
RAW: Encoding
JPEG: Encoding
PNG: Encoding
H264: Encoding
FRAME_QUALITY_UNSPECIFIED: FrameQuality
FRAME_QUALITY_GOOD: FrameQuality
FRAME_QUALITY_DEGRADED: FrameQuality
FRAME_QUALITY_STALE: FrameQuality
TRACKING_STATE_UNSPECIFIED: TrackingState
TRACKING_STATE_READY: TrackingState
TRACKING_STATE_HELD: TrackingState
TRACKING_STATE_LOST: TrackingState
TRACKING_STATE_RECOVERING: TrackingState
TRACKING_STATE_CONFIRM_REQUIRED: TrackingState
TRACKING_STATE_TRANSIENT_LOSS: TrackingState
TRACKING_STATE_REFERENCE_PENDING: TrackingState
TRACKING_STATE_POSE_DISCONTINUITY: TrackingState
TRACKING_STATE_POSE_CONFIRM_REQUIRED: TrackingState
TRACKING_STATE_MAP_CHANGED: TrackingState
TRACKING_READINESS_STATE_UNSPECIFIED: TrackingReadinessState
TRACKING_READINESS_STATE_NOT_APPLICABLE: TrackingReadinessState
TRACKING_READINESS_STATE_STARTING: TrackingReadinessState
TRACKING_READINESS_STATE_WAITING_LIGHTHOUSE: TrackingReadinessState
TRACKING_READINESS_STATE_SOLVING_GLOBAL_SCENE: TrackingReadinessState
TRACKING_READINESS_STATE_VERIFYING_STABILITY: TrackingReadinessState
TRACKING_READINESS_STATE_READY: TrackingReadinessState
TRACKING_READINESS_STATE_LOST: TrackingReadinessState
TRACKING_READINESS_STATE_MAP_CHANGED: TrackingReadinessState
TRACKING_READINESS_STATE_ERROR: TrackingReadinessState
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
    __slots__ = ("key", "data", "produce_ts", "quality", "tracking_state")
    KEY_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    PRODUCE_TS_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    TRACKING_STATE_FIELD_NUMBER: _ClassVar[int]
    key: str
    data: bytes
    produce_ts: _timestamp_pb2.Timestamp
    quality: FrameQuality
    tracking_state: TrackingState
    def __init__(self, key: _Optional[str] = ..., data: _Optional[bytes] = ..., produce_ts: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., quality: _Optional[_Union[FrameQuality, str]] = ..., tracking_state: _Optional[_Union[TrackingState, str]] = ...) -> None: ...

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

class ActionResult(_message.Message):
    __slots__ = ("features", "safety")
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    SAFETY_FIELD_NUMBER: _ClassVar[int]
    features: _containers.RepeatedCompositeFieldContainer[OneFeature]
    safety: SafetyReport
    def __init__(self, features: _Optional[_Iterable[_Union[OneFeature, _Mapping]]] = ..., safety: _Optional[_Union[SafetyReport, _Mapping]] = ...) -> None: ...

class SafetyReport(_message.Message):
    __slots__ = ("flags", "applied_mask", "reason", "collision_pair_a", "collision_pair_b", "min_distance_m", "pos_err_m", "rot_err_rad", "manipulability", "reject_streak", "reject_duration_s")
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    APPLIED_MASK_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    COLLISION_PAIR_A_FIELD_NUMBER: _ClassVar[int]
    COLLISION_PAIR_B_FIELD_NUMBER: _ClassVar[int]
    MIN_DISTANCE_M_FIELD_NUMBER: _ClassVar[int]
    POS_ERR_M_FIELD_NUMBER: _ClassVar[int]
    ROT_ERR_RAD_FIELD_NUMBER: _ClassVar[int]
    MANIPULABILITY_FIELD_NUMBER: _ClassVar[int]
    REJECT_STREAK_FIELD_NUMBER: _ClassVar[int]
    REJECT_DURATION_S_FIELD_NUMBER: _ClassVar[int]
    flags: int
    applied_mask: int
    reason: str
    collision_pair_a: str
    collision_pair_b: str
    min_distance_m: float
    pos_err_m: float
    rot_err_rad: float
    manipulability: float
    reject_streak: int
    reject_duration_s: float
    def __init__(self, flags: _Optional[int] = ..., applied_mask: _Optional[int] = ..., reason: _Optional[str] = ..., collision_pair_a: _Optional[str] = ..., collision_pair_b: _Optional[str] = ..., min_distance_m: _Optional[float] = ..., pos_err_m: _Optional[float] = ..., rot_err_rad: _Optional[float] = ..., manipulability: _Optional[float] = ..., reject_streak: _Optional[int] = ..., reject_duration_s: _Optional[float] = ...) -> None: ...

class HoldRequest(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class HoldResponse(_message.Message):
    __slots__ = ("held", "hold_epoch")
    HELD_FIELD_NUMBER: _ClassVar[int]
    HOLD_EPOCH_FIELD_NUMBER: _ClassVar[int]
    held: bool
    hold_epoch: int
    def __init__(self, held: _Optional[bool] = ..., hold_epoch: _Optional[int] = ...) -> None: ...

class ResumeRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResumeResponse(_message.Message):
    __slots__ = ("resumed", "hold_epoch")
    RESUMED_FIELD_NUMBER: _ClassVar[int]
    HOLD_EPOCH_FIELD_NUMBER: _ClassVar[int]
    resumed: bool
    hold_epoch: int
    def __init__(self, resumed: _Optional[bool] = ..., hold_epoch: _Optional[int] = ...) -> None: ...

class TrackingReadiness(_message.Message):
    __slots__ = ("state", "reason", "context_epoch", "global_scene_generation", "lighthouse_cohort_generation", "readiness_generation", "expected_lighthouses", "solved_lighthouses", "token", "stable_sample_count", "stable_window_s", "position_spread_m", "rotation_spread_rad")
    STATE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_EPOCH_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_SCENE_GENERATION_FIELD_NUMBER: _ClassVar[int]
    LIGHTHOUSE_COHORT_GENERATION_FIELD_NUMBER: _ClassVar[int]
    READINESS_GENERATION_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_LIGHTHOUSES_FIELD_NUMBER: _ClassVar[int]
    SOLVED_LIGHTHOUSES_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    STABLE_SAMPLE_COUNT_FIELD_NUMBER: _ClassVar[int]
    STABLE_WINDOW_S_FIELD_NUMBER: _ClassVar[int]
    POSITION_SPREAD_M_FIELD_NUMBER: _ClassVar[int]
    ROTATION_SPREAD_RAD_FIELD_NUMBER: _ClassVar[int]
    state: TrackingReadinessState
    reason: str
    context_epoch: int
    global_scene_generation: int
    lighthouse_cohort_generation: int
    readiness_generation: int
    expected_lighthouses: _containers.RepeatedScalarFieldContainer[str]
    solved_lighthouses: _containers.RepeatedScalarFieldContainer[str]
    token: str
    stable_sample_count: int
    stable_window_s: float
    position_spread_m: float
    rotation_spread_rad: float
    def __init__(self, state: _Optional[_Union[TrackingReadinessState, str]] = ..., reason: _Optional[str] = ..., context_epoch: _Optional[int] = ..., global_scene_generation: _Optional[int] = ..., lighthouse_cohort_generation: _Optional[int] = ..., readiness_generation: _Optional[int] = ..., expected_lighthouses: _Optional[_Iterable[str]] = ..., solved_lighthouses: _Optional[_Iterable[str]] = ..., token: _Optional[str] = ..., stable_sample_count: _Optional[int] = ..., stable_window_s: _Optional[float] = ..., position_spread_m: _Optional[float] = ..., rotation_spread_rad: _Optional[float] = ...) -> None: ...

class SetReferenceRequest(_message.Message):
    __slots__ = ("readiness_token",)
    READINESS_TOKEN_FIELD_NUMBER: _ClassVar[int]
    readiness_token: str
    def __init__(self, readiness_token: _Optional[str] = ...) -> None: ...

class GetInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetInfoResponse(_message.Message):
    __slots__ = ("observation_features", "action_features", "feedback_features", "effective_action_features")
    OBSERVATION_FEATURES_FIELD_NUMBER: _ClassVar[int]
    ACTION_FEATURES_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_FEATURES_FIELD_NUMBER: _ClassVar[int]
    EFFECTIVE_ACTION_FEATURES_FIELD_NUMBER: _ClassVar[int]
    observation_features: _containers.RepeatedCompositeFieldContainer[OneFeatureInfo]
    action_features: _containers.RepeatedCompositeFieldContainer[OneFeatureInfo]
    feedback_features: _containers.RepeatedCompositeFieldContainer[OneFeatureInfo]
    effective_action_features: _containers.RepeatedCompositeFieldContainer[OneFeatureInfo]
    def __init__(self, observation_features: _Optional[_Iterable[_Union[OneFeatureInfo, _Mapping]]] = ..., action_features: _Optional[_Iterable[_Union[OneFeatureInfo, _Mapping]]] = ..., feedback_features: _Optional[_Iterable[_Union[OneFeatureInfo, _Mapping]]] = ..., effective_action_features: _Optional[_Iterable[_Union[OneFeatureInfo, _Mapping]]] = ...) -> None: ...
