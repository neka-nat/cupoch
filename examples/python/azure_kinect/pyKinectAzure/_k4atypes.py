import ctypes

# K4A_DECLARE_HANDLE(k4a_device_t);
class _handle_k4a_device_t(ctypes.Structure):
    _fields_ = [
        ("_rsvd", ctypes.c_size_t),
    ]


k4a_device_t = ctypes.POINTER(_handle_k4a_device_t)

# K4A_DECLARE_HANDLE(k4a_capture_t);
class _handle_k4a_capture_t(ctypes.Structure):
    _fields_ = [
        ("_rsvd", ctypes.c_size_t),
    ]


k4a_capture_t = ctypes.POINTER(_handle_k4a_capture_t)

# K4A_DECLARE_HANDLE(k4a_image_t);
class _handle_k4a_image_t(ctypes.Structure):
    _fields_ = [
        ("_rsvd", ctypes.c_size_t),
    ]


k4a_image_t = ctypes.POINTER(_handle_k4a_image_t)

# K4A_DECLARE_HANDLE(k4a_transformation_t);
class _handle_k4a_transformation_t(ctypes.Structure):
    _fields_ = [
        ("_rsvd", ctypes.c_size_t),
    ]


k4a_transformation_t = ctypes.POINTER(_handle_k4a_transformation_t)

# class k4a_result_t(CtypeIntEnum):
k4a_result_t = ctypes.c_int
K4A_RESULT_SUCCEEDED = 0
K4A_RESULT_FAILED = 1

# class k4a_buffer_result_t(CtypeIntEnum):
k4a_buffer_result_t = ctypes.c_int
K4A_BUFFER_RESULT_SUCCEEDED = 0
K4A_BUFFER_RESULT_FAILED = 1
K4A_BUFFER_RESULT_TOO_SMALL = 2

# class k4a_wait_result_t(CtypeIntEnum):
k4a_wait_result_t = ctypes.c_int
K4A_WAIT_RESULT_SUCCEEDED = 0
K4A_WAIT_RESULT_FAILED = 1
K4A_WAIT_RESULT_TIMEOUT = 2

# class k4a_log_level_t(CtypeIntEnum):
k4a_log_level_t = ctypes.c_int
K4A_LOG_LEVEL_CRITICAL = 0
K4A_LOG_LEVEL_ERROR = 1
K4A_LOG_LEVEL_WARNING = 2
K4A_LOG_LEVEL_INFO = 3
K4A_LOG_LEVEL_TRACE = 4
K4A_LOG_LEVEL_OFF = 5

# class k4a_depth_mode_t(CtypeIntEnum):
k4a_depth_mode_t = ctypes.c_int
K4A_DEPTH_MODE_OFF = 0
K4A_DEPTH_MODE_NFOV_2X2BINNED = 1
K4A_DEPTH_MODE_NFOV_UNBINNED = 2
K4A_DEPTH_MODE_WFOV_2X2BINNED = 3
K4A_DEPTH_MODE_WFOV_UNBINNED = 4
K4A_DEPTH_MODE_PASSIVE_IR = 5

# class k4a_color_resolution_t(CtypeIntEnum):
k4a_color_resolution_t = ctypes.c_int
K4A_COLOR_RESOLUTION_OFF = 0
K4A_COLOR_RESOLUTION_720P = 1
K4A_COLOR_RESOLUTION_1080P = 2
K4A_COLOR_RESOLUTION_1440P = 3
K4A_COLOR_RESOLUTION_1536P = 4
K4A_COLOR_RESOLUTION_2160P = 5
K4A_COLOR_RESOLUTION_3072P = 6

# class k4a_image_format_t(CtypeIntEnum):
k4a_image_format_t = ctypes.c_int
K4A_IMAGE_FORMAT_COLOR_MJPG = 0
K4A_IMAGE_FORMAT_COLOR_NV12 = 1
K4A_IMAGE_FORMAT_COLOR_YUY2 = 2
K4A_IMAGE_FORMAT_COLOR_BGRA32 = 3
K4A_IMAGE_FORMAT_DEPTH16 = 4
K4A_IMAGE_FORMAT_IR16 = 5
K4A_IMAGE_FORMAT_CUSTOM8 = 6
K4A_IMAGE_FORMAT_CUSTOM16 = 7
K4A_IMAGE_FORMAT_CUSTOM = 8

# class k4a_transformation_interpolation_type_t(CtypeIntEnum):
k4a_transformation_interpolation_type_t = ctypes.c_int
K4A_TRANSFORMATION_INTERPOLATION_TYPE_NEAREST = 0
K4A_TRANSFORMATION_INTERPOLATION_TYPE_LINEAR = 1

# class k4a_fps_t(CtypeIntEnum):
k4a_fps_t = ctypes.c_int
K4A_FRAMES_PER_SECOND_5 = 0
K4A_FRAMES_PER_SECOND_15 = 1
K4A_FRAMES_PER_SECOND_30 = 2

# class k4a_color_control_command_t(CtypeIntEnum):
k4a_color_control_command_t = ctypes.c_int
K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE = 0
K4A_COLOR_CONTROL_AUTO_EXPOSURE_PRIORITY = 1
K4A_COLOR_CONTROL_BRIGHTNESS = 2
K4A_COLOR_CONTROL_CONTRAST = 3
K4A_COLOR_CONTROL_SATURATION = 4
K4A_COLOR_CONTROL_SHARPNESS = 5
K4A_COLOR_CONTROL_WHITEBALANCE = 6
K4A_COLOR_CONTROL_BACKLIGHT_COMPENSATION = 7
K4A_COLOR_CONTROL_GAIN = 8
K4A_COLOR_CONTROL_POWERLINE_FREQUENCY = 9

# class k4a_color_control_mode_t(CtypeIntEnum):
k4a_color_control_mode_t = ctypes.c_int
K4A_COLOR_CONTROL_MODE_AUTO = 0
K4A_COLOR_CONTROL_MODE_MANUAL = 1


# class k4a_wired_sync_mode_t(CtypeIntEnum):
k4a_wired_sync_mode_t = ctypes.c_int
K4A_WIRED_SYNC_MODE_STANDALONE = 0
K4A_WIRED_SYNC_MODE_MASTER = 1
K4A_WIRED_SYNC_MODE_SUBORDINATE = 2

# class k4a_calibration_type_t(CtypeIntEnum):
k4a_calibration_type_t = ctypes.c_int
K4A_CALIBRATION_TYPE_UNKNOWN = -1
K4A_CALIBRATION_TYPE_DEPTH = 0
K4A_CALIBRATION_TYPE_COLOR = 1
K4A_CALIBRATION_TYPE_GYRO = 2
K4A_CALIBRATION_TYPE_ACCEL = 3
K4A_CALIBRATION_TYPE_NUM = 4

# class k4a_calibration_model_type_t(CtypeIntEnum):
k4a_calibration_model_type_t = ctypes.c_int
K4A_CALIBRATION_LENS_DISTORTION_MODEL_UNKNOWN = 0
K4A_CALIBRATION_LENS_DISTORTION_MODEL_THETA = 1
K4A_CALIBRATION_LENS_DISTORTION_MODEL_POLYNOMIAL_3K = 2
K4A_CALIBRATION_LENS_DISTORTION_MODEL_RATIONAL_6KT = 3
K4A_CALIBRATION_LENS_DISTORTION_MODEL_BROWN_CONRADY = 4

# class k4a_firmware_build_t(CtypeIntEnum):
k4a_firmware_build_t = ctypes.c_int
K4A_FIRMWARE_BUILD_RELEASE = 0
K4A_FIRMWARE_BUILD_DEBUG = 1

# class k4a_firmware_signature_t(CtypeIntEnum):
k4a_firmware_signature_t = ctypes.c_int
K4A_FIRMWARE_SIGNATURE_MSFT = 0
K4A_FIRMWARE_SIGNATURE_TEST = 1
K4A_FIRMWARE_SIGNATURE_UNSIGNED = 2

# define K4A_SUCCEEDED(_result_) (_result_ == K4A_RESULT_SUCCEEDED)
def K4A_SUCCEEDED(result):
    return result == K4A_RESULT_SUCCEEDED


# define K4A_FAILED(_result_) (!K4A_SUCCEEDED(_result_))
def K4A_FAILED(result):
    return not K4A_SUCCEEDED(result)


"""
typedef void(k4a_logging_message_cb_t)(void *context,
									   k4a_log_level_t level,
									   const char *file,
									   const int line,
									   const char *message);

typedef void(k4a_memory_destroy_cb_t)(void *buffer, void *context);

typedef uint8_t *(k4a_memory_allocate_cb_t)(int size, void **context);
"""


class _k4a_device_configuration_t(ctypes.Structure):
    _fields_ = [
        ("color_format", ctypes.c_int),
        ("color_resolution", ctypes.c_int),
        ("depth_mode", ctypes.c_int),
        ("camera_fps", ctypes.c_int),
        ("synchronized_images_only", ctypes.c_bool),
        ("depth_delay_off_color_usec", ctypes.c_int32),
        ("wired_sync_mode", ctypes.c_int),
        ("subordinate_delay_off_master_usec", ctypes.c_uint32),
        ("disable_streaming_indicator", ctypes.c_bool),
    ]


k4a_device_configuration_t = _k4a_device_configuration_t


class _k4a_calibration_extrinsics_t(ctypes.Structure):
    _fields_ = [
        ("rotation", ctypes.c_float * 9),
        ("translation", ctypes.c_float * 3),
    ]


k4a_calibration_extrinsics_t = _k4a_calibration_extrinsics_t


class _param(ctypes.Structure):
    _fields_ = [
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("k1", ctypes.c_float),
        ("k2", ctypes.c_float),
        ("k3", ctypes.c_float),
        ("k4", ctypes.c_float),
        ("k5", ctypes.c_float),
        ("k6", ctypes.c_float),
        ("codx", ctypes.c_float),
        ("cody", ctypes.c_float),
        ("p2", ctypes.c_float),
        ("p1", ctypes.c_float),
        ("metric_radius", ctypes.c_float),
    ]


class k4a_calibration_intrinsic_parameters_t(ctypes.Union):
    _fields_ = [
        ("param", _param),
        ("v", ctypes.c_float * 15),
    ]


class _k4a_calibration_intrinsics_t(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("parameter_count", ctypes.c_uint),
        ("parameters", k4a_calibration_intrinsic_parameters_t),
    ]


k4a_calibration_intrinsics_t = _k4a_calibration_intrinsics_t


class _k4a_calibration_camera_t(ctypes.Structure):
    _fields_ = [
        ("extrinsics", k4a_calibration_extrinsics_t),
        ("intrinsics", k4a_calibration_intrinsics_t),
        ("resolution_width", ctypes.c_int),
        ("resolution_height", ctypes.c_int),
        ("metric_radius", ctypes.c_float),
    ]


k4a_calibration_camera_t = _k4a_calibration_camera_t


class _k4a_calibration_t(ctypes.Structure):
    _fields_ = [
        ("depth_camera_calibration", k4a_calibration_camera_t),
        ("color_camera_calibration", k4a_calibration_camera_t),
        ("extrinsics", (k4a_calibration_extrinsics_t * K4A_CALIBRATION_TYPE_NUM) * K4A_CALIBRATION_TYPE_NUM),
        ("depth_mode", ctypes.c_int),
        ("color_resolution", ctypes.c_int),
    ]


k4a_calibration_t = _k4a_calibration_t


class _k4a_version_t(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("iteration", ctypes.c_uint32),
    ]


k4a_version_t = _k4a_version_t


class _k4a_hardware_version_t(ctypes.Structure):
    _fields_ = [
        ("rgb", k4a_version_t),
        ("depth", k4a_version_t),
        ("audio", k4a_version_t),
        ("depth_sensor", k4a_version_t),
        ("firmware_build", ctypes.c_int),
        ("firmware_signature", ctypes.c_int),
    ]


k4a_hardware_version_t = _k4a_hardware_version_t


class _xy(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
    ]


class k4a_float2_t(ctypes.Union):
    _fields_ = [("xy", _xy), ("v", ctypes.c_float * 2)]


class _xyz(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class k4a_float3_t(ctypes.Union):
    _fields_ = [("xyz", _xyz), ("v", ctypes.c_float * 3)]


class k4a_imu_sample_t(ctypes.Structure):
    _fields_ = [
        ("temperature", ctypes.c_float),
        ("acc_sample", k4a_float3_t),
        ("acc_timestamp_usec", ctypes.c_uint64),
        ("gyro_sample", k4a_float3_t),
        ("gyro_timestamp_usec", ctypes.c_uint64),
    ]


IMU_SAMPLE_SIZE = ctypes.sizeof(k4a_imu_sample_t)


K4A_DEVICE_DEFAULT = 0
K4A_WAIT_INFINITE = -1
