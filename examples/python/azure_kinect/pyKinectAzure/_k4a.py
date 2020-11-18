import ctypes
import sys
from ._k4atypes import *
import traceback

class k4a:

	def __init__(self,modulePath):
		try: 
			dll = ctypes.CDLL(modulePath)

		except Exception as e:

			if e.winerror == 193:
				print("Failed to load library. \n\nChange the module path to the 32 bit version.")
				sys.exit(1)

			print(e, "\n\nFailed to lad Windows library. Trying to load Linux library...\n")

			try:
				dll = ctypes.CDLL('k4a.so')
			except Exception as ee:
				print("Failed to load library", ee)
				sys.exit(1)

		#K4A_EXPORT uint32_t k4a_device_get_installed_count(void);
		self.k4a_device_get_installed_count = dll.k4a_device_get_installed_count
		self.k4a_device_get_installed_count.restype = ctypes.c_uint32

		"""
		K4A_EXPORT k4a_result_t k4a_set_debug_message_handler(k4a_logging_message_cb_t *message_cb,
															  void *message_cb_context,
															  k4a_log_level_t min_level);
		"""

		"""
		K4A_EXPORT k4a_result_t k4a_set_allocator(k4a_memory_allocate_cb_t allocate, k4a_memory_destroy_cb_t free);                                                                                        
		"""


		#K4A_EXPORT k4a_result_t k4a_device_open(uint32_t index, k4a_device_t *device_handle);
		self.k4a_device_open = dll.k4a_device_open
		self.k4a_device_open.restype=ctypes.c_int
		self.k4a_device_open.argtypes=(ctypes.c_uint32, ctypes.POINTER(k4a_device_t))

		#K4A_EXPORT void k4a_device_close(k4a_device_t device_handle);
		self.k4a_device_close = dll.k4a_device_close
		self.k4a_device_close.restype=None
		self.k4a_device_close.argtypes=(k4a_device_t,)

		"""
		K4A_EXPORT k4a_wait_result_t k4a_device_get_capture(k4a_device_t device_handle,
															k4a_capture_t *capture_handle,
															int32_t timeout_in_ms);
		"""
		self.k4a_device_get_capture = dll.k4a_device_get_capture
		self.k4a_device_get_capture.restype=ctypes.c_int
		self.k4a_device_get_capture.argtypes=(k4a_device_t, ctypes.POINTER(k4a_capture_t), ctypes.c_int32)

		"""
		K4A_EXPORT k4a_wait_result_t k4a_device_get_imu_sample(k4a_device_t device_handle,
															   k4a_imu_sample_t *imu_sample,
															   int32_t timeout_in_ms);
		"""
		self.k4a_device_get_imu_sample = dll.k4a_device_get_imu_sample
		self.k4a_device_get_imu_sample.restype=ctypes.c_int
		self.k4a_device_get_imu_sample.argtypes=(k4a_device_t, ctypes.POINTER(k4a_imu_sample_t), ctypes.c_int32)


		#K4A_EXPORT k4a_result_t k4a_capture_create(k4a_capture_t *capture_handle);
		self.k4a_capture_create = dll.k4a_capture_create
		self.k4a_capture_create.restype= k4a_result_t
		self.k4a_capture_create.argtypes=(ctypes.POINTER(k4a_capture_t), )

		#K4A_EXPORT void k4a_capture_release(k4a_capture_t capture_handle);
		self.k4a_capture_release = dll.k4a_capture_release
		self.k4a_capture_release.restype = None
		self.k4a_capture_release.argtypes=(k4a_capture_t,)

		#K4A_EXPORT void k4a_capture_reference(k4a_capture_t capture_handle);
		self.k4a_capture_reference = dll.k4a_capture_reference
		self.k4a_capture_reference.restype = None
		self.k4a_capture_reference.argtypes=(k4a_capture_t,)

		#K4A_EXPORT k4a_image_t k4a_capture_get_color_image(k4a_capture_t capture_handle)
		self.k4a_capture_get_color_image = dll.k4a_capture_get_color_image
		self.k4a_capture_get_color_image.restype=k4a_image_t
		self.k4a_capture_get_color_image.argtypes=(k4a_capture_t,)

		#K4A_EXPORT k4a_image_t k4a_capture_get_depth_image(k4a_capture_t capture_handle);
		self.k4a_capture_get_depth_image = dll.k4a_capture_get_depth_image
		self.k4a_capture_get_depth_image.restype=k4a_image_t
		self.k4a_capture_get_depth_image.argtypes=(k4a_capture_t,)

		#K4A_EXPORT k4a_image_t k4a_capture_get_ir_image(k4a_capture_t capture_handle);
		self.k4a_capture_get_ir_image = dll.k4a_capture_get_ir_image
		self.k4a_capture_get_ir_image.restype=k4a_image_t
		self.k4a_capture_get_ir_image.argtypes=(k4a_capture_t,)

		#K4A_EXPORT void k4a_capture_set_color_image(k4a_capture_t capture_handle, k4a_image_t image_handle);
		self.k4a_capture_set_color_image = dll.k4a_capture_set_color_image
		self.k4a_capture_set_color_image.restype=None
		self.k4a_capture_set_color_image.argtypes=(k4a_capture_t,k4a_image_t,)

		#K4A_EXPORT void k4a_capture_set_depth_image(k4a_capture_t capture_handle, k4a_image_t image_handle);
		self.k4a_capture_set_depth_image = dll.k4a_capture_set_depth_image
		self.k4a_capture_set_depth_image.restype=None
		self.k4a_capture_set_depth_image.argtypes=(k4a_capture_t,k4a_image_t,)

		#K4A_EXPORT void k4a_capture_set_ir_image(k4a_capture_t capture_handle, k4a_image_t image_handle);
		self.k4a_capture_set_ir_image = dll.k4a_capture_set_ir_image
		self.k4a_capture_set_ir_image.restype=None
		self.k4a_capture_set_ir_image.argtypes=(k4a_capture_t,k4a_image_t,)

		#K4A_EXPORT void k4a_capture_set_temperature_c(k4a_capture_t capture_handle, float temperature_c);
		self.k4a_capture_set_temperature_c = dll.k4a_capture_set_temperature_c
		self.k4a_capture_set_temperature_c.argtypes=(k4a_capture_t,k4a_image_t,)
		self.k4a_capture_set_temperature_c.argtypes=(k4a_capture_t,ctypes.c_float,)

		#K4A_EXPORT float k4a_capture_get_temperature_c(k4a_capture_t capture_handle);
		self.k4a_capture_get_temperature_c = dll.k4a_capture_get_temperature_c
		self.k4a_capture_get_temperature_c.restype=ctypes.c_float
		self.k4a_capture_get_temperature_c.argtypes=(k4a_capture_t, )

		"""
		K4A_EXPORT k4a_result_t k4a_image_create(k4a_image_format_t format,
												 int width_pixels,
												 int height_pixels,
												 int stride_bytes,
												 k4a_image_t *image_handle);
		"""
		self.k4a_image_create = dll.k4a_image_create
		self.k4a_image_create.restype=k4a_result_t
		self.k4a_image_create.argtypes=(k4a_image_format_t,\
										ctypes.c_int,\
										ctypes.c_int,\
										ctypes.c_int,\
										ctypes.POINTER(k4a_image_t),)


		"""
		K4A_EXPORT k4a_result_t k4a_image_create_from_buffer(k4a_image_format_t format,
															 int width_pixels,
															 int height_pixels,
															 int stride_bytes,
															 uint8_t *buffer,
															 size_t buffer_size,
															 k4a_memory_destroy_cb_t *buffer_release_cb,
															 void *buffer_release_cb_context,
															 k4a_image_t *image_handle);
		"""
		self.k4a_image_create_from_buffer = dll.k4a_image_create_from_buffer
		self.k4a_image_create_from_buffer.restype=k4a_result_t
		self.k4a_image_create_from_buffer.argtypes=(k4a_image_format_t,\
													ctypes.c_int,\
													ctypes.c_int,\
													ctypes.c_int,\
													ctypes.POINTER(ctypes.c_uint8),\
													ctypes.c_size_t,\
													ctypes.c_void_p,\
													ctypes.c_void_p,\
													ctypes.POINTER(k4a_image_t),)


		#K4A_EXPORT uint8_t *k4a_image_get_buffer(k4a_image_t image_handle);
		self.k4a_image_get_buffer = dll.k4a_image_get_buffer
		self.k4a_image_get_buffer.restype=ctypes.POINTER(ctypes.c_uint8)
		self.k4a_image_get_buffer.argtypes=(k4a_image_t, )

		#K4A_EXPORT size_t k4a_image_get_size(k4a_image_t image_handle);
		self.k4a_image_get_size = dll.k4a_image_get_size
		self.k4a_image_get_size.restype=ctypes.c_size_t
		self.k4a_image_get_size.argtypes=(k4a_image_t, )

		#K4A_EXPORT k4a_image_format_t k4a_image_get_format(k4a_image_t image_handle);
		self.k4a_image_get_format = dll.k4a_image_get_format
		self.k4a_image_get_format.restype=k4a_image_format_t
		self.k4a_image_get_format.argtypes=(k4a_image_t, )

		#K4A_EXPORT int k4a_image_get_width_pixels(k4a_image_t image_handle);
		self.k4a_image_get_width_pixels = dll.k4a_image_get_width_pixels
		self.k4a_image_get_width_pixels.restype=ctypes.c_int
		self.k4a_image_get_width_pixels.argtypes=(k4a_image_t, )

		#K4A_EXPORT int k4a_image_get_height_pixels(k4a_image_t image_handle);
		self.k4a_image_get_height_pixels = dll.k4a_image_get_height_pixels
		self.k4a_image_get_height_pixels.restype=ctypes.c_int
		self.k4a_image_get_height_pixels.argtypes=(k4a_image_t, )

		#K4A_EXPORT int k4a_image_get_stride_bytes(k4a_image_t image_handle);
		self.k4a_image_get_stride_bytes = dll.k4a_image_get_stride_bytes
		self.k4a_image_get_stride_bytes.restype=ctypes.c_int
		self.k4a_image_get_stride_bytes.argtypes=(k4a_image_t, )

		#K4A_DEPRECATED_EXPORT uint64_t k4a_image_get_timestamp_usec(k4a_image_t image_handle);
		self.k4a_image_get_timestamp_usec = dll.k4a_image_get_timestamp_usec
		self.k4a_image_get_timestamp_usec.restype=ctypes.c_uint64
		self.k4a_image_get_timestamp_usec.argtypes=(k4a_image_t, )

		#K4A_EXPORT uint64_t k4a_image_get_device_timestamp_usec(k4a_image_t image_handle);
		self.k4a_image_get_device_timestamp_usec = dll.k4a_image_get_device_timestamp_usec
		self.k4a_image_get_device_timestamp_usec.restype=ctypes.c_uint64
		self.k4a_image_get_device_timestamp_usec.argtypes=(k4a_image_t, )

		#K4A_EXPORT uint64_t k4a_image_get_system_timestamp_nsec(k4a_image_t image_handle);
		self.k4a_image_get_system_timestamp_nsec = dll.k4a_image_get_system_timestamp_nsec
		self.k4a_image_get_system_timestamp_nsec.restype=ctypes.c_uint64
		self.k4a_image_get_system_timestamp_nsec.argtypes=(k4a_image_t, )

		#K4A_EXPORT uint64_t k4a_image_get_exposure_usec(k4a_image_t image_handle);
		self.k4a_image_get_exposure_usec = dll.k4a_image_get_exposure_usec
		self.k4a_image_get_exposure_usec.restype=ctypes.c_uint64
		self.k4a_image_get_exposure_usec.argtypes=(k4a_image_t, )

		#K4A_EXPORT uint32_t k4a_image_get_white_balance(k4a_image_t image_handle);
		self.k4a_image_get_white_balance = dll.k4a_image_get_white_balance
		self.k4a_image_get_white_balance.restype=ctypes.c_uint32
		self.k4a_image_get_white_balance.argtypes=(k4a_image_t, )

		#K4A_EXPORT uint32_t k4a_image_get_iso_speed(k4a_image_t image_handle);
		self.k4a_image_get_iso_speed = dll.k4a_image_get_iso_speed
		self.k4a_image_get_iso_speed.restype=ctypes.c_uint32
		self.k4a_image_get_iso_speed.argtypes=(k4a_image_t, )

		#K4A_EXPORT void k4a_image_set_device_timestamp_usec(k4a_image_t image_handle, uint64_t timestamp_usec);
		self.k4a_image_set_device_timestamp_usec = dll.k4a_image_set_device_timestamp_usec
		self.k4a_image_set_device_timestamp_usec.restype=None
		self.k4a_image_set_device_timestamp_usec.argtypes=(k4a_image_t, ctypes.c_uint64,)

		#K4A_DEPRECATED_EXPORT void k4a_image_set_timestamp_usec(k4a_image_t image_handle, uint64_t timestamp_usec);
		self.k4a_image_set_timestamp_usec = dll.k4a_image_set_timestamp_usec
		self.k4a_image_set_timestamp_usec.restype=None
		self.k4a_image_set_timestamp_usec.argtypes=(k4a_image_t, ctypes.c_uint64,)

		#K4A_EXPORT void k4a_image_set_system_timestamp_nsec(k4a_image_t image_handle, uint64_t timestamp_nsec);
		self.k4a_image_set_system_timestamp_nsec = dll.k4a_image_set_system_timestamp_nsec
		self.k4a_image_set_system_timestamp_nsec.restype=None
		self.k4a_image_set_system_timestamp_nsec.argtypes=(k4a_image_t, ctypes.c_uint64,)

		#K4A_EXPORT void k4a_image_set_exposure_usec(k4a_image_t image_handle, uint64_t exposure_usec);
		self.k4a_image_set_exposure_usec = dll.k4a_image_set_exposure_usec
		self.k4a_image_set_exposure_usec.restype=None
		self.k4a_image_set_exposure_usec.argtypes=(k4a_image_t, ctypes.c_uint64,)

		#K4A_DEPRECATED_EXPORT void k4a_image_set_exposure_time_usec(k4a_image_t image_handle, uint64_t exposure_usec);
		self.k4a_image_set_exposure_time_usec = dll.k4a_image_set_exposure_time_usec
		self.k4a_image_set_exposure_time_usec.restype=None
		self.k4a_image_set_exposure_time_usec.argtypes=(k4a_image_t, ctypes.c_uint64,)

		#K4A_EXPORT void k4a_image_set_white_balance(k4a_image_t image_handle, uint32_t white_balance);
		self.k4a_image_set_white_balance = dll.k4a_image_set_white_balance
		self.k4a_image_set_white_balance.restype=None
		self.k4a_image_set_white_balance.argtypes=(k4a_image_t, ctypes.c_uint32,)

		#K4A_EXPORT void k4a_image_set_iso_speed(k4a_image_t image_handle, uint32_t iso_speed);
		self.k4a_image_set_iso_speed = dll.k4a_image_set_iso_speed
		self.k4a_image_set_iso_speed.restype=None
		self.k4a_image_set_iso_speed.argtypes=(k4a_image_t, ctypes.c_uint32,)

		#K4A_EXPORT void k4a_image_reference(k4a_image_t image_handle);
		self.k4a_image_reference = dll.k4a_image_reference
		self.k4a_image_reference.restype=None
		self.k4a_image_reference.argtypes=(k4a_image_t,)

		#K4A_EXPORT void k4a_image_release(k4a_image_t image_handle);
		self.k4a_image_release = dll.k4a_image_release
		self.k4a_image_release.restype=None
		self.k4a_image_release.argtypes=(k4a_image_t,)

		#K4A_EXPORT k4a_result_t k4a_device_start_cameras(k4a_device_t device_handle, const k4a_device_configuration_t *config);
		self.k4a_device_start_cameras = dll.k4a_device_start_cameras
		self.k4a_device_start_cameras.restype=k4a_result_t
		self.k4a_device_start_cameras.argtypes=(k4a_device_t, ctypes.POINTER(k4a_device_configuration_t),)

		#K4A_EXPORT void k4a_device_stop_cameras(k4a_device_t device_handle);
		self.k4a_device_stop_cameras = dll.k4a_device_stop_cameras
		self.k4a_device_stop_cameras.restype=None
		self.k4a_device_stop_cameras.argtypes=(k4a_device_t,)

		#K4A_EXPORT k4a_result_t k4a_device_start_imu(k4a_device_t device_handle);
		self.k4a_device_start_imu = dll.k4a_device_start_imu
		self.k4a_device_start_imu.restype=k4a_result_t
		self.k4a_device_start_imu.argtypes=(k4a_device_t,)

		#K4A_EXPORT void k4a_device_stop_imu(k4a_device_t device_handle);
		self.k4a_device_stop_imu = dll.k4a_device_stop_imu
		self.k4a_device_stop_imu.restype=None
		self.k4a_device_stop_imu.argtypes=(k4a_device_t,)

		"""
		K4A_EXPORT k4a_buffer_result_t k4a_device_get_serialnum(k4a_device_t device_handle,
																char *serial_number,
                                                       			size_t *serial_number_size);
        """
		self.k4a_device_get_serialnum = dll.k4a_device_get_serialnum
		self.k4a_device_get_serialnum.restype=k4a_buffer_result_t
		self.k4a_device_get_serialnum.argtypes=(k4a_device_t,ctypes.c_char_p,ctypes.POINTER(ctypes.c_size_t))

		#K4A_EXPORT k4a_result_t k4a_device_get_version(k4a_device_t device_handle, k4a_hardware_version_t *version);
		self.k4a_device_get_version = dll.k4a_device_get_version
		self.k4a_device_get_version.restype=k4a_result_t
		self.k4a_device_get_version.argtypes=(k4a_device_t,ctypes.POINTER(k4a_hardware_version_t),)

		"""
		K4A_EXPORT k4a_result_t k4a_device_get_color_control_capabilities(k4a_device_t device_handle,
																		  k4a_color_control_command_t command,
																		  bool *supports_auto,
																		  int32_t *min_value,
																		  int32_t *max_value,
																		  int32_t *step_value,
																		  int32_t *default_value,
																		  k4a_color_control_mode_t *default_mode);
		"""
		self.k4a_device_get_color_control_capabilities = dll.k4a_device_get_color_control_capabilities
		self.k4a_device_get_color_control_capabilities.restype=k4a_result_t
		self.k4a_device_get_color_control_capabilities.argtypes=(k4a_device_t,\
																k4a_color_control_command_t,\
																ctypes.POINTER(ctypes.c_bool),\
																ctypes.POINTER(ctypes.c_int32),\
																ctypes.POINTER(ctypes.c_int32),\
																ctypes.POINTER(ctypes.c_int32),\
																ctypes.POINTER(ctypes.c_int32),\
																ctypes.POINTER(k4a_color_control_mode_t),\
																)

		"""
		K4A_EXPORT k4a_result_t k4a_device_get_color_control(k4a_device_t device_handle,
															 k4a_color_control_command_t command,
															 k4a_color_control_mode_t *mode,
															 int32_t *value);
		"""
		self.k4a_device_get_color_control = dll.k4a_device_get_color_control
		self.k4a_device_get_color_control.restype=k4a_result_t
		self.k4a_device_get_color_control.argtypes=(k4a_device_t,\
													k4a_color_control_command_t,\
													ctypes.POINTER(k4a_color_control_mode_t),\
													ctypes.POINTER(ctypes.c_int32),\
													)

		"""
		K4A_EXPORT k4a_result_t k4a_device_set_color_control(k4a_device_t device_handle,
															 k4a_color_control_command_t command,
															 k4a_color_control_mode_t mode,
															 int32_t value);
		"""
		self.k4a_device_set_color_control = dll.k4a_device_set_color_control
		self.k4a_device_set_color_control.restype=k4a_result_t
		self.k4a_device_set_color_control.argtypes=(k4a_device_t,\
													k4a_color_control_command_t,\
													k4a_color_control_mode_t,\
													ctypes.c_int32,\
													)

		"""
		K4A_EXPORT k4a_buffer_result_t k4a_device_get_raw_calibration(k4a_device_t device_handle,
																	  uint8_t *data,
																	  size_t *data_size);
		"""
		self.k4a_device_get_raw_calibration = dll.k4a_device_get_raw_calibration
		self.k4a_device_get_raw_calibration.restype=k4a_buffer_result_t
		self.k4a_device_get_raw_calibration.argtypes=(k4a_device_t,\
													  ctypes.POINTER(ctypes.c_uint8),\
													  ctypes.POINTER(ctypes.c_size_t),\
													  )

		"""
		K4A_EXPORT k4a_result_t k4a_device_get_calibration(k4a_device_t device_handle,
														   const k4a_depth_mode_t depth_mode,
														   const k4a_color_resolution_t color_resolution,
														   k4a_calibration_t *calibration);
		"""
		self.k4a_device_get_calibration = dll.k4a_device_get_calibration
		self.k4a_device_get_calibration.restype=k4a_result_t
		self.k4a_device_get_calibration.argtypes=(k4a_device_t, \
												  k4a_depth_mode_t, \
												  k4a_color_resolution_t, \
												  ctypes.POINTER(k4a_calibration_t),\
												  )

		"""
		K4A_EXPORT k4a_result_t k4a_device_get_sync_jack(k4a_device_t device_handle,
														 bool *sync_in_jack_connected,
														 bool *sync_out_jack_connected);
		"""
		self.k4a_device_get_sync_jack = dll.k4a_device_get_sync_jack
		self.k4a_device_get_sync_jack.restype=k4a_result_t
		self.k4a_device_get_sync_jack.argtypes=(k4a_device_t, \
												ctypes.POINTER(ctypes.c_bool),\
												ctypes.POINTER(ctypes.c_bool),\
												)

		"""
		K4A_EXPORT k4a_result_t k4a_calibration_get_from_raw(char *raw_calibration,
															 size_t raw_calibration_size,
															 const k4a_depth_mode_t depth_mode,
															 const k4a_color_resolution_t color_resolution,
															 k4a_calibration_t *calibration);
		"""
		self.k4a_calibration_get_from_raw = dll.k4a_calibration_get_from_raw
		self.k4a_calibration_get_from_raw.restype=k4a_result_t
		self.k4a_calibration_get_from_raw.argtypes=(ctypes.POINTER(ctypes.c_char), \
													ctypes.c_size_t,\
													k4a_depth_mode_t,\
													k4a_color_resolution_t,\
													ctypes.POINTER(k4a_calibration_t),\
													)

		"""
		K4A_EXPORT k4a_result_t k4a_calibration_3d_to_3d(const k4a_calibration_t *calibration,
														 const k4a_float3_t *source_point3d_mm,
														 const k4a_calibration_type_t source_camera,
														 const k4a_calibration_type_t target_camera,
														 k4a_float3_t *target_point3d_mm);
		"""
		self.k4a_calibration_3d_to_3d = dll.k4a_calibration_3d_to_3d
		self.k4a_calibration_3d_to_3d.restype=k4a_result_t
		self.k4a_calibration_3d_to_3d.argtypes=(ctypes.POINTER(k4a_calibration_t), \
												ctypes.POINTER(k4a_float3_t), \
												k4a_calibration_type_t,\
												k4a_calibration_type_t,\
												ctypes.POINTER(k4a_float3_t),\
												)

		"""
		K4A_EXPORT k4a_result_t k4a_calibration_2d_to_3d(const k4a_calibration_t *calibration,
														 const k4a_float2_t *source_point2d,
														 const float source_depth_mm,
														 const k4a_calibration_type_t source_camera,
														 const k4a_calibration_type_t target_camera,
														 k4a_float3_t *target_point3d_mm,
														 int *valid);
		"""
		self.k4a_calibration_2d_to_3d = dll.k4a_calibration_2d_to_3d
		self.k4a_calibration_2d_to_3d.restype=k4a_result_t
		self.k4a_calibration_2d_to_3d.argtypes=(ctypes.POINTER(k4a_calibration_t), \
												ctypes.POINTER(k4a_float2_t), \
												ctypes.c_float,\
												k4a_calibration_type_t,\
												k4a_calibration_type_t,\
												ctypes.POINTER(k4a_float3_t),\
												ctypes.POINTER(ctypes.c_int),\
												)

		"""
		K4A_EXPORT k4a_result_t k4a_calibration_3d_to_2d(const k4a_calibration_t *calibration,
														 const k4a_float3_t *source_point3d_mm,
														 const k4a_calibration_type_t source_camera,
														 const k4a_calibration_type_t target_camera,
														 k4a_float2_t *target_point2d,
														 int *valid);
		"""
		self.k4a_calibration_3d_to_2d = dll.k4a_calibration_3d_to_2d
		self.k4a_calibration_3d_to_2d.restype=k4a_result_t
		self.k4a_calibration_3d_to_2d.argtypes=(ctypes.POINTER(k4a_calibration_t), \
												ctypes.POINTER(k4a_float3_t), \
												k4a_calibration_type_t,\
												k4a_calibration_type_t,\
												ctypes.POINTER(k4a_float2_t),\
												ctypes.POINTER(ctypes.c_int),\
												)

		"""
		K4A_EXPORT k4a_result_t k4a_calibration_2d_to_2d(const k4a_calibration_t *calibration,
														 const k4a_float2_t *source_point2d,
														 const float source_depth_mm,
														 const k4a_calibration_type_t source_camera,
														 const k4a_calibration_type_t target_camera,
														 k4a_float2_t *target_point2d,
														 int *valid);
		"""
		self.k4a_calibration_2d_to_2d = dll.k4a_calibration_2d_to_2d
		self.k4a_calibration_2d_to_2d.restype=k4a_result_t
		self.k4a_calibration_2d_to_2d.argtypes=(ctypes.POINTER(k4a_calibration_t), \
												ctypes.POINTER(k4a_float2_t), \
												ctypes.c_float,\
												k4a_calibration_type_t,\
												k4a_calibration_type_t,\
												ctypes.POINTER(k4a_float2_t),\
												ctypes.POINTER(ctypes.c_int),\
												)

		"""
		K4A_EXPORT k4a_result_t k4a_calibration_color_2d_to_depth_2d(const k4a_calibration_t *calibration,
																	 const k4a_float2_t *source_point2d,
																	 const k4a_image_t depth_image,
																	 k4a_float2_t *target_point2d,
																	 int *valid);
		"""
		self.k4a_calibration_color_2d_to_depth_2d = dll.k4a_calibration_color_2d_to_depth_2d
		self.k4a_calibration_color_2d_to_depth_2d.restype=k4a_result_t
		self.k4a_calibration_color_2d_to_depth_2d.argtypes=(ctypes.POINTER(k4a_calibration_t), \
															ctypes.POINTER(k4a_float2_t), \
															k4a_image_t,\
															ctypes.POINTER(k4a_float2_t),\
															ctypes.POINTER(ctypes.c_int),\
															)

		#K4A_EXPORT k4a_transformation_t k4a_transformation_create(const k4a_calibration_t *calibration);
		self.k4a_transformation_create = dll.k4a_transformation_create
		self.k4a_transformation_create.restype=k4a_transformation_t
		self.k4a_transformation_create.argtypes=(ctypes.POINTER(k4a_calibration_t),)

		#K4A_EXPORT void k4a_transformation_destroy(k4a_transformation_t transformation_handle);
		self.k4a_transformation_destroy = dll.k4a_transformation_destroy
		self.k4a_transformation_destroy.restype=k4a_transformation_t
		self.k4a_transformation_destroy.argtypes=(k4a_transformation_t,)

		"""
		K4A_EXPORT k4a_result_t k4a_transformation_depth_image_to_color_camera(k4a_transformation_t transformation_handle,
																			   const k4a_image_t depth_image,
																			   k4a_image_t transformed_depth_image);
		"""
		self.k4a_transformation_depth_image_to_color_camera = dll.k4a_transformation_depth_image_to_color_camera
		self.k4a_transformation_depth_image_to_color_camera.restype=k4a_result_t
		self.k4a_transformation_depth_image_to_color_camera.argtypes=(k4a_transformation_t, \
																	  k4a_image_t,\
																	  k4a_image_t,\
																	  )

		"""
		K4A_EXPORT k4a_result_t k4a_transformation_depth_image_to_color_camera_custom(k4a_transformation_t transformation_handle,
															  const k4a_image_t depth_image,
															  const k4a_image_t custom_image,
															  k4a_image_t transformed_depth_image,
															  k4a_image_t transformed_custom_image,
															  k4a_transformation_interpolation_type_t interpolation_type,
															  uint32_t invalid_custom_value);
		"""
		self.k4a_transformation_depth_image_to_color_camera_custom = dll.k4a_transformation_depth_image_to_color_camera_custom
		self.k4a_transformation_depth_image_to_color_camera_custom.restype=k4a_result_t
		self.k4a_transformation_depth_image_to_color_camera_custom.argtypes=(k4a_transformation_t, \
																			k4a_image_t, \
																			k4a_image_t,\
																			k4a_image_t,\
																			k4a_image_t,\
																			k4a_transformation_interpolation_type_t,\
																			ctypes.c_uint32,\
																			)

		"""
		K4A_EXPORT k4a_result_t k4a_transformation_color_image_to_depth_camera(k4a_transformation_t transformation_handle,
																			   const k4a_image_t depth_image,
																			   const k4a_image_t color_image,
																			   k4a_image_t transformed_color_image);
		"""
		self.k4a_transformation_color_image_to_depth_camera = dll.k4a_transformation_color_image_to_depth_camera
		self.k4a_transformation_color_image_to_depth_camera.restype=k4a_result_t
		self.k4a_transformation_color_image_to_depth_camera.argtypes=(k4a_transformation_t, \
																		k4a_image_t, \
																		k4a_image_t,\
																		k4a_image_t,\
																		)

		"""
		K4A_EXPORT k4a_result_t k4a_transformation_depth_image_to_point_cloud(k4a_transformation_t transformation_handle,
																			  const k4a_image_t depth_image,
																			  const k4a_calibration_type_t camera,
																			  k4a_image_t xyz_image);
		"""
		self.k4a_transformation_depth_image_to_point_cloud = dll.k4a_transformation_depth_image_to_point_cloud
		self.k4a_transformation_depth_image_to_point_cloud.restype=k4a_result_t
		self.k4a_transformation_depth_image_to_point_cloud.argtypes=(k4a_transformation_t, \
																	k4a_image_t, \
																	k4a_calibration_type_t,\
																	k4a_image_t,\
																	)
def VERIFY(result, error):
	if result != K4A_RESULT_SUCCEEDED:
		print(error)
		traceback.print_stack()
		sys.exit(1)

