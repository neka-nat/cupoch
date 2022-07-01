from . import _k4a
import numpy as np
import cv2
import sys
import ctypes
from .config import config
from . import postProcessing
import platform


class pyKinectAzure:
    def __init__(self, modulePath=None):
        if modulePath is None:
            if platform.system().lower() == "linux":
                modulePath = r"/usr/lib/x86_64-linux-gnu/libk4a.so"
            else:
                modulePath = (
                    "C:\\Program Files\\Azure Kinect SDK v1.4.0\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll"
                )
        self.k4a = _k4a.k4a(modulePath)

        self.device_handle = _k4a.k4a_device_t()
        self.capture_handle = _k4a.k4a_capture_t()
        self.config = config()
        self.imu_sample = _k4a.k4a_imu_sample_t()

        self.cameras_running = False
        self.imu_running = False

    def device_get_installed_count(self):
        """Gets the number of connected devices

        Parameters:
        None

        Returns:
        int: Number of sensors connected to the PC.

        Remarks:
        This API counts the number of Azure Kinect devices connected to the host PC.
        """
        return int(self.k4a.k4a_device_get_installed_count())

    def device_open(self, index=0):
        """Open an Azure Kinect device.

        Parameters:
        index (int): The index of the device to open, starting with

        Returns:
        None

        Remarks:
        If successful, k4a_device_open() will return a device handle in the device_handle parameter.
        This handle grants exclusive access to the device and may be used in the other Azure Kinect API calls.

        When done with the device, close the handle with k4a_device_close()
        """
        _k4a.VERIFY(self.k4a.k4a_device_open(index, self.device_handle), "Open K4A Device failed!")

    def device_close(self):
        """Closes an Azure Kinect device.

        Parameters:
        None

        Returns:
        None

        Remarks:
        Once closed, the handle is no longer valid.

        Before closing the handle to the device, ensure that all captures have been released with
        k4a_capture_release().
        """
        self.k4a.k4a_device_close(self.device_handle)

    def device_get_serialnum(self):
        """Get the Azure Kinect device serial number.

        Parameters:
        None

        Returns:
        A return of ::K4A_BUFFER_RESULT_SUCCEEDED means that the serial_number has been filled in. If the buffer is too
        small the function returns ::K4A_BUFFER_RESULT_TOO_SMALL and the size of the serial number is
        returned in the serial_number_size parameter. All other failures return ::K4A_BUFFER_RESULT_FAILED.

        Remarks:
        Queries the device for its serial number. If the caller needs to know the size of the serial number to allocate
        memory, the function should be called once with a NULL serial_number to get the needed size in the
        serial_number_size output, and then again with the allocated buffer.

        Only a complete serial number will be returned. If the caller's buffer is too small, the function will return
        ::K4A_BUFFER_RESULT_TOO_SMALL without returning any data in serial_number.
        """
        # First call to get the size of the buffer
        serial_number_size = ctypes.c_size_t()
        result = self.k4a.k4a_device_get_serialnum(self.device_handle, None, serial_number_size)

        if result == _k4a.K4A_BUFFER_RESULT_TOO_SMALL:
            serial_number = ctypes.create_string_buffer(serial_number_size.value)

        _k4a.VERIFY(
            self.k4a.k4a_device_get_serialnum(self.device_handle, serial_number, serial_number_size),
            "Read serial number failed!",
        )

        return serial_number.value.decode("utf-8")

    def device_start_cameras(self, device_config=None):
        """Starts color and depth camera capture.

        Parameters:
        device_config (k4a_device_configuration_t): The configuration we want to run the device in. This can be initialized with ::K4A_DEVICE_CONFIG_INIT_DEFAULT.

        Returns:
        None

        Remarks:
        Individual sensors configured to run will now start to stream captured data..

        It is not valid to call k4a_device_start_cameras() a second time on the same k4a_device_t until
        k4a_device_stop_cameras() has been called.
        """
        if device_config is not None:
            self.config = device_config

        if not self.cameras_running:
            _k4a.VERIFY(
                self.k4a.k4a_device_start_cameras(self.device_handle, self.config.current_config),
                "Start K4A cameras failed!",
            )
            self.cameras_running = True

    def device_stop_cameras(self):
        """Stops the color and depth camera capture..

        Parameters:
        None

        Returns:
        None

        Remarks:
        The streaming of individual sensors stops as a result of this call. Once called, k4a_device_start_cameras() may
        be called again to resume sensor streaming.
        """
        if self.cameras_running:
            self.k4a.k4a_device_stop_cameras(self.device_handle)
            self.cameras_running = False

    def device_start_imu(self):
        """Starts the IMU sample stream.

        Parameters:
        None

        Returns:
        None

        Remarks:
        Call this API to start streaming IMU data. It is not valid to call this function a second time on the same
        k4a_device_t until k4a_device_stop_imu() has been called.

        This function is dependent on the state of the cameras. The color or depth camera must be started before the IMU.
        K4A_RESULT_FAILED will be returned if one of the cameras is not running.
        """
        if self.cameras_running:
            if not self.imu_running:
                _k4a.VERIFY(self.k4a.k4a_device_start_imu(self.device_handle), "Start K4A IMU failed!")
                self.imu_running = True
        else:
            print("\nTurn on cameras before running IMU.\n")

    def device_stop_imu(self):
        """Stops the IMU capture.

        Parameters:
        None

        Returns:
        None

        Remarks:
        The streaming of the IMU stops as a result of this call. Once called, k4a_device_start_imu() may
        be called again to resume sensor streaming, so long as the cameras are running.

        This function may be called while another thread is blocking in k4a_device_get_imu_sample().
        Calling this function while another thread is in that function will result in that function returning a failure.

        """
        if self.imu_running:
            self.k4a.k4a_device_stop_imu(self.device_handle)
            self.imu_running = False

    def device_get_capture(self, timeout_in_ms=_k4a.K4A_WAIT_INFINITE):
        """Reads a sensor capture.

        Parameters:h
        timeout_in_ms (int):Specifies the time in milliseconds the function should block waiting for the capture. If set to 0, the function will
                                                return without blocking. Passing a value of #K4A_WAIT_INFINITE will block indefinitely until data is available, the
                                                device is disconnected, or another error occurs.

        Returns:
        None

        Remarks:
        Gets the next capture in the streamed sequence of captures from the camera. If a new capture is not currently
        available, this function will block until the timeout is reached. The SDK will buffer at least two captures worth
        of data before dropping the oldest capture. Callers needing to capture all data need to ensure they read the data as
        fast as the data is being produced on average.

        Upon successfully reading a capture this function will return success and populate capture.
        If a capture is not available in the configured timeout_in_ms, then the API will return ::K4A_WAIT_RESULT_TIMEOUT.

        """
        if self.cameras_running:
            _k4a.VERIFY(
                self.k4a.k4a_device_get_capture(self.device_handle, self.capture_handle, timeout_in_ms),
                "Get capture failed!",
            )

    def device_get_imu_sample(self, timeout_in_ms=_k4a.K4A_WAIT_INFINITE):
        """Reads an IMU sample.

        Parameters:h
        timeout_in_ms (int):Specifies the time in milliseconds the function should block waiting for the capture. If set to 0, the function will
                                                return without blocking. Passing a value of #K4A_WAIT_INFINITE will block indefinitely until data is available, the
                                                device is disconnected, or another error occurs.

        Returns:
        None

        Remarks:
        Gets the next sample in the streamed sequence of IMU samples from the device. If a new sample is not currently
        available, this function will block until the timeout is reached. The API will buffer at least two camera capture
        intervals worth of samples before dropping the oldest sample. Callers needing to capture all data need to ensure they
        read the data as fast as the data is being produced on average.

        Upon successfully reading a sample this function will return success and populate imu_sample.
        If a sample is not available in the configured timeout_in_ms, then the API will return ::K4A_WAIT_RESULT_TIMEOUT.
        """
        if self.imu_running:
            _k4a.VERIFY(
                self.k4a.k4a_device_get_imu_sample(self.device_handle, self.imu_sample, timeout_in_ms),
                "Get IMU failed!",
            )

    def device_get_calibration(self, depth_mode, color_resolution, calibration):
        """Get the camera calibration for the entire Azure Kinect device.

        Parameters:h
        depth_mode(k4a_depth_mode_t): Mode in which depth camera is operated.
        color_resolution(k4a_color_resolution_t): Resolution in which color camera is operated.
        calibration(k4a_calibration_t):Location to write the calibration

        Returns:
        K4A_RESULT_SUCCEEDED if calibration was successfully written. ::K4A_RESULT_FAILED otherwise.

        Remarks:
        The calibration represents the data needed to transform between the camera views and may be
        different for each operating depth_mode and color_resolution the device is configured to operate in.

        The calibration output is used as input to all calibration and transformation functions.
        """
        _k4a.VERIFY(
            self.k4a.k4a_device_get_calibration(self.device_handle, depth_mode, color_resolution, calibration),
            "Get calibration failed!",
        )

    def capture_get_color_image(self):
        """Get the color image associated with the given capture.

        Parameters:
        None

        Returns:
        k4a_image_t: Handle to the Image

        Remarks:
        Call this function to access the color image part of this capture. Release the ref k4a_image_t with
        k4a_image_release();
        """

        return self.k4a.k4a_capture_get_color_image(self.capture_handle)

    def capture_get_depth_image(self):
        """Get the depth image associated with the given capture.

        Parameters:
        None

        Returns:
        k4a_image_t: Handle to the Image

        Remarks:
        Call this function to access the depth image part of this capture. Release the k4a_image_t with
        k4a_image_release();
        """

        return self.k4a.k4a_capture_get_depth_image(self.capture_handle)

    def capture_get_ir_image(self):
        """Get the IR image associated with the given capture.

        Parameters:
        None

        Returns:
        k4a_image_t: Handle to the Image

        Remarks:
        Call this function to access the IR image part of this capture. Release the k4a_image_t with
        k4a_image_release();
        """

        return self.k4a.k4a_capture_get_ir_image(self.capture_handle)

    def image_create(self, image_format, width_pixels, height_pixels, stride_bytes, image_handle):
        """Create an image.

        Parameters:
        image_format(k4a_image_format_t): The format of the image that will be stored in this image container.
        width_pixels(int): Width in pixels.
        height_pixels(int): Height in pixels.
        stride_bytes(int): The number of bytes per horizontal line of the image.
                                           If set to 0, the stride will be set to the minimum size given the format and width_pixels.
        image_handle(k4a_image_t): Pointer to store image handle in.

        Returns:
        Returns #K4A_RESULT_SUCCEEDED on success. Errors are indicated with #K4A_RESULT_FAILED.

        Remarks:
        This function is used to create images of formats that have consistent stride. The function is not suitable for
        compressed formats that may not be represented by the same number of bytes per line.

        For most image formats, the function will allocate an image buffer of size height_pixels * stride_bytes.
        Buffers #K4A_IMAGE_FORMAT_COLOR_NV12 format will allocate an additional height_pixels / 2 set of lines (each of
        stride_bytes). This function cannot be used to allocate #K4A_IMAGE_FORMAT_COLOR_MJPG buffers.
        """
        _k4a.VERIFY(
            self.k4a.k4a_image_create(image_format, width_pixels, height_pixels, stride_bytes, image_handle),
            "Create image failed!",
        )

    def image_get_buffer(self, image_handle):
        """Get the image buffer.

        Parameters:
        image_handle (k4a_image_t): Handle to the Image

        Returns:
        ctypes.POINTER(ctypes.c_uint8): The function will return NULL if there is an error, and will normally return a pointer to the image buffer.
                                                                        Since all k4a_image_t instances are created with an image buffer, this function should only return NULL if the
                                                                        image_handle is invalid.

        Remarks:
        Use this buffer to access the raw image data.
        """

        return self.k4a.k4a_image_get_buffer(image_handle)

    def image_get_size(self, image_handle):
        """Get the image buffer size.

        Parameters:
        image_handle (k4a_image_t): Handle to the Image

        Returns:
        int: The function will return 0 if there is an error, and will normally return the image size.
        Since all k4a_image_t instances are created with an image buffer, this function should only return 0 if the
        image_handle is invalid.

        Remarks:
        Use this function to know what the size of the image buffer is returned by k4a_image_get_buffer().
        """

        return int(self.k4a.k4a_image_get_size(image_handle))

    def image_get_format(self, image_handle):
        """Get the format of the image.

        Parameters:
        image_handle (k4a_image_t): Handle to the Image

        Returns:
        int: This function is not expected to fail, all k4a_image_t's are created with a known format. If the
        image_handle is invalid, the function will return ::K4A_IMAGE_FORMAT_CUSTOM.

        Remarks:
        Use this function to determine the format of the image buffer.
        """

        return int(self.k4a.k4a_image_get_format(image_handle))

    def image_get_width_pixels(self, image_handle):
        """Get the image width in pixels.

        Parameters:
        image_handle (k4a_image_t): Handle to the Image

        Returns:
        int: This function is not expected to fail, all k4a_image_t's are created with a known width. If the part
        image_handle is invalid, the function will return 0.
        """

        return int(self.k4a.k4a_image_get_width_pixels(image_handle))

    def image_get_height_pixels(self, image_handle):
        """Get the image height in pixels.

        Parameters:
        image_handle (k4a_image_t): Handle to the Image

        Returns:
        int: This function is not expected to fail, all k4a_image_t's are created with a known height. If the part
        image_handle is invalid, the function will return 0.
        """

        return int(self.k4a.k4a_image_get_height_pixels(image_handle))

    def image_get_stride_bytes(self, image_handle):
        """Get the image stride in bytes.

        Parameters:
        image_handle (k4a_image_t): Handle to the Image

        Returns:
        int: This function is not expected to fail, all k4a_image_t's are created with a known stride. If the
        image_handle is invalid, or the image's format does not have a stride, the function will return 0.
        """

        return int(self.k4a.k4a_image_get_stride_bytes(image_handle))

    def transformation_create(self, calibration):
        """Get handle to transformation handle.

        Parameters:
        calibration(k4a_calibration_t): A calibration structure obtained by k4a_device_get_calibration().

        Returns:
        k4a_transformation_t: A transformation handle. A NULL is returned if creation fails.

        Remarks:
        The transformation handle is used to transform images from the coordinate system of one camera into the other. Each
        transformation handle requires some pre-computed resources to be allocated, which are retained until the handle is
        destroyed.

        The transformation handle must be destroyed with k4a_transformation_destroy() when it is no longer to be used.
        """

        return self.k4a.k4a_transformation_create(calibration)

    def transformation_destroy(self, transformation_handle):
        """Destroy transformation handle.

        Parameters:
        transformation_handle(k4a_transformation_t): Transformation handle to destroy.

        Returns:
        None

        Remarks:
        None
        """

        self.k4a.k4a_transformation_destroy(transformation_handle)

    def transformation_depth_image_to_color_camera(
        self, transformation_handle, input_depth_image_handle, transformed_depth_image_handle
    ):
        """Transforms the depth map into the geometry of the color camera.

        Parameters:
        transformation_handle (k4a_transformation_t): Transformation handle.
        input_depth_image_handle (k4a_image_t): Handle to input depth image.
        transformed_depth_image_handle (k4a_image_t): Handle to output transformed depth image.

        Returns:
        K4A_RESULT_SUCCEEDED if transformed_depth_image was successfully written and ::K4A_RESULT_FAILED otherwise.

        Remarks:
        This produces a depth image for which each pixel matches the corresponding pixel coordinates of the color camera.

        transformed_depth_image must have a width and height matching the width and height of the color camera in the mode
        specified by the k4a_calibration_t used to create the transformation_handle with k4a_transformation_create().
        """

        _k4a.VERIFY(
            self.k4a.k4a_transformation_depth_image_to_color_camera(
                transformation_handle, input_depth_image_handle, transformed_depth_image_handle
            ),
            "Transformation from depth to color failed!",
        )

    def image_convert_to_numpy(self, image_handle):
        """Get the image data as a numpy array

        Parameters:
        image_handle (k4a_image_t): Handle to the Image

        Returns:
        numpy.ndarray: Numpy array with the image data
        """

        # Get the pointer to the buffer containing the image data
        buffer_pointer = self.image_get_buffer(image_handle)

        # Get the size of the buffer
        image_size = self.image_get_size(image_handle)
        image_width = self.image_get_width_pixels(image_handle)
        image_height = self.image_get_height_pixels(image_handle)

        # Get the image format
        image_format = self.image_get_format(image_handle)

        # Read the data in the buffer
        buffer_array = np.ctypeslib.as_array(buffer_pointer, shape=(image_size,))

        # Parse buffer based on image format
        if image_format == _k4a.K4A_IMAGE_FORMAT_COLOR_MJPG:
            return cv2.imdecode(np.frombuffer(buffer_array, dtype=np.uint8), -1)
        elif image_format == _k4a.K4A_IMAGE_FORMAT_COLOR_NV12:
            yuv_image = np.frombuffer(buffer_array, dtype=np.uint8).reshape(int(image_height * 1.5), image_width)
            return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
        elif image_format == _k4a.K4A_IMAGE_FORMAT_COLOR_YUY2:
            yuv_image = np.frombuffer(buffer_array, dtype=np.uint8).reshape(image_height, image_width, 2)
            return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_YUY2)
        elif image_format == _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32:
            return np.frombuffer(buffer_array, dtype=np.uint8).reshape(image_height, image_width, 4)
        elif image_format == _k4a.K4A_IMAGE_FORMAT_DEPTH16:
            return np.frombuffer(buffer_array, dtype="<u2").reshape(
                image_height, image_width
            )  # little-endian 16 bits unsigned Depth data
        elif image_format == _k4a.K4A_IMAGE_FORMAT_IR16:
            return np.frombuffer(buffer_array, dtype="<u2").reshape(
                image_height, image_width
            )  # little-endian 16 bits unsigned IR data. For more details see: https://microsoft.github.io/Azure-Kinect-Sensor-SDK/release/1.2.x/namespace_microsoft_1_1_azure_1_1_kinect_1_1_sensor_a7a3cb7a0a3073650bf17c2fef2bfbd1b.html

    def transform_depth_to_color(self, input_depth_image_handle, color_image_handle):
        calibration = _k4a.k4a_calibration_t()

        # Get desired image format
        image_format = self.image_get_format(input_depth_image_handle)
        image_width = self.image_get_width_pixels(color_image_handle)
        image_height = self.image_get_height_pixels(color_image_handle)
        image_stride = 0

        # Get the camera calibration
        self.device_get_calibration(self.config.depth_mode, self.config.color_resolution, calibration)

        # Create transformation
        transformation_handle = self.transformation_create(calibration)

        # Create the image handle
        transformed_depth_image_handle = _k4a.k4a_image_t()
        self.image_create(image_format, image_width, image_height, image_stride, transformed_depth_image_handle)

        # Transform the depth image to the color image format
        self.transformation_depth_image_to_color_camera(
            transformation_handle, input_depth_image_handle, transformed_depth_image_handle
        )

        # Get transformed image data
        transformed_image = self.image_convert_to_numpy(transformed_depth_image_handle)

        # Close transformation
        self.transformation_destroy(transformation_handle)

        return transformed_image

    def image_release(self, image_handle):
        """Remove a reference from the k4a_image_t.

        Parameters:
        image_handle (k4a_image_t): Handle to the Image

        Returns:
        None

        Remarks:
        References manage the lifetime of the object. When the references reach zero the object is destroyed. A caller must
        not access the object after its reference is released.
        """

        self.k4a.k4a_image_release(image_handle)

    def capture_release(self):
        """Release a capture.

        Parameters:
        None

        Returns:
        None

        Remarks:
        Call this function when finished using the capture.
        """

        self.k4a.k4a_capture_release(self.capture_handle)

    def get_imu_sample(self, timeout_in_ms=_k4a.K4A_WAIT_INFINITE):

        # Get the sample from the device
        self.device_get_imu_sample(timeout_in_ms)

        # Read the raw data from the buffer pointer
        buffer_array = np.array(np.ctypeslib.as_array(self.imu_sample, shape=(_k4a.IMU_SAMPLE_SIZE,)).tolist())

        imu_results = self.imu_results
        imu_results.temperature = buffer_array[0]
        imu_results.acc_sample = buffer_array[1][1]
        imu_results.acc_timestamp_usec = buffer_array[2]
        imu_results.gyro_sample = buffer_array[3][1]
        imu_results.gyro_timestamp_usec = buffer_array[4]

        return imu_results

    class imu_results:
        def __init__(self):
            self.temperature = None
            self.acc_sample = None
            self.acc_timestamp_usec = None
            self.gyro_sample = None
            self.gyro_timestamp_usec = None
