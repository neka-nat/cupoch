from . import _k4a


class config:
    def __init__(
        self,
        color_format=_k4a.K4A_IMAGE_FORMAT_COLOR_MJPG,
        color_resolution=_k4a.K4A_COLOR_RESOLUTION_720P,
        depth_mode=_k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED,
        camera_fps=_k4a.K4A_FRAMES_PER_SECOND_30,
        synchronized_images_only=False,
        depth_delay_off_color_usec=0,
        wired_sync_mode=_k4a.K4A_WIRED_SYNC_MODE_STANDALONE,
        subordinate_delay_off_master_usec=0,
        disable_streaming_indicator=False,
    ):
        self.color_format = color_format
        self.color_resolution = color_resolution
        self.depth_mode = depth_mode
        self.camera_fps = camera_fps
        self.synchronized_images_only = synchronized_images_only
        self.depth_delay_off_color_usec = depth_delay_off_color_usec
        self.wired_sync_mode = wired_sync_mode
        self.subordinate_delay_off_master_usec = subordinate_delay_off_master_usec
        self.disable_streaming_indicator = disable_streaming_indicator

        self._on_change()

    def __setattr__(self, name, value):
        """Run on change function when configuration parameters are changed"""
        if hasattr(self, name):
            if name != "current_config":
                if int(self.__dict__[name]) != value:
                    self.__dict__[name] = value
                    self._on_change()
            else:
                self.__dict__[name] = value
        else:
            self.__dict__[name] = value

    def __str__(self):
        """Print the current settings and a short explanation"""
        message = (
            "Device configuration: \n"
            f"\tcolor_format: {self.color_format} \n\t(0:JPG, 1:NV12, 2:YUY2, 3:BGRA32)\n\n"
            f"\tcolor_resolution: {self.color_resolution} \n\t(0:OFF, 1:720p, 2:1080p, 3:1440p, 4:1536p, 5:2160p, 6:3072p)\n\n"
            f"\tdepth_mode: {self.depth_mode} \n\t(0:OFF, 1:NFOV_2X2BINNED, 2:NFOV_UNBINNED,3:WFOV_2X2BINNED, 4:WFOV_UNBINNED, 5:Passive IR)\n\n"
            f"\tcamera_fps: {self.camera_fps} \n\t(0:5 FPS, 1:15 FPS, 2:30 FPS)\n\n"
            f"\tsynchronized_images_only: {self.synchronized_images_only} \n\t(True of False). Drop images if the color and depth are not synchronized\n\n"
            f"\tdepth_delay_off_color_usec: {self.depth_delay_off_color_usec} ms. \n\tDelay between the color image and the depth image\n\n"
            f"\twired_sync_mode: {self.wired_sync_mode}\n\t(0:Standalone mode, 1:Master mode, 2:Subordinate mode)\n\n"
            f"\tsubordinate_delay_off_master_usec: {self.subordinate_delay_off_master_usec} ms.\n\tThe external synchronization timing.\n\n"
            f"\tdisable_streaming_indicator: {self.disable_streaming_indicator} \n\t(True or False). Streaming indicator automatically turns on when the color or depth camera's are in use.\n\n"
        )
        return message

    def _on_change(self):
        self.current_config = _k4a.k4a_device_configuration_t(
            self.color_format,
            self.color_resolution,
            self.depth_mode,
            self.camera_fps,
            self.synchronized_images_only,
            self.depth_delay_off_color_usec,
            self.wired_sync_mode,
            self.subordinate_delay_off_master_usec,
            self.disable_streaming_indicator,
        )
