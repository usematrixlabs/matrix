"""S1 Frame Extractor

Extracts individual frames from UAV video input.
"""


class FrameExtractor:
    """Extract frames from UAV video source."""

    def __init__(self, video_path: str, frame_rate: float = 1.0):
        """
        Initialize a frame extractor for a video.
        
        Parameters:
            video_path (str): Path to the UAV video.
            frame_rate (float): Frame extraction rate.
        """
        self.video_path = video_path
        self.frame_rate = frame_rate

    def extract(self, start_time: float = 0.0, end_time: float | None = None):
        """
        Provides frames from the configured video within an optional time range. Frame extraction is currently unavailable.
        
        Parameters:
            start_time (float): Beginning of the extraction range in seconds.
            end_time (float | None): End of the extraction range in seconds, or None to use the video's end.
        
        Returns:
            list: Frame identifiers with timestamps; currently always empty.
        """
        # TODO: Implement video frame extraction
        return []

    def get_keyframes(self):
        """
        Provide keyframe identifiers for the configured video.
        
        Returns:
            list: An empty list because keyframe selection is not currently implemented.
        """
        # TODO: Implement keyframe selection
        return []