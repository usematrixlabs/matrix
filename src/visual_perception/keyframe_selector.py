"""S1 Keyframe Selector

Selects a subset of frames as keyframes for downstream processing.
"""


class KeyframeSelector:
    """Select keyframes from extracted frames."""

    def __init__(self, extraction_result, selection_criteria: dict | None = None):
        """
        Initialize a keyframe selector with extracted frames and optional selection criteria.
        
        Parameters:
            extraction_result: The extracted frame data to evaluate.
            selection_criteria (dict | None): Criteria used to select keyframes. Defaults to an empty dictionary.
        """
        self.extraction_result = extraction_result
        self.selection_criteria = selection_criteria or {}

    def select(self):
        """Provides the keyframe identifiers selected by this selector.
        
        Returns:
            list: An empty list until keyframe selection is implemented.
        """
        # TODO: Implement keyframe selection
        return []