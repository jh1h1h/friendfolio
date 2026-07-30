class DeepSeekError(RuntimeError):
    """Base class for failures produced by the DeepSeek proposal pipeline."""


class DeepSeekAPIError(DeepSeekError):
    """DeepSeek could not be reached or returned an unsuccessful HTTP response."""


class DeepSeekResponseError(DeepSeekError):
    """DeepSeek returned content that could not be parsed or validated."""


class NoteOperationError(DeepSeekError, ValueError):
    """A validated model operation could not be applied safely to the saved note."""

    def __init__(
        self,
        message: str,
        *,
        action: str | None = None,
        match_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.action = action
        self.match_count = match_count
