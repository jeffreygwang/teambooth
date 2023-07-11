from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class EmptyResponse(_message.Message):
    __slots__ = ["success"]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class GetRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class MergeRequest(_message.Message):
    __slots__ = ["ckpt_diff_id"]
    CKPT_DIFF_ID_FIELD_NUMBER: _ClassVar[int]
    ckpt_diff_id: str
    def __init__(self, ckpt_diff_id: _Optional[str] = ...) -> None: ...

class ModelResponse(_message.Message):
    __slots__ = ["hosted_id", "success"]
    HOSTED_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    hosted_id: str
    success: bool
    def __init__(self, success: bool = ..., hosted_id: _Optional[str] = ...) -> None: ...

class RaftRequestVoteRequest(_message.Message):
    __slots__ = ["candidate_id", "term"]
    CANDIDATE_ID_FIELD_NUMBER: _ClassVar[int]
    TERM_FIELD_NUMBER: _ClassVar[int]
    candidate_id: str
    term: int
    def __init__(self, term: _Optional[int] = ..., candidate_id: _Optional[str] = ...) -> None: ...

class RaftRequestVoteResponse(_message.Message):
    __slots__ = ["vote"]
    VOTE_FIELD_NUMBER: _ClassVar[int]
    vote: bool
    def __init__(self, vote: bool = ...) -> None: ...

class RaftUpdateStateRequest(_message.Message):
    __slots__ = ["data", "replica_id"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    replica_id: str
    def __init__(self, replica_id: _Optional[str] = ..., data: _Optional[bytes] = ...) -> None: ...
