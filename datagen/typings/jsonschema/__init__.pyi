from collections.abc import Iterator, Mapping, Sequence

class ValidationError(Exception):
    message: str
    absolute_path: Sequence[str | int]

class Draft202012Validator:
    def __init__(self, schema: Mapping[str, object]) -> None: ...
    def iter_errors(self, instance: object) -> Iterator[ValidationError]: ...
