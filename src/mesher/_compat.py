"""Standard-library compatibility for the supported Python versions."""

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        __str__ = str.__str__
        __format__ = str.__format__


if sys.version_info >= (3, 10):
    from functools import partial

    zip_strict = partial(zip, strict=True)
else:
    from itertools import zip_longest

    def zip_strict(first, second):
        missing = object()
        for left, right in zip_longest(first, second, fillvalue=missing):
            if left is missing:
                raise ValueError("zip() argument 2 is longer than argument 1")
            if right is missing:
                raise ValueError("zip() argument 2 is shorter than argument 1")
            yield left, right
