"""Translate geometric failures at adapter transaction boundaries."""
from functools import wraps

from ....tolerance import GeometryToleranceError
from ._types import QuadLayerError


def layer_error(error, **context):
    details = dict(error.context)
    for key, value in context.items():
        details.setdefault(key, value)
    return QuadLayerError(str(error), code=error.code, **details)


def geometric_errors(operation):
    @wraps(operation)
    def wrapped(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except GeometryToleranceError as error:
            raise layer_error(error) from error
    return wrapped
