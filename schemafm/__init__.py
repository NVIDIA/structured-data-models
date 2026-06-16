"""Top-level package for schema-fm."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("schema-fm")
except PackageNotFoundError:
    __version__ = "0+unknown"
