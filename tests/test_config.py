"""Tests for config module."""
from __future__ import annotations

import pytest

from pixel_snapper.config import Config, PixelSnapperError, validate_image_dimensions


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = Config()
        assert config.k_colors == 16
        assert config.k_seed == 42
        assert config.max_kmeans_iterations == 15
        assert config.peak_threshold_multiplier == 0.2
        assert config.palette is None
        assert config.palette_space == "lab"

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = Config(k_colors=32, k_seed=123, palette="perler")
        assert config.k_colors == 32
        assert config.k_seed == 123
        assert config.palette == "perler"

    def test_paths(self) -> None:
        """Config should store input/output paths."""
        config = Config(input_path="in.png", output_path="out.png")
        assert config.input_path == "in.png"
        assert config.output_path == "out.png"


class TestValidateImageDimensions:
    """Tests for validate_image_dimensions function."""

    def test_valid_dimensions(self) -> None:
        """Should accept valid dimensions."""
        validate_image_dimensions(100, 100)
        validate_image_dimensions(1, 1)
        validate_image_dimensions(10000, 10000)

    def test_zero_width(self) -> None:
        """Should reject zero width."""
        with pytest.raises(PixelSnapperError, match="cannot be zero"):
            validate_image_dimensions(0, 100)

    def test_zero_height(self) -> None:
        """Should reject zero height."""
        with pytest.raises(PixelSnapperError, match="cannot be zero"):
            validate_image_dimensions(100, 0)

    def test_too_large_width(self) -> None:
        """Should reject width over 10000."""
        with pytest.raises(PixelSnapperError, match="too large"):
            validate_image_dimensions(10001, 100)

    def test_too_large_height(self) -> None:
        """Should reject height over 10000."""
        with pytest.raises(PixelSnapperError, match="too large"):
            validate_image_dimensions(100, 10001)


class TestPixelSnapperError:
    """Tests for PixelSnapperError exception."""

    def test_is_exception(self) -> None:
        """Should be a proper Exception subclass."""
        assert issubclass(PixelSnapperError, Exception)

    def test_message(self) -> None:
        """Should preserve error message."""
        error = PixelSnapperError("test message")
        assert str(error) == "test message"
