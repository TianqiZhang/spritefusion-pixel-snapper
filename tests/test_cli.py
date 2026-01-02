"""Tests for cli module."""
from __future__ import annotations

import pytest

from pixel_snapper.config import PixelSnapperError
from pixel_snapper.cli import parse_args, process_image_bytes


class TestParseArgs:
    """Tests for parse_args function."""

    def test_minimal_args(self) -> None:
        """Should parse minimal required arguments."""
        config = parse_args(["prog", "input.png", "output.png"])
        assert config.input_path == "input.png"
        assert config.output_path == "output.png"

    def test_k_colors(self) -> None:
        """Should parse k_colors argument."""
        config = parse_args(["prog", "in.png", "out.png", "32"])
        assert config.k_colors == 32

    def test_invalid_k_colors_warning(self, capsys) -> None:
        """Should warn on invalid k_colors and use default."""
        config = parse_args(["prog", "in.png", "out.png", "abc"])
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert config.k_colors == 16  # Default

    def test_negative_k_colors_warning(self, capsys) -> None:
        """Should warn on negative k_colors and use default."""
        config = parse_args(["prog", "in.png", "out.png", "-5"])
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert config.k_colors == 16

    def test_preview_flag(self) -> None:
        """Should parse --preview flag."""
        config = parse_args(["prog", "in.png", "out.png", "--preview"])
        assert config.preview is True

    def test_timing_flag(self) -> None:
        """Should parse --timing flag."""
        config = parse_args(["prog", "in.png", "out.png", "--timing"])
        assert config.timing is True

    def test_palette_option(self) -> None:
        """Should parse --palette option."""
        config = parse_args(["prog", "in.png", "out.png", "--palette", "perler"])
        assert config.palette == "perler"

    def test_palette_space_rgb(self) -> None:
        """Should parse --palette-space rgb."""
        config = parse_args([
            "prog", "in.png", "out.png",
            "--palette", "perler",
            "--palette-space", "rgb"
        ])
        assert config.palette_space == "rgb"

    def test_palette_space_lab(self) -> None:
        """Should parse --palette-space lab."""
        config = parse_args([
            "prog", "in.png", "out.png",
            "--palette", "perler",
            "--palette-space", "lab"
        ])
        assert config.palette_space == "lab"

    def test_invalid_palette_space(self) -> None:
        """Should reject invalid palette-space."""
        with pytest.raises(PixelSnapperError, match="must be 'rgb' or 'lab'"):
            parse_args([
                "prog", "in.png", "out.png",
                "--palette-space", "xyz"
            ])

    def test_missing_input(self) -> None:
        """Should require input path."""
        with pytest.raises(PixelSnapperError, match="Usage"):
            parse_args(["prog"])

    def test_missing_output(self) -> None:
        """Should require output path."""
        with pytest.raises(PixelSnapperError, match="Usage"):
            parse_args(["prog", "input.png"])

    def test_too_many_positional(self) -> None:
        """Should reject extra positional arguments."""
        with pytest.raises(PixelSnapperError, match="Usage"):
            parse_args(["prog", "in.png", "out.png", "16", "extra"])

    def test_missing_palette_value(self) -> None:
        """Should require value after --palette."""
        with pytest.raises(PixelSnapperError, match="Usage"):
            parse_args(["prog", "in.png", "out.png", "--palette"])

    def test_missing_palette_space_value(self) -> None:
        """Should require value after --palette-space."""
        with pytest.raises(PixelSnapperError, match="Usage"):
            parse_args(["prog", "in.png", "out.png", "--palette-space"])

    def test_combined_flags(self) -> None:
        """Should parse multiple flags together."""
        config = parse_args([
            "prog", "in.png", "out.png", "24",
            "--preview", "--timing",
            "--palette", "hama",
            "--palette-space", "lab"
        ])
        assert config.input_path == "in.png"
        assert config.output_path == "out.png"
        assert config.k_colors == 24
        assert config.preview is True
        assert config.timing is True
        assert config.palette == "hama"
        assert config.palette_space == "lab"

    def test_qwen_flag(self) -> None:
        """Should enable Qwen pre-processing."""
        config = parse_args(["prog", "in.png", "out.png", "--qwen"])
        assert config.qwen_enabled is True

    def test_qwen_override_not_supported(self) -> None:
        """Should reject unsupported Qwen override flags."""
        with pytest.raises(PixelSnapperError, match="Unsupported Qwen option"):
            parse_args([
                "prog", "in.png", "out.png",
                "--qwen-prompt", "make it cute pixel art",
            ])


class TestProcessImageBytes:
    """Tests for process_image_bytes function."""

    def test_returns_bytes(self, sample_image_bytes: bytes) -> None:
        """Should return PNG bytes."""
        result = process_image_bytes(sample_image_bytes)
        assert isinstance(result, bytes)
        # Check PNG signature
        assert result[:8] == b'\x89PNG\r\n\x1a\n'

    def test_default_config(self, sample_image_bytes: bytes) -> None:
        """Should work with default config."""
        result = process_image_bytes(sample_image_bytes, None)
        assert len(result) > 0

    def test_custom_config(self, sample_image_bytes: bytes) -> None:
        """Should accept custom config."""
        from pixel_snapper import Config
        config = Config(k_colors=4)
        result = process_image_bytes(sample_image_bytes, config)
        assert len(result) > 0
