import cv2
import numpy as np
import pytest

from sketch import create_sketch


def test_create_sketch_reads_custom_input_and_writes_output(tmp_path):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "result.jpg"
    source = np.full((24, 32, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(source_path), source)

    result = create_sketch(source_path, output_path)

    assert result == output_path
    sketch = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
    assert sketch is not None
    assert sketch.shape == (24, 32)


def test_create_sketch_reports_unreadable_input(tmp_path):
    with pytest.raises(ValueError, match="Could not read an image"):
        create_sketch(tmp_path / "missing.png", tmp_path / "result.jpg")
