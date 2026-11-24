import argparse
from pathlib import Path

import cv2


def create_sketch(input_path: Path, output_path: Path) -> Path:
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not read an image from {input_path}.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    if not cv2.imwrite(str(output_path), sketch):
        raise OSError(f"Could not save the sketch to {output_path}.")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a pencil sketch from an image.")
    parser.add_argument("input", nargs="?", type=Path, default=Path("test.png"))
    parser.add_argument("output", nargs="?", type=Path, default=Path("sketch4.jpg"))
    args = parser.parse_args()

    try:
        create_sketch(args.input, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
