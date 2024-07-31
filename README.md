# Doctor Strange Shield Effects using OpenCV and MediaPipe

This project uses OpenCV and MediaPipe to create visual effects resembling Doctor Strange's shield when specific hand gestures are detected. The program captures live video input, detects hand landmarks, and overlays shield animations when a certain gesture is recognized.

## Prerequisites

To run this project, you need the following packages installed:

- Python 3.x
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

You can install these packages using pip:

```sh
pip install opencv-python mediapipe numpy
```

## How It Works

1. **Hand Detection:** The program uses MediaPipe's Hands module to detect hand landmarks.
2. **Gesture Recognition:** It checks if the thumb, index, and middle fingers are extended while the ring and pinky fingers are closed. If this gesture is detected, the program considers it as a gesture for activating the shield effect.
3. **Overlay Effect:** When the specified gesture is detected, a shield animation (loaded from a video file) is overlaid on the video feed at the position of the detected hand.

## Code Overview

### Import Libraries

```python
import cv2
import mediapipe as mp
import numpy as np
```

### Functions

- **is_three_fingers_extended(landmarks):** Checks if the thumb, index, and middle fingers are extended.
- **mapFromTo(x, a, b, c, d):** Maps a value from one range to another.
- **Overlay(background, overlay, x, y, size):** Overlays an image onto a background image.
- **draw_spark_effect(image, points, shield_frame):** Draws the shield effect on the given points.

## Running the Program

1. Clone this repository or copy the provided code.
2. Make sure you have the necessary dependencies installed.
3. Download or create a `shield.mp4` video file to use as the shield animation.
4. Run the script:

```sh
python main.py
```

Replace `main.py` with the name of your Python file.

## Usage

- **Start the program:** The program will start capturing video from the webcam.
- **Gesture Detection:** Extend the thumb, index, and middle fingers while keeping the ring and pinky fingers closed to activate the shield effect.
- **Exit:** Press 'q' to exit the program.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking.
- [OpenCV](https://opencv.org/) for video processing.
