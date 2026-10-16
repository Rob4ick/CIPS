import cv2
import numpy as np
import time
import os


def to_binary(image_path, threshold=127):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть {image_path}")
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary


def dilation_opencv(binary):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def dilation_native(binary):
    h, w = binary.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h + 1, 1:w + 1] = binary

    result = np.zeros_like(binary)

    for i in range(h):
        for j in range(w):
            window = padded[i:i + 3, j:j + 3]
            if window.max() == 255:
                result[i, j] = 255
    return result


def main():
    input_path = "images/input.jpg"
    os.makedirs("results", exist_ok=True)

    binary = to_binary(input_path)
    cv2.imwrite("results/binary.png", binary)

    t0 = time.perf_counter()
    res_cv = dilation_opencv(binary)
    t_cv = time.perf_counter() - t0
    cv2.imwrite("results/dilation_opencv.png", res_cv)

    t0 = time.perf_counter()
    res_native = dilation_native(binary)
    t_native = time.perf_counter() - t0
    cv2.imwrite("results/dilation_native.png", res_native)

    identical = np.array_equal(res_cv, res_native)

    print(f"OpenCV время:  {t_cv:.6f} с")
    print(f"Native время:  {t_native:.6f} с")
    print(f"Ускорение OpenCV в {t_native / t_cv:.1f} раз")
    print(f"Результаты идентичны: {identical}")


if __name__ == "__main__":
    main()