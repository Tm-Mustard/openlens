import cv2
import numpy as np

BLUR_THRESHOLD = 25.0
DARKNESS_THRESHOLD = 70.0
LOW_CONTRAST_THRESHOLD = 30.0


def load_image(image_bytes: bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def is_blurry(img: np.ndarray, threshold: float = BLUR_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return blur_score < threshold, blur_score


def needs_lighting_enhancement(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    contrast_score = float(gray.std())
    should_enhance = (
        mean_brightness < DARKNESS_THRESHOLD
        or contrast_score < LOW_CONTRAST_THRESHOLD
    )
    return should_enhance, mean_brightness, contrast_score


def enhance_lighting_preserve_color(img: np.ndarray):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )
    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )


def encode_as_jpeg(img: np.ndarray, quality: int = 95):
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to encode enhanced image")
    return buffer.tobytes()


def process_image(
    image_bytes: bytes,
    hard_blur_threshold: float = BLUR_THRESHOLD,
    apply_enhancement: bool = True
):
    img = load_image(image_bytes)
    if img is None:
        return {
            "status": "rejected",
            "reason": "Could not read the uploaded file as an image."
        }

    blurry, blur_score = is_blurry(img, hard_blur_threshold)
    if blurry:
        return {
            "status": "rejected",
            "reason": "Image is too blurry to read reliably.",
            "blur_score": blur_score
        }

    should_enhance, brightness, contrast = needs_lighting_enhancement(img)
    if apply_enhancement and should_enhance:
        enhanced_img = enhance_lighting_preserve_color(img)
        return {
            "status": "ok",
            "image_bytes": encode_as_jpeg(enhanced_img),
            "blur_score": blur_score,
            "mean_brightness": brightness,
            "contrast_score": contrast,
            "enhanced": True
        }

    return {
        "status": "ok",
        "image_bytes": encode_as_jpeg(img),  
        "blur_score": blur_score,
        "mean_brightness": brightness,
        "contrast_score": contrast,
        "enhanced": False,
    }
