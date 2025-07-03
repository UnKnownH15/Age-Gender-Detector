import pytest
import numpy as np
import cv2
from project import load_models, highlightFace, predict_age_gender


@pytest.fixture
def dummy_image():
    return np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

def test_load_models():
    faceNet, ageNet, genderNet = load_models()
    assert faceNet is not None
    assert ageNet is not None
    assert genderNet is not None
    assert hasattr(faceNet, 'setInput')
    assert hasattr(ageNet, 'forward')
    assert hasattr(genderNet, 'forward')

def test_highlightFace(dummy_image):
    faceNet, _, _ = load_models()
    frame, faceBoxes = highlightFace(faceNet, dummy_image, conf_threshold=0.0)  # Lower threshold to simulate detection
    assert isinstance(frame, np.ndarray)
    assert isinstance(faceBoxes, list)
    for box in faceBoxes:
        assert len(box) == 4
        assert all(isinstance(coord, int) for coord in box)

def test_predict_age_gender(dummy_image):
    _, ageNet, genderNet = load_models()
    dummy_face = cv2.resize(dummy_image, (227, 227))  # Match input shape
    gender, age = predict_age_gender(dummy_face, ageNet, genderNet)

    valid_genders = ['Male', 'Female']
    valid_ages = ['(0-2)', '(4-6)', '(8-12)','(13-17)', '(18-19)','(20-21)','(22-24)',
                  '(25-38)','(29-30)', '(33-43)', '(48-60)', '(60-100)']

    assert gender in valid_genders, f"Unexpected gender: {gender}"
    assert age in valid_ages, f"Unexpected age: {age}"
