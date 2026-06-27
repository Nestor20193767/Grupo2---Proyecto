from arrys.constants import CLASSIFIER_CLASSES, FS, GENERATOR_CLASSES, LATENT_DIM, SEQ_LEN


def test_reported_dimensions():
    assert FS == 500
    assert SEQ_LEN == 325
    assert LATENT_DIM == 32


def test_generator_and_classifier_classes_are_explicit():
    assert "Others" in GENERATOR_CLASSES
    assert "Others" not in CLASSIFIER_CLASSES
    assert len(GENERATOR_CLASSES) == 6
    assert len(CLASSIFIER_CLASSES) == 5
