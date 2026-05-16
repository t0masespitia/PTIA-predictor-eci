import sys

import numpy as np

sys.path.append(".")

from app.api.dtos.request.sensor_reading import SensorReading
from app.services.model_service import FEATURE_ORDER, ModelService
from tests.conftest import make_reading


def test_sequence_to_array_uses_descriptive_feature_order():
    reading = make_reading()
    for index, feature_name in enumerate(FEATURE_ORDER, start=1):
        reading[feature_name] = float(index)

    sequence = [SensorReading(**reading)]

    arr = ModelService._sequence_to_array(sequence)

    assert arr.shape == (1, len(FEATURE_ORDER))
    assert arr.dtype == np.float32
    assert arr.tolist()[0] == [float(index) for index in range(1, len(FEATURE_ORDER) + 1)]
