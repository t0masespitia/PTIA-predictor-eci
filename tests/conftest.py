import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

sys.path.append(".")

from app.models.entities.prediction_record import Base
from main import app


def make_reading(value: float = 0.5) -> dict:
    return {
        "altitude": value,
        "mach_number": value,
        "throttle_resolver_angle": value,
        "lpc_outlet_temperature": value,
        "hpc_outlet_temperature": value,
        "lpt_outlet_temperature": value,
        "hpc_outlet_pressure": value,
        "physical_fan_speed": value,
        "physical_core_speed": value,
        "hpc_outlet_static_pressure": value,
        "fuel_flow_ratio_ps30": value,
        "corrected_fan_speed": value,
        "corrected_core_speed": value,
        "bypass_ratio": value,
        "bleed_enthalpy": value,
        "hpc_cooling_air_flow": value,
        "lpt_cooling_air_flow": value,
    }


@pytest.fixture
def valid_window() -> list[dict]:
    return [make_reading() for _ in range(30)]


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()


@pytest.fixture
def client(session_factory):
    with TestClient(app) as client:
        client.app.state.session_factory = session_factory
        yield client
