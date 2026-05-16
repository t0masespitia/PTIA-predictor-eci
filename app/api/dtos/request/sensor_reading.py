from pydantic import BaseModel


class SensorReading(BaseModel):
    altitude: float
    mach_number: float
    throttle_resolver_angle: float
    lpc_outlet_temperature: float
    hpc_outlet_temperature: float
    lpt_outlet_temperature: float
    hpc_outlet_pressure: float
    physical_fan_speed: float
    physical_core_speed: float
    hpc_outlet_static_pressure: float
    fuel_flow_ratio_ps30: float
    corrected_fan_speed: float
    corrected_core_speed: float
    bypass_ratio: float
    bleed_enthalpy: float
    hpc_cooling_air_flow: float
    lpt_cooling_air_flow: float
