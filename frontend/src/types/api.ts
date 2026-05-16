export interface SensorReading {
  altitude: number;
  mach_number: number;
  throttle_resolver_angle: number;
  lpc_outlet_temperature: number;
  hpc_outlet_temperature: number;
  lpt_outlet_temperature: number;
  hpc_outlet_pressure: number;
  physical_fan_speed: number;
  physical_core_speed: number;
  hpc_outlet_static_pressure: number;
  fuel_flow_ratio_ps30: number;
  corrected_fan_speed: number;
  corrected_core_speed: number;
  bypass_ratio: number;
  bleed_enthalpy: number;
  hpc_cooling_air_flow: number;
  lpt_cooling_air_flow: number;
}

export interface PredictRequest {
  engine_id: string;
  sequence: SensorReading[];
}

export interface PredictResponse {
  id: string;
  engine_id: string;
  rul_predicted: number;
  timestamp: string;
  unit: string;
}

export interface HistoryItem {
  id: string;
  engine_id: string;
  rul_predicted: number;
  timestamp: string;
}

export interface HistoryResponse {
  count: number;
  predictions: HistoryItem[];
}

export interface MetricsResponse {
  rmse: number;
  mae: number;
  units_evaluated: number;
}

export interface HealthResponse {
  status: string;
  project: string;
  version: string;
}
