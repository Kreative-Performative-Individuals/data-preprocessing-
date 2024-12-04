import pytest
import requests

BASE_URL = "http://localhost:8003"

def test_consumption_working():
    machine_name = "Large Capacity Cutting Machine 1"
    asset_id = "ast-yhc1llzju2eT"
    kpi = "consumption"
    operation = "working"
    timestamp_start = "2024-05-02T00:00:00"
    timestamp_end = "2024-05-02T00:00:00"
    
    url = f"{BASE_URL}/get_forecasting/?machine_name={machine_name}&asset_id={asset_id}&kpi={kpi}&operation={operation}&timestamp_start={timestamp_start}&timestamp_end={timestamp_end}"
    
    response = requests.get(url)
    
    assert response.status_code == 200
    assert "application/json" in response.headers["Content-Type"]
    data = response.json()
    assert data['avg'] == pytest.approx(0.08587782626918902)

def test_idle_time_idle():
    machine_name = "Large Capacity Cutting Machine 1"
    asset_id = "ast-yhc1llzju2eT"
    kpi = "idle_time"
    operation = "idle"
    timestamp_start = "2024-05-02T00:00:00"
    timestamp_end = "2024-05-02T00:00:00"
    
    url = f"{BASE_URL}/get_forecasting/?machine_name={machine_name}&asset_id={asset_id}&kpi={kpi}&operation={operation}&timestamp_start={timestamp_start}&timestamp_end={timestamp_end}"
    
    response = requests.get(url)
    
    assert response.status_code == 200
    assert "application/json" in response.headers["Content-Type"]
    data = response.json()
    assert data['min'] == 0
    assert data['max'] == 0

if __name__ == "__main__":
    pytest.main()
