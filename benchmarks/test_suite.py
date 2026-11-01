"""Comprehensive Pytest Verification Suite for Agritech Subsystem.

Verifies:
1. Problem 1: Latent Disease Outbreak early warning & exact TreeSHAP feature attribution.
2. Problem 2: Meteorological Spray-Lock Override (Precipitation > 10mm & Wind > 18km/h).
3. Problem 3: Subspace Isolation & Zero Cross-Crop Chemical Contamination.
4. Problem 4: Out-of-Distribution Confidence Gate (< 0.60) blocking chemical synthesis.
5. Problem 5: Root-Zone Leaching & Hypoxia Gate (Soil Moisture > 0.32 m^3/m^3).
6. Zero-Downtime Hot-Swapping between Baseline XGBoost and Calibrated GBDT Triad.
"""

import pytest

from app.schemas.swarm_schemas import (
    RAGQueryResponse,
    StandardizedRiskOutput,
    SwarmExecutionResult,
    WeatherTelemetryInput,
)
from app.services.rag_engine.retriever import DeterministicICARRAG
from app.services.risk_engine.base import get_predictor
from app.services.risk_engine.calibrated_triad import CalibratedTriadPredictor
from app.services.risk_engine.constants import (
    CONFIDENCE_GATE_TAU,
)
from app.services.risk_engine.pipeline import UnifiedAgronomicEngine
from app.services.risk_engine.xgb_baseline import BaselineXGBPredictor


@pytest.fixture(scope="session")
def trained_xgb_predictor():
    """Loads or fits production BaselineXGBPredictor."""
    try:
        pred = get_predictor("baseline_xgb", "artifacts/xgb_baseline.joblib")
    except Exception:
        from benchmarks.train_and_evaluate import generate_agronomic_dataset

        X, y = generate_agronomic_dataset(samples_per_class=100)
        pred = BaselineXGBPredictor(n_estimators=50, max_depth=3).fit(X, y)
    return pred


@pytest.fixture(scope="session")
def trained_triad_predictor():
    """Loads or fits production CalibratedTriadPredictor."""
    try:
        pred = get_predictor("calibrated_triad", "artifacts/calibrated_triad.joblib")
    except Exception:
        from benchmarks.train_and_evaluate import generate_agronomic_dataset

        X, y = generate_agronomic_dataset(samples_per_class=100)
        pred = CalibratedTriadPredictor().fit(X, y)
    return pred


@pytest.fixture(scope="session")
def rag_retriever():
    """Instantiates DeterministicICARRAG retriever."""
    return DeterministicICARRAG()


# ==============================================================================
# TEST 1: Latent Epidemiological Outbreak Detection & TreeSHAP Attribution (P1)
# ==============================================================================
def test_farmer_problem_1_latent_outbreak_and_treeshap(trained_triad_predictor):
    """Verifies that latent epidemiological conditions for Potato Late Blight

    trigger early warning, and TreeSHAP attributes humidity and temperature as primary drivers.
    """
    telemetry = WeatherTelemetryInput(
        crop_name="potato",
        temp_min_c=12.5,
        temp_max_c=19.0,
        humidity_3d_avg=93.5,
        rain_7d_mm=45.0,
        soil_moisture=0.28,
        rain_next_24h_mm=2.0,
        wind_speed_kmh=8.0,
        soil_moisture_root_zone=0.26,
    )

    risk_output: StandardizedRiskOutput = trained_triad_predictor.predict_and_explain(
        telemetry
    )

    assert risk_output.predicted_disease == "late_blight", (
        f"Expected 'late_blight', got '{risk_output.predicted_disease}'"
    )
    assert risk_output.calibrated_probability >= 0.70, (
        f"Expected calibrated probability >= 0.70, got {risk_output.calibrated_probability}"
    )
    assert risk_output.risk_level in ["HIGH", "SEVERE"]

    # Verify TreeSHAP attribution
    assert len(risk_output.shap_top_drivers) == 3
    driver_names = [d.feature_name for d in risk_output.shap_top_drivers]
    assert "humidity_3d_avg" in driver_names or "temp_min_c" in driver_names

    top_driver = risk_output.shap_top_drivers[0]
    assert top_driver.shap_value > 0.0, (
        "Top risk driver must have positive Shapley score"
    )
    assert top_driver.effect == "increases_risk"


def test_treeshap_attributions_follow_each_crop_weather_input(trained_xgb_predictor):
    """Different model inputs must produce their own disease-class explanations."""
    wet_rice_telemetry = WeatherTelemetryInput(
        crop_name="rice",
        temp_min_c=12.0,
        temp_max_c=20.0,
        humidity_3d_avg=95.0,
        rain_7d_mm=50.0,
        soil_moisture=0.35,
    )
    hot_dry_wheat_telemetry = WeatherTelemetryInput(
        crop_name="wheat",
        temp_min_c=30.0,
        temp_max_c=40.0,
        humidity_3d_avg=25.0,
        rain_7d_mm=0.0,
        soil_moisture=0.08,
    )

    wet_rice_risk = trained_xgb_predictor.predict_and_explain(wet_rice_telemetry)
    hot_dry_wheat_risk = trained_xgb_predictor.predict_and_explain(
        hot_dry_wheat_telemetry
    )

    assert wet_rice_telemetry.to_feature_vector() != (
        hot_dry_wheat_telemetry.to_feature_vector()
    )
    assert wet_rice_risk.raw_shap_vector != hot_dry_wheat_risk.raw_shap_vector


# ==============================================================================
# TEST 2: Meteorological Spray-Lock Override (P2)
# ==============================================================================
def test_farmer_problem_2_spray_lock_rain_and_wind(
    trained_xgb_predictor, rag_retriever
):
    """Verifies that rain > 10mm or wind > 18km/h triggers deterministic spray lock."""
    engine = UnifiedAgronomicEngine(
        predictor=trained_xgb_predictor, rag_retriever=rag_retriever
    )

    # Case A: Heavy rain forecast
    rain_telemetry = WeatherTelemetryInput(
        crop_name="potato",
        temp_min_c=14.0,
        temp_max_c=22.0,
        humidity_3d_avg=88.0,
        rain_7d_mm=30.0,
        soil_moisture=0.25,
        rain_next_24h_mm=15.5,  # Exceeds 10.0 mm
        wind_speed_kmh=10.0,
        soil_moisture_root_zone=0.25,
    )
    res_rain = engine.execute(
        rain_telemetry, farmer_query="What fungicide should I spray?"
    )
    assert res_rain.safety_guardrails.hold_spray is True
    assert any(
        "[CRITICAL_HOLD_SPRAY]" in r
        for r in res_rain.safety_guardrails.override_reasons
    )
    assert res_rain.safety_guardrails.safe_spray_window_hours is not None

    # Case B: High wind speed
    wind_telemetry = WeatherTelemetryInput(
        crop_name="potato",
        temp_min_c=14.0,
        temp_max_c=22.0,
        humidity_3d_avg=88.0,
        rain_7d_mm=30.0,
        soil_moisture=0.25,
        rain_next_24h_mm=0.0,
        wind_speed_kmh=24.5,  # Exceeds 18.0 km/h
        soil_moisture_root_zone=0.25,
    )
    res_wind = engine.execute(
        wind_telemetry, farmer_query="What fungicide should I spray?"
    )
    assert res_wind.safety_guardrails.hold_spray is True
    assert any(
        "drift threshold" in r for r in res_wind.safety_guardrails.override_reasons
    )


# ==============================================================================
# TEST 3: Subspace Isolation & Cross-Crop Chemical Contamination Prevention (P3)
# ==============================================================================
def test_farmer_problem_3_subspace_isolation(rag_retriever):
    """Verifies that querying Potato Late Blight NEVER retrieves Rice or Tomato chemicals."""
    response: RAGQueryResponse = rag_retriever.retrieve(
        crop_name="potato",
        disease_or_pest="late_blight",
        farmer_query="What chemical fungicide and dilution dosage to cure late blight?",
    )

    assert response.status == "SUCCESS"
    assert len(response.citations) > 0

    forbidden_chemicals = [
        "Tricyclazole",
        "Streptocycline",
        "Imidacloprid",
        "Propiconazole",
    ]

    for cit in response.citations:
        assert cit.crop_name == "potato"
        assert cit.disease_or_pest == "late_blight"
        for forbidden in forbidden_chemicals:
            assert forbidden.lower() not in cit.active_ingredient.lower(), (
                f"Cross-contamination detected! Found {forbidden} in potato late blight citation."
            )
        # Provenance verification
        assert "CPRI" in cit.source_doc_title
        assert cit.page_number > 0
        assert cit.gov_url.startswith("https://")


# ==============================================================================
# TEST 4: Out-of-Distribution Confidence Gate (P4)
# ==============================================================================
def test_farmer_problem_4_ood_confidence_gate(rag_retriever):
    """Verifies that irrelevant queries are blocked by the confidence gate (< 0.60)."""
    irrelevant_query = (
        "What is the capital city of France and how to bake chocolate cake?"
    )

    response: RAGQueryResponse = rag_retriever.retrieve(
        crop_name="potato",
        disease_or_pest="late_blight",
        farmer_query=irrelevant_query,
    )

    assert response.status == "LOW_CONFIDENCE_GUARDRAIL_TRIGGERED"
    assert response.chemical_synthesis_blocked is True
    assert response.top_similarity_score < CONFIDENCE_GATE_TAU
    assert len(response.citations) == 0
    assert "Low Confidence Guardrail" in response.guardrail_message


# ==============================================================================
# TEST 5: Root-Zone Leaching & Hypoxia Gate (P5)
# ==============================================================================
def test_farmer_problem_5_soil_moisture_leaching_gate(
    trained_xgb_predictor, rag_retriever
):
    """Verifies that root-zone soil moisture > 0.32 triggers irrigation & urea lock."""
    engine = UnifiedAgronomicEngine(
        predictor=trained_xgb_predictor, rag_retriever=rag_retriever
    )

    leaching_telemetry = WeatherTelemetryInput(
        crop_name="wheat",
        temp_min_c=12.0,
        temp_max_c=22.0,
        humidity_3d_avg=60.0,
        rain_7d_mm=5.0,
        soil_moisture=0.20,
        rain_next_24h_mm=0.0,
        wind_speed_kmh=10.0,
        soil_moisture_root_zone=0.36,  # Exceeds field capacity 0.32
    )

    result = engine.execute(
        leaching_telemetry, farmer_query="Should I irrigate and apply urea?"
    )
    assert result.safety_guardrails.hold_irrigation is True
    assert any(
        "[CRITICAL_HOLD_IRRIGATION]" in r
        for r in result.safety_guardrails.override_reasons
    )
    assert any(
        "nitrogen" in r.lower() and "leach" in r.lower()
        for r in result.safety_guardrails.override_reasons
    )


# ==============================================================================
# TEST 6: Zero-Downtime Hot-Swapping Architecture
# ==============================================================================
def test_zero_downtime_hot_swapping(
    trained_xgb_predictor, trained_triad_predictor, rag_retriever
):
    """Verifies that risk models can be hot-swapped without altering engine contracts."""
    engine = UnifiedAgronomicEngine(
        predictor=trained_xgb_predictor, rag_retriever=rag_retriever
    )

    telemetry = WeatherTelemetryInput(
        crop_name="rice",
        temp_min_c=22.0,
        temp_max_c=28.0,
        humidity_3d_avg=94.0,
        rain_7d_mm=45.0,
        soil_moisture=0.31,
        rain_next_24h_mm=0.0,
        wind_speed_kmh=12.0,
        soil_moisture_root_zone=0.28,
    )

    # 1. Run inference with Baseline XGBoost
    res_v1 = engine.execute(telemetry, farmer_query="Symptoms of rice blast")
    assert res_v1.risk_assessment.model_version == "1.0.0-xgb"
    assert isinstance(res_v1, SwarmExecutionResult)

    # 2. Hot-Swap to Calibrated Triad
    engine.set_predictor(trained_triad_predictor)

    # 3. Run inference on identical input with zero code change
    res_v2 = engine.execute(telemetry, farmer_query="Symptoms of rice blast")
    assert res_v2.risk_assessment.model_version == "2.0.0-triad-calibrated"
    assert isinstance(res_v2, SwarmExecutionResult)

    # Contract integrity
    assert res_v1.crop_name == res_v2.crop_name
    assert (
        len(res_v1.risk_assessment.raw_shap_vector)
        == len(res_v2.risk_assessment.raw_shap_vector)
        == 6
    )
