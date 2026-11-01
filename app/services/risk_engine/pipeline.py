"""Unified Agronomic Engine: Orchestrating Risk Inference, Subspace RAG, and Safety Overrides.

Chains the ML Predictor (Baseline XGBoost or Calibrated Triad) with Deterministic ICAR RAG
and Cross-Node Meteorological Safety Overrides (Spray Lock & Leaching Gate).
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from app.schemas.swarm_schemas import (
    SafetyOverride,
    SwarmExecutionResult,
    WeatherTelemetryInput,
)
from app.services.risk_engine.base import BaseDiseasePredictor
from app.services.risk_engine.constants import (
    SOIL_MOISTURE_LEACHING_THRESHOLD,
    SPRAY_RAIN_THRESHOLD_MM,
    SPRAY_WIND_THRESHOLD_KMH,
)

if TYPE_CHECKING:
    from app.services.rag_engine.retriever import DeterministicICARRAG


class UnifiedAgronomicEngine:
    """End-to-End Orchestrator for Agronomic Intelligence and Guardrails."""

    def __init__(
        self,
        predictor: BaseDiseasePredictor | None = None,
        rag_retriever: Optional["DeterministicICARRAG"] = None,
    ):
        self.predictor = predictor
        if rag_retriever is not None:
            self.rag = rag_retriever
        else:
            from app.services.rag_engine.retriever import DeterministicICARRAG

            self.rag = DeterministicICARRAG()

    def set_predictor(self, predictor: BaseDiseasePredictor) -> None:
        """Hot-swaps the underlying risk model at runtime with zero downtime."""
        self.predictor = predictor

    def evaluate_safety_overrides(
        self, telemetry: WeatherTelemetryInput
    ) -> SafetyOverride:
        """Deterministic Cross-Node Guardrails: Evaluates Rain/Wind Spray-Lock & Soil Leaching Gate."""
        hold_spray = False
        hold_irrigation = False
        override_reasons: list[str] = []
        safe_window: str | None = None

        # 1. Rain Wash-Off Guardrail (Farmer Problem 2)
        if telemetry.rain_next_24h_mm >= SPRAY_RAIN_THRESHOLD_MM:
            hold_spray = True
            override_reasons.append(
                f"[CRITICAL_HOLD_SPRAY] Incoming precipitation forecast ({telemetry.rain_next_24h_mm:.1f} mm in 24h) "
                f"exceeds safety threshold ({SPRAY_RAIN_THRESHOLD_MM:.1f} mm). "
                f"Foliar application now will wash off 80%+ of chemical into the soil, wasting input costs."
            )

        # 2. Wind Drift Guardrail (Farmer Problem 2)
        if telemetry.wind_speed_kmh >= SPRAY_WIND_THRESHOLD_KMH:
            hold_spray = True
            override_reasons.append(
                f"[CRITICAL_HOLD_SPRAY] Surface wind speed ({telemetry.wind_speed_kmh:.1f} km/h) "
                f"exceeds drift threshold ({SPRAY_WIND_THRESHOLD_KMH:.1f} km/h). "
                f"Chemical droplet drift presents severe phytotoxicity and environmental runoff hazard."
            )

        # 3. Root-Zone Hypoxia & Nitrogen Leaching Gate (Farmer Problem 5)
        max_moisture = max(telemetry.soil_moisture, telemetry.soil_moisture_root_zone)
        if max_moisture >= SOIL_MOISTURE_LEACHING_THRESHOLD:
            hold_irrigation = True
            override_reasons.append(
                f"[CRITICAL_HOLD_IRRIGATION] Root-zone soil moisture ({max_moisture:.2f} m^3/m^3) "
                f"exceeds field capacity ({SOIL_MOISTURE_LEACHING_THRESHOLD:.2f} m^3/m^3). "
                f"Further irrigation or Urea top-dressing causes root zone oxygen starvation (hypoxia) "
                f"and severe nitrogen leaching into groundwater."
            )

        if hold_spray:
            safe_window = (
                "Postpone spray application by 24-36 hours. "
                "Safe spray window: Tomorrow after 10:00 AM once wind drops below 12 km/h "
                "and rain probability is under 15%."
            )
            action_rec = (
                "HOLD SPRAY IMMEDIATELY. Do not apply chemicals today due to adverse weather. "
                "Review safe spray window."
            )
        elif hold_irrigation:
            action_rec = (
                "SUSPEND IRRIGATION & UREA APPLICATION. Soil is currently water-saturated. "
                "Allow 48 hours for drainage."
            )
        else:
            action_rec = "CONDITIONS PERMISSIVE: Weather conditions are optimal for recommended agronomic actions."

        return SafetyOverride(
            hold_spray=hold_spray,
            hold_irrigation=hold_irrigation,
            override_reasons=override_reasons,
            safe_spray_window_hours=safe_window,
            action_recommendation=action_rec,
        )

    def execute(
        self,
        telemetry: WeatherTelemetryInput,
        farmer_query: str,
    ) -> SwarmExecutionResult:
        """Executes full intelligence loop: Guardrails -> Risk ML -> Subspace RAG -> Synthesis."""
        if self.predictor is None:
            raise RuntimeError(
                "No predictor configured. Please set a predictor before calling execute()."
            )

        # 1. Deterministic Safety Guardrails Check
        safety_guardrails = self.evaluate_safety_overrides(telemetry)

        # 2. Epidemiological Risk Model Inference & TreeSHAP Attribution
        risk_output = self.predictor.predict_and_explain(telemetry)

        # 3. Deterministic Subspace RAG Retrieval
        target_condition = risk_output.predicted_disease
        rag_response = self.rag.retrieve(
            crop_name=telemetry.crop_name,
            disease_or_pest=target_condition,
            farmer_query=farmer_query,
        )

        # 4. Synthesize Executive Vernacular-Ready Advisory
        advisory_lines = [
            f"=== EXPERIMENTAL WEATHER-RISK ASSESSMENT ({telemetry.crop_name.upper()}) ===",
            f"Condition Assessment: {risk_output.predicted_disease.replace('_', ' ').title()}",
            f"Calibrated Outbreak Probability: {risk_output.calibrated_probability * 100:.1f}% ({risk_output.risk_level} RISK)",
            f"Model Engine: {risk_output.model_version}",
            "",
            "--- METEOROLOGICAL DRIVERS (TreeSHAP Feature Attribution) ---",
        ]

        for d in risk_output.shap_top_drivers:
            direction = (
                "Elevating Risk (+)" if d.shap_value > 0 else "Mitigating Risk (-)"
            )
            advisory_lines.append(
                f" - {d.feature_name} = {d.feature_value:.1f}: SHAP score {d.shap_value:+.4f} ({direction})"
            )

        advisory_lines.append("")
        advisory_lines.append("--- SAFETY OVERRIDES & WEATHER LOCKS ---")
        if safety_guardrails.override_reasons:
            for r in safety_guardrails.override_reasons:
                advisory_lines.append(f" ! {r}")
            if safety_guardrails.safe_spray_window_hours:
                advisory_lines.append(
                    f" -> Recommended Safe Window: {safety_guardrails.safe_spray_window_hours}"
                )
        else:
            advisory_lines.append(
                " [OK] No weather locks active. Conditions suitable for field intervention."
            )

        advisory_lines.append("")
        advisory_lines.append("--- ICAR SCIENTIFIC REMEDY & CITATION ---")
        if rag_response.chemical_synthesis_blocked:
            advisory_lines.append(
                f" [GUARDRAIL ACTIVE] {rag_response.guardrail_message}"
            )
        elif safety_guardrails.hold_spray:
            advisory_lines.append(
                f" [SPRAY WITHHELD] Chemical remedy retrieved ({rag_response.citations[0].active_ingredient}), "
                f"but application is temporarily LOCKED by rain/wind guardrail."
            )
            advisory_lines.append(
                f" Citation: {rag_response.citations[0].source_doc_title} (Page {rag_response.citations[0].page_number})"
            )
        elif rag_response.citations:
            top_cit = rag_response.citations[0]
            advisory_lines.append(f" - Active Ingredient: {top_cit.active_ingredient}")
            advisory_lines.append(f" - Recommended Dosage: {top_cit.dilution_dosage}")
            advisory_lines.append(
                f" - Pre-Harvest Interval (PHI): {top_cit.phi_days} Days"
            )
            advisory_lines.append(
                f" - Source Provenance: {top_cit.source_doc_title} (p. {top_cit.page_number})"
            )
            advisory_lines.append(f" - Official Portal: {top_cit.gov_url}")
        else:
            advisory_lines.append(
                " No specific ICAR chemical protocol required for current stage."
            )

        executive_advisory = "\n".join(advisory_lines)

        return SwarmExecutionResult(
            crop_name=telemetry.crop_name,
            risk_assessment=risk_output,
            rag_guidance=rag_response,
            safety_guardrails=safety_guardrails,
            executive_advisory=executive_advisory,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
