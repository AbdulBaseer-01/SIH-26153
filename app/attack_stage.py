from __future__ import annotations

import pandas as pd


class AttackStageMapper:
    """
    Explainable ATT&CK-oriented interpretation layer.

    IMPORTANT:
    This mapper does not claim that the LSTM directly predicts
    MITRE ATT&CK techniques.

    It interprets the learned network anomaly signals together
    with observed infiltration telemetry.
    """

    def __init__(self):
        pass

    # ==========================================================
    # HELPERS
    # ==========================================================

    @staticmethod
    def _safe_float(value, default=0.0):

        try:
            if pd.isna(value):
                return default

            return float(value)

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _extract_feature_names(top_features):

        if top_features is None:
            return []

        names = []

        for item in top_features:

            if isinstance(item, dict):

                name = item.get(
                    "feature",
                    item.get(
                        "name",
                        ""
                    )
                )

            elif isinstance(item, (tuple, list)):

                if len(item) > 0:
                    name = item[0]
                else:
                    name = ""

            else:

                name = str(item)

            if name:
                names.append(str(name))

        return names

    # ==========================================================
    # MAIN CLASSIFIER
    # ==========================================================

    def classify(
        self,
        prediction_mse=0.0,
        forecast_score=0.0,
        infiltration_ratio=0.0,
        top_features=None,
        forecast_state="LOW",
    ):

        mse = self._safe_float(
            prediction_mse
        )

        score = self._safe_float(
            forecast_score
        )

        infiltration = self._safe_float(
            infiltration_ratio
        )

        feature_names = (
            self._extract_feature_names(
                top_features
            )
        )

        feature_text = " ".join(
            name.lower()
            for name in feature_names
        )

        # ======================================================
        # 1. CONFIRMED OBSERVED INFILTRATION
        # ======================================================

        if infiltration >= 0.50:

            return self._result(
                stage="Active Infiltration / Exploitation",
                confidence="High",

                evidence=[
                    (
                        f"Observed infiltration ratio: "
                        f"{infiltration:.2f}"
                    ),
                    (
                        f"World-model prediction error: "
                        f"{mse:.2f}"
                    ),
                    (
                        f"Forecast score: "
                        f"{score:.1f}/10"
                    ),
                ],

                explanation=(
                    "The observed network state contains a "
                    "substantial infiltration signal."
                ),

                mitre_context=(
                    "ATT&CK-oriented interpretation: the observed "
                    "traffic is consistent with an active compromise "
                    "or exploitation phase. Specific ATT&CK techniques "
                    "are not confirmed by this telemetry alone."
                ),
            )

        # ======================================================
        # 2. HIGH ANOMALY BUT NO OBSERVED INFILTRATION
        # ======================================================

        if score >= 8 or mse >= 5:

            return self._result(
                stage="Suspicious / Pre-Attack Anomaly",
                confidence="Medium",

                evidence=[
                    (
                        f"Elevated prediction error: "
                        f"{mse:.2f}"
                    ),
                    (
                        f"High forecast score: "
                        f"{score:.1f}/10"
                    ),
                    (
                        f"Forecast state: "
                        f"{forecast_state}"
                    ),
                    (
                        "Observed infiltration has not been "
                        "confirmed in the current state."
                    ),
                ],

                explanation=(
                    "The current network state differs substantially "
                    "from the dynamics learned from normal traffic, "
                    "but the available telemetry does not confirm "
                    "active infiltration."
                ),

                mitre_context=(
                    "ATT&CK-oriented interpretation: suspicious "
                    "pre-attack or anomalous behaviour requiring "
                    "investigation. No specific ATT&CK technique "
                    "is confirmed."
                ),
            )

        # ======================================================
        # 3. RECONNAISSANCE / DISCOVERY
        # ======================================================

        reconnaissance_terms = [
            "dst port",
            "destination port",
            "flow pkts/s",
            "fwd pkts/s",
            "pkt size",
            "flow count",
            "syn flag",
            "syn flag cnt",
        ]

        recon_hits = [
            term
            for term in reconnaissance_terms
            if term in feature_text
        ]

        if recon_hits and score >= 4:

            return self._result(
                stage="Reconnaissance / Discovery",
                confidence="Low",

                evidence=[
                    (
                        "Elevated traffic-rate or port-related "
                        "features are contributing to the anomaly."
                    ),
                    (
                        "Signals: "
                        + ", ".join(
                            recon_hits[:4]
                        )
                    ),
                    (
                        f"Forecast score: "
                        f"{score:.1f}/10"
                    ),
                ],

                explanation=(
                    "Traffic characteristics are consistent with "
                    "increased network probing or discovery activity."
                ),

                mitre_context=(
                    "ATT&CK-oriented interpretation: potentially "
                    "related to reconnaissance or discovery behaviour. "
                    "No specific ATT&CK technique is confirmed."
                ),
            )

        # ======================================================
        # 4. COMMAND & CONTROL-LIKE BEHAVIOUR
        # ======================================================

        c2_terms = [
            "iat",
            "flow iat",
            "fwd iat",
            "bwd iat",
            "active",
            "idle",
        ]

        c2_hits = [
            term
            for term in c2_terms
            if term in feature_text
        ]

        if c2_hits and score >= 4:

            return self._result(
                stage="Command & Control-Like Behaviour",
                confidence="Low",

                evidence=[
                    (
                        "Timing/inter-arrival features contribute "
                        "strongly to the current anomaly."
                    ),
                    (
                        "Signals: "
                        + ", ".join(
                            c2_hits[:4]
                        )
                    ),
                    (
                        f"Forecast score: "
                        f"{score:.1f}/10"
                    ),
                ],

                explanation=(
                    "Temporal flow behaviour is anomalous and may "
                    "warrant investigation for command-and-control-like "
                    "activity."
                ),

                mitre_context=(
                    "ATT&CK-oriented interpretation: potentially "
                    "consistent with command-and-control behaviour. "
                    "No specific ATT&CK technique is confirmed."
                ),
            )

        # ======================================================
        # 5. MODERATE ANOMALY
        # ======================================================

        if (
            forecast_state in {
                "HIGH",
                "MEDIUM"
            }
            or score >= 4
        ):

            return self._result(
                stage="Suspicious Network Activity",
                confidence="Low",

                evidence=[
                    (
                        f"Forecast state: "
                        f"{forecast_state}"
                    ),
                    (
                        f"Forecast score: "
                        f"{score:.1f}/10"
                    ),
                    (
                        f"Prediction error: "
                        f"{mse:.2f}"
                    ),
                ],

                explanation=(
                    "An elevated network anomaly is present, "
                    "but the available telemetry is insufficient "
                    "to identify a specific attack stage."
                ),

                mitre_context=(
                    "ATT&CK-oriented interpretation: suspicious "
                    "network behaviour requiring investigation. "
                    "No technique is confirmed."
                ),
            )

        # ======================================================
        # 6. NORMAL
        # ======================================================

        return self._result(
            stage="Normal",
            confidence="High",

            evidence=[
                "No strong attack-stage signal detected.",
                (
                    f"Forecast state: "
                    f"{forecast_state}"
                ),
                (
                    f"Forecast score: "
                    f"{score:.1f}/10"
                ),
            ],

            explanation=(
                "The current network behaviour is close to "
                "the dynamics learned from normal traffic."
            ),

            mitre_context=(
                "No ATT&CK-oriented attack-stage signal is "
                "currently identified."
            ),
        )

    # ==========================================================
    # RESULT FORMATTER
    # ==========================================================

    @staticmethod
    def _result(
        stage,
        confidence,
        evidence,
        explanation,
        mitre_context,
    ):

        return {
            "stage": stage,
            "confidence": confidence,
            "evidence": evidence,
            "explanation": explanation,
            "mitre_context": mitre_context,
        }


# ==============================================================
# CLI TEST
# ==============================================================

if __name__ == "__main__":

    mapper = AttackStageMapper()

    result = mapper.classify(
        prediction_mse=0.75,
        forecast_score=10,
        infiltration_ratio=0.0,

        top_features=[
            ("Fwd Pkts/s", 2.61),
            ("Flow Pkts/s", 2.08),
            ("Dst Port", 1.71),
        ],

        forecast_state="HIGH",
    )

    print()
    print("========== ATTACK STAGE TEST ==========")

    print(
        "Stage:",
        result["stage"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )

    print()
    print("Evidence:")

    for item in result["evidence"]:
        print(
            "-",
            item
        )

    print()
    print("Explanation:")
    print(
        result["explanation"]
    )

    print()
    print("MITRE Context:")
    print(
        result["mitre_context"]
    )

    print(
        "========================================"
    )