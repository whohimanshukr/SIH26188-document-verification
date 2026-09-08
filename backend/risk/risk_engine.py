"""Deterministic risk assessment engine for document verification pipeline evidence."""

from typing import Any, Literal, TypedDict


class RiskEvaluationResult(TypedDict):
    """Normalized risk evaluation output."""

    risk_score: int
    decision: Literal["LOW", "MEDIUM", "HIGH"]
    reasons: list[str]


class RiskEngine:
    """Evaluate pipeline verification evidence into a normalized risk score and decision.

    Note on Prototype Design:
    This scoring system is an explainable heuristic for demonstration purposes.
    It flags discrepancies between the visual zone, MRZ, and image forensics.
    It does not constitute legal or biometric proof of document authenticity or fraud.
    """

    # Scoring weights for specific deterministic rules
    WEIGHT_PASSPORT_MISMATCH = 25
    WEIGHT_DOB_MISMATCH = 20
    WEIGHT_NAME_MISMATCH = 20
    WEIGHT_NATIONALITY_MISMATCH = 15
    WEIGHT_METADATA_SUSPICIOUS = 10
    WEIGHT_ELA_SUSPICIOUS = 10
    WEIGHT_MRZ_UNAVAILABLE = 10

    MAX_RISK_SCORE = 100

    def evaluate(self, pipeline_result: dict[str, Any]) -> RiskEvaluationResult:
        """Calculate the total risk score and decision category from pipeline output.

        Args:
            pipeline_result: Output dictionary from DocumentVerificationPipeline.verify().

        Returns:
            RiskEvaluationResult containing risk_score, decision, and explanatory reasons.
        """
        score = 0
        reasons: list[str] = []

        mrz_section = pipeline_result.get("mrz", {})
        forensics_section = pipeline_result.get("forensics", {})

        # 1. MRZ Availability
        # If MRZ could not be detected or parsed, apply mild penalty.
        # Note: Some ID documents or cropped cards naturally lack MRZ.
        if mrz_section.get("available") is False:
            score += self.WEIGHT_MRZ_UNAVAILABLE
            reasons.append(f"MRZ unavailable (+{self.WEIGHT_MRZ_UNAVAILABLE})")

        # 2. MRZ Field Consistency Checks
        # A value of False indicates explicit conflict between visual zone and MRZ.
        # A value of None means the field could not be extracted for comparison (no penalty).
        matches = mrz_section.get("matches", {})

        if matches.get("passport_number") is False:
            score += self.WEIGHT_PASSPORT_MISMATCH
            reasons.append(
                f"Passport number does not match MRZ (+{self.WEIGHT_PASSPORT_MISMATCH})"
            )

        if matches.get("date_of_birth") is False:
            score += self.WEIGHT_DOB_MISMATCH
            reasons.append(
                f"Date of birth does not match MRZ (+{self.WEIGHT_DOB_MISMATCH})"
            )

        if matches.get("name") is False:
            score += self.WEIGHT_NAME_MISMATCH
            reasons.append(
                f"Name does not match MRZ (+{self.WEIGHT_NAME_MISMATCH})"
            )

        if matches.get("nationality") is False:
            score += self.WEIGHT_NATIONALITY_MISMATCH
            reasons.append(
                f"Nationality does not match MRZ (+{self.WEIGHT_NATIONALITY_MISMATCH})"
            )

        # 3. Forensics Checks
        # Flags editing software or high compression discrepancies.
        metadata_res = forensics_section.get("metadata", {})
        if metadata_res.get("suspicious") is True:
            score += self.WEIGHT_METADATA_SUSPICIOUS
            reasons.append(
                f"Suspicious metadata detected (+{self.WEIGHT_METADATA_SUSPICIOUS})"
            )

        ela_res = forensics_section.get("ela", {})
        if ela_res.get("suspicious") is True:
            score += self.WEIGHT_ELA_SUSPICIOUS
            reasons.append(
                f"Suspicious ELA compression discrepancy (+{self.WEIGHT_ELA_SUSPICIOUS})"
            )

        # Cap score at maximum bounds
        final_score = min(score, self.MAX_RISK_SCORE)

        # Decision threshold mapping:
        # 0–20   -> LOW
        # 21–50  -> MEDIUM
        # 51–100 -> HIGH
        if final_score <= 20:
            decision: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
        elif final_score <= 50:
            decision = "MEDIUM"
        else:
            decision = "HIGH"

        if not reasons:
            reasons = ["All available verification checks passed"]

        return {
            "risk_score": final_score,
            "decision": decision,
            "reasons": reasons,
        }

