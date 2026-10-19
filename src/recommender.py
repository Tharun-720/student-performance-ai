"""Rule-based recommendation engine.

The rules look at the student's inputs AND the model's predicted support tier.
Recommendations are decision support for a teacher/tutor, not verdicts about a
student. Every rule is listed in docs/TECHNICAL_REPORT (section "Recommendation rules").
"""
from typing import Any, Dict, List, Optional

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}
MAX_PRIORITY_ACTIONS = 4

# Thresholds (documented). Absences are the recorded count, not a percentage.
ABSENCES_CRITICAL = 20
ABSENCES_MEDIUM = 10
LOW_GRADE_BELOW = 10      # latest recorded grade (G2) under the pass mark of 10/20
NEAR_PASS_BELOW = 12      # 10 <= G2 < 12: close to the pass mark
DECLINE_CRITICAL = -3     # grade drop G1 -> G2 of 3+ points
DECLINE_MEDIUM = -1

# Resource links are examples and were not checked automatically; verify before use.
RESOURCES = {
    "Mathematics": [
        {"name": "Khan Academy - Mathematics", "url": "https://www.khanacademy.org/math",
         "use": "Practice exercises by topic with instant feedback."},
        {"name": "Teacher/tutor practice sets", "url": None,
         "use": "Ask the subject teacher for past-paper style questions on the weakest topic."},
    ],
    "Portuguese": [
        {"name": "RTP Ensina", "url": "https://ensina.rtp.pt/",
         "use": "Free Portuguese-language learning videos and activities."},
        {"name": "Teacher/tutor practice sets", "url": None,
         "use": "Ask the subject teacher for reading and writing exercises with feedback."},
    ],
}


def _gap(gid, category, severity, finding, action, detail):
    return {"id": gid, "category": category, "severity": severity, "finding": finding,
            "title": action, "detail": detail}


def diagnose(student: Dict[str, Any], predicted_score_pct: Optional[float] = None,
             support_level: Optional[str] = None) -> List[Dict]:
    """Return the list of identified gaps (unsorted)."""
    gaps = []
    absences = student.get("absences")
    g1, g2 = student.get("G1"), student.get("G2")
    studytime = student.get("studytime")
    failures = student.get("failures")
    goout, freetime = student.get("goout"), student.get("freetime")

    # 1. Predicted outcome (model output; a risk indicator, not a confirmed result)
    if support_level == "High Support":
        shown = f"; predicted score about {predicted_score_pct:.0f}%" if predicted_score_pct is not None else ""
        gaps.append(_gap("pred_high", "Predicted outcome", "Critical",
                         f"The classifier places this student in the High Support tier (final grade below the "
                         f"pass mark of 10/20, i.e. 50%{shown}).",
                         "Arrange early academic support",
                         "Discuss this prediction with the student's teacher. Agree on a short catch-up plan with "
                         "weekly check-ins and re-assess after the next test."))
    elif support_level == "Medium Support":
        shown = f" (predicted score about {predicted_score_pct:.0f}%)" if predicted_score_pct is not None else ""
        gaps.append(_gap("pred_medium", "Predicted outcome", "Medium",
                         f"The classifier places this student in the Medium Support tier, i.e. near the pass "
                         f"mark{shown}.",
                         "Monitor progress and target weak topics",
                         "Review the last two assessments with the teacher and pick one or two topics to revise "
                         "before the next test."))

    # 2. Current attainment
    if g2 is not None and g2 < LOW_GRADE_BELOW:
        gaps.append(_gap("low_grade", "Current attainment", "Critical",
                         f"Latest recorded grade G2 = {g2:g}/20 is below the pass mark of 10.",
                         "Rebuild core skills on the weakest topics",
                         "Ask the teacher which topics cost the most marks in the last assessment and practise "
                         "those with worked examples, 20-30 minutes per day."))
    elif g2 is not None and g2 < NEAR_PASS_BELOW:
        gaps.append(_gap("near_pass", "Current attainment", "Medium",
                         f"Latest recorded grade G2 = {g2:g}/20 is close to the pass mark.",
                         "Secure the pass with focused revision",
                         "Use past questions to find recurring mistakes and fix them before the final assessment."))

    # 3. Trend
    if g1 is not None and g2 is not None:
        trend = g2 - g1
        if trend <= DECLINE_CRITICAL:
            gaps.append(_gap("trend_critical", "Grade trend", "High",
                             f"Grade fell by {abs(trend):g} points from G1 ({g1:g}) to G2 ({g2:g}).",
                             "Find out what changed between the two periods",
                             "Have a short conversation with the student to identify the cause (content, health, "
                             "workload, home situation) before choosing an intervention."))
        elif trend <= DECLINE_MEDIUM:
            gaps.append(_gap("trend_medium", "Grade trend", "Medium",
                             f"Grade slipped by {abs(trend):g} point(s) from G1 to G2.",
                             "Watch the trend over the next assessment",
                             "Check the next test result; if it falls again, treat it as a priority."))

    # 4. Recorded absences (count, not a verified attendance percentage)
    if absences is not None:
        if absences >= ABSENCES_CRITICAL:
            gaps.append(_gap("absences_high", "Absences", "High",
                             f"{absences:g} absences are recorded.",
                             "Talk to the student about attendance",
                             "Agree on practical steps to reduce missed classes and share missed material."))
        elif absences >= ABSENCES_MEDIUM:
            gaps.append(_gap("absences_medium", "Absences", "Medium",
                             f"{absences:g} absences are recorded.",
                             "Make sure missed lessons are caught up",
                             "Share notes or exercises from missed classes and keep an eye on further absences."))

    # 5. Study time (coded 1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h per week)
    if studytime is not None:
        if studytime == 1:
            gaps.append(_gap("study_low", "Study habits", "High",
                             "Weekly study time is under 2 hours (lowest category).",
                             "Build a short daily study routine",
                             "Start with 20-30 minutes on most days (about 2-3 hours a week) in a fixed place and "
                             "time, and increase gradually."))
        elif studytime == 2 and support_level in ("High Support", "Medium Support"):
            gaps.append(_gap("study_medium", "Study habits", "Low",
                             "Weekly study time is 2-5 hours.",
                             "Consider adding one extra focused session per week",
                             "Use the extra session for the topic that cost the most marks."))

    # 6. Past failures
    if failures is not None and failures >= 1:
        sev = "High" if failures >= 2 else "Medium"
        gaps.append(_gap("failures", "Academic history", sev,
                         f"{failures} past class failure(s) recorded.",
                         "Check for missing foundations from earlier years",
                         "Ask the teacher whether earlier topics need revisiting; prerequisite gaps are a common "
                         "reason for repeated difficulty."))

    # 7. Time balance
    if goout is not None and freetime is not None and goout >= 4 and (studytime is not None and studytime <= 2):
        gaps.append(_gap("balance", "Time balance", "Low",
                         "Frequent going out (self-rated 4-5/5) together with low study time.",
                         "Protect fixed study blocks in the weekly schedule",
                         "Plan study sessions first and fit social time around them."))
    return gaps


def generate_recommendations(student: Dict[str, Any], predicted_score_pct: Optional[float] = None,
                             support_level: Optional[str] = None,
                             defaulted_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build prioritised, consistent recommendations.

    ``defaulted_fields`` lists inputs that were not provided; rules only use fields the
    user actually supplied, so a default value can never trigger a recommendation.
    """
    defaulted = set(defaulted_fields or [])
    used = {k: v for k, v in student.items() if k not in defaulted}
    gaps = sorted(diagnose(used, predicted_score_pct, support_level),
                  key=lambda g: SEVERITY_ORDER[g["severity"]])
    actions = gaps[:MAX_PRIORITY_ACTIONS]
    subject = student.get("subject", "Mathematics")

    if not gaps:
        if support_level == "Low Support":
            actions = [_gap("maintain", "Overall", "Info",
                            "No gaps were found in the supplied inputs and the predicted tier is Low Support.",
                            "Maintain current habits and consider extension work",
                            "Optional: ask the teacher for enrichment tasks in this subject.")]
        else:
            actions = [_gap("insufficient", "Overall", "Info",
                            "No specific gap could be identified from the supplied inputs.",
                            "Review progress with the teacher",
                            "Supply more inputs (absences, study time, failures) for more specific guidance.")]

    notes = []
    if defaulted:
        notes.append("Not provided (excluded from the rules, and set to training-data medians/modes for "
                     "the model only): " + ", ".join(sorted(defaulted)) + ".")
    notes.append("These are suggestions for a teacher or tutor to discuss with the student. They are based on the "
                 "inputs and a statistical prediction, not on a confirmed outcome.")
    return {
        "priority_actions": actions,
        "all_gaps": gaps,
        "total_gaps_identified": len(gaps),
        "resources": RESOURCES.get(subject, RESOURCES["Mathematics"]),
        "data_notes": notes,
        "weekly_template": weekly_template(subject, studytime=student.get("studytime")),
    }


def weekly_template(subject: str, studytime: Optional[int] = None) -> List[Dict[str, str]]:
    """A generic starting template (the same shape for every student; edit with the student)."""
    return [
        {"day": "Mon", "activity": f"{subject}: review the week's class notes", "minutes": "30"},
        {"day": "Tue", "activity": f"{subject}: practise questions on one weak topic", "minutes": "30"},
        {"day": "Wed", "activity": "Other subjects / catch-up", "minutes": "30"},
        {"day": "Thu", "activity": f"{subject}: redo mistakes from Tuesday", "minutes": "30"},
        {"day": "Fri", "activity": "Short self-test and note what to ask the teacher", "minutes": "20"},
        {"day": "Sat", "activity": "Optional longer session on the hardest topic", "minutes": "45"},
        {"day": "Sun", "activity": "Rest", "minutes": "0"},
    ]
