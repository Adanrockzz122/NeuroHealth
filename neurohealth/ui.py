from __future__ import annotations

import argparse
import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import parse_qs
from uuid import uuid4

from .config import Settings
from .engine import NeuroHealthEngine
from .models import NeuroHealthRequest, Recommendation, RecommendationFeedback, SymptomReport, UserProfile
from .runtime import build_engine, parse_biometrics, parse_csv, parse_optional_int

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAGRAM_PATH = REPO_ROOT / "assets" / "Diagram.png"
LEGACY_DIAGRAM_PATH = REPO_ROOT / "ProjectDescription" / "Diagram.png"


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _value(form: dict[str, str], key: str, default: str = "") -> str:
    return form.get(key, default)


def _selected(current: str, expected: str) -> str:
    return " selected" if current == expected else ""


def _list_items(values: list[str], empty_label: str) -> str:
    return "".join(f"<li>{_escape(value)}</li>" for value in values) or f"<li>{_escape(empty_label)}</li>"


def _source_chips(values: list[str]) -> str:
    return "".join(f'<span class="chip">{_escape(value)}</span>' for value in values) or '<span class="chip muted">No source cited</span>'


def _urgency_badge_class(urgency: str) -> str:
    mapping = {
        "self_care": "urgency-self-care",
        "routine": "urgency-routine",
        "urgent": "urgency-urgent",
        "emergency": "urgency-emergency",
    }
    return mapping.get(urgency, "urgency-routine")


def _render_recommendation(recommendation: Recommendation, user_query: str) -> str:
    safety_html = _list_items(recommendation.safety_instructions, "No safety instructions provided.")
    clarifications_html = _list_items(recommendation.clarifying_questions, "No follow-up question needed.")
    reasoning_html = _list_items(recommendation.reasoning_notes, "No reasoning notes captured.")
    sources_html = _source_chips(recommendation.sources)
    urgency_label = recommendation.urgency.replace("_", " ").title()
    alert_html = (
        '<div class="alert-banner"><strong>Emergency signal detected:</strong> Seek immediate care now.</div>'
        if recommendation.needs_emergency
        else ""
    )

    return f"""
<section class="result-shell">
  {alert_html}
  <div class="result-top">
    <h2>Clinical Guidance Output</h2>
    <span class="urgency-pill {_urgency_badge_class(recommendation.urgency)}">{_escape(urgency_label)}</span>
  </div>
  <div class="conversation">
    <div class="bubble bubble-user">
      <p class="bubble-title">You</p>
      <p>{_escape(user_query)}</p>
    </div>
    <div class="bubble bubble-assistant">
      <p class="bubble-title">NeuroHealth</p>
      <p>{_escape(recommendation.assistant_message)}</p>
    </div>
  </div>
  <div class="insight-grid">
    <article class="insight-card">
      <h3>Appointment routing</h3>
      <p>{_escape(recommendation.appointment_recommendation)}</p>
    </article>
    <article class="insight-card">
      <h3>Safety guardrails</h3>
      <ul>{safety_html}</ul>
    </article>
    <article class="insight-card">
      <h3>Clarifying questions</h3>
      <ul>{clarifications_html}</ul>
    </article>
    <article class="insight-card">
      <h3>Reasoning trace</h3>
      <ul>{reasoning_html}</ul>
    </article>
  </div>
  <div class="source-wrap">
    <h3>Grounding sources</h3>
    <div class="chip-wrap">{sources_html}</div>
  </div>
</section>
""".strip()


def _render_page(form: dict[str, str], result: Recommendation | None = None, error: str = "") -> str:
    query_value = _value(form, "query")
    result_html = _render_recommendation(result, query_value) if result else """
<section class="result-shell empty-state">
  <h2>Clinical Guidance Output</h2>
  <p>Your recommendation appears here after submitting the intake form.</p>
  <div class="placeholder-grid">
    <div class="placeholder-card"><strong>Urgency assessment</strong><span>self-care / routine / urgent / emergency</span></div>
    <div class="placeholder-card"><strong>Appointment routing</strong><span>primary care, same-day care, or emergency escalation</span></div>
    <div class="placeholder-card"><strong>Safety guidance</strong><span>red-flag instructions and escalation cues</span></div>
    <div class="placeholder-card"><strong>Knowledge grounding</strong><span>evidence snippets from validated sources</span></div>
  </div>
</section>
""".strip()
    error_html = f'<div class="error-banner">{_escape(error)}</div>' if error else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NeuroHealth UI</title>
  <style>
    :root {{
      --bg-1: #f3f7ff;
      --bg-2: #eefaf5;
      --panel: #ffffff;
      --text: #10213a;
      --muted: #5c6b80;
      --primary: #2962ff;
      --primary-soft: #e8eeff;
      --success: #1e8e3e;
      --warning: #d97a00;
      --danger: #c62828;
      --shadow: 0 14px 40px rgba(16, 33, 58, 0.12);
      --radius-lg: 18px;
      --radius-md: 12px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Inter", "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(1200px 600px at 10% -10%, #dfe9ff, transparent 60%),
        radial-gradient(1000px 500px at 90% -10%, #dcf5e8, transparent 60%),
        linear-gradient(180deg, var(--bg-1), var(--bg-2));
    }}
    .page {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 28px 18px 36px;
    }}
    .hero {{
      background: linear-gradient(120deg, #10213a 0%, #193766 40%, #2550a6 100%);
      color: #f5f8ff;
      border-radius: 22px;
      padding: 24px 24px 18px;
      box-shadow: var(--shadow);
      margin-bottom: 16px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 1.95rem;
      letter-spacing: 0.2px;
    }}
    .hero p {{
      margin: 0;
      color: #d8e5ff;
      line-height: 1.5;
      max-width: 920px;
    }}
    .workflow {{
      background: var(--panel);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      padding: 18px;
      margin-bottom: 16px;
    }}
    .workflow h2 {{
      margin: 0 0 12px;
      font-size: 1.15rem;
    }}
    .workflow-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }}
    .workflow-step {{
      background: linear-gradient(180deg, #f8fbff, #f0f5ff);
      border: 1px solid #d8e3ff;
      border-radius: var(--radius-md);
      padding: 10px 10px 12px;
    }}
    .workflow-step strong {{
      display: block;
      font-size: 0.84rem;
      margin-bottom: 4px;
      color: #16345f;
    }}
    .workflow-step span {{
      color: #5a6b84;
      font-size: 0.8rem;
      line-height: 1.35;
    }}
    .diagram-panel {{
      background: var(--panel);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      padding: 18px;
      margin-bottom: 16px;
    }}
    .diagram-panel h2 {{
      margin: 0 0 10px;
      font-size: 1.1rem;
    }}
    .diagram-panel p {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .diagram-image {{
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      background: #f7faff;
      border-radius: var(--radius-md);
      border: 1px solid #dbe7ff;
    }}
    .error-banner {{
      background: #ffe8e8;
      border: 1px solid #ffbbbb;
      color: #7f1d1d;
      border-radius: 10px;
      padding: 10px 12px;
      margin-bottom: 14px;
      font-weight: 600;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 1fr) minmax(380px, 1.15fr);
      gap: 16px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      padding: 18px;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 1.2rem;
    }}
    .panel h3 {{
      margin: 0 0 8px;
      font-size: 0.98rem;
    }}
    .muted {{
      color: var(--muted);
      margin-bottom: 12px;
      font-size: 0.9rem;
    }}
    .section-title {{
      margin: 14px 0 8px;
      color: #21446d;
      font-size: 0.88rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-weight: 700;
    }}
    label {{
      display: block;
      font-size: 0.84rem;
      margin-bottom: 6px;
      color: #21446d;
      font-weight: 600;
    }}
    input, select, textarea {{
      width: 100%;
      padding: 10px 11px;
      border: 1px solid #ccd9f2;
      border-radius: 10px;
      background: #fcfdff;
      color: var(--text);
      margin-bottom: 10px;
      font: inherit;
    }}
    textarea {{ min-height: 108px; resize: vertical; }}
    input:focus, select:focus, textarea:focus {{
      outline: none;
      border-color: #7da1ff;
      box-shadow: 0 0 0 3px rgba(41, 98, 255, 0.15);
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .submit-row {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 6px;
    }}
    button {{
      border: none;
      border-radius: 999px;
      background: linear-gradient(90deg, #2452d9, #1f7fe2);
      color: white;
      font-weight: 700;
      padding: 11px 18px;
      cursor: pointer;
      letter-spacing: 0.02em;
      box-shadow: 0 10px 22px rgba(33, 86, 194, 0.28);
    }}
    button:hover {{ filter: brightness(1.03); }}
    .hint {{
      color: var(--muted);
      font-size: 0.83rem;
      margin: 0;
    }}
    .result-shell h2 {{
      margin: 0;
      font-size: 1.15rem;
    }}
    .result-shell p {{ line-height: 1.48; }}
    .result-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .urgency-pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 5px 11px;
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border: 1px solid transparent;
    }}
    .urgency-self-care {{ background: #edf7ed; color: #226235; border-color: #b9dfc0; }}
    .urgency-routine {{ background: #eef3ff; color: #234fa8; border-color: #c9d7fb; }}
    .urgency-urgent {{ background: #fff3e6; color: #b26500; border-color: #ffd7ad; }}
    .urgency-emergency {{ background: #ffebee; color: #b71c1c; border-color: #f5b4b4; }}
    .alert-banner {{
      background: #fee6e6;
      border: 1px solid #f7b2b2;
      color: #9f1c1c;
      border-radius: 10px;
      padding: 10px 12px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .conversation {{
      display: grid;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .bubble {{
      border-radius: 12px;
      padding: 10px 12px;
      border: 1px solid #d9e4fa;
    }}
    .bubble-title {{
      margin: 0 0 4px;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #5a6b84;
    }}
    .bubble-user {{ background: #f6f9ff; }}
    .bubble-assistant {{
      background: linear-gradient(180deg, #f7fcff, #eef8ff);
      border-color: #cfe3ff;
    }}
    .insight-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .insight-card {{
      border: 1px solid #d8e4fa;
      border-radius: 12px;
      padding: 10px 11px;
      background: #fcfdff;
    }}
    .insight-card h3 {{ margin: 0 0 8px; font-size: 0.92rem; color: #184276; }}
    .insight-card ul {{ margin: 0; padding-left: 18px; color: #2f4161; }}
    .insight-card p {{ margin: 0; color: #2f4161; }}
    .source-wrap h3 {{
      margin: 0 0 8px;
      font-size: 0.92rem;
      color: #184276;
    }}
    .chip-wrap {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
    }}
    .chip {{
      display: inline-block;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 0.78rem;
      background: var(--primary-soft);
      color: #2a4b96;
      border: 1px solid #ccdaff;
      font-weight: 600;
    }}
    .chip.muted {{
      background: #f2f4f7;
      border-color: #e0e5ed;
      color: #66758f;
    }}
    .empty-state p {{ color: var(--muted); margin: 8px 0 14px; }}
    .placeholder-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .placeholder-card {{
      border: 1px dashed #cad8f5;
      border-radius: 10px;
      padding: 10px;
      background: #f8fbff;
    }}
    .placeholder-card strong {{
      display: block;
      margin-bottom: 4px;
      color: #25456f;
      font-size: 0.85rem;
    }}
    .placeholder-card span {{
      color: #5a6b84;
      font-size: 0.82rem;
      line-height: 1.35;
    }}
    @media (max-width: 1060px) {{
      .workflow-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .layout {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 720px) {{
      .hero h1 {{ font-size: 1.6rem; }}
      .workflow-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .form-grid, .insight-grid, .placeholder-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <h1>NeuroHealth Interactive UI</h1>
      <p>
        A clinically cautious AI health assistant prototype for symptom interpretation, urgency triage,
        appointment routing, and personalized guidance grounded in validated medical knowledge.
      </p>
    </header>

    <section class="workflow">
      <h2>NeuroHealth pipeline (aligned with project architecture)</h2>
      <div class="workflow-grid">
        <div class="workflow-step"><strong>1. Intent recognition</strong><span>Capture user goals, symptom language, and context.</span></div>
        <div class="workflow-step"><strong>2. Symptom extraction</strong><span>Structure symptoms, duration, pain, and biometrics.</span></div>
        <div class="workflow-step"><strong>3. RAG grounding</strong><span>Retrieve validated health snippets before generation.</span></div>
        <div class="workflow-step"><strong>4. Urgency assessment</strong><span>Apply safety guardrails and emergency escalation logic.</span></div>
        <div class="workflow-step"><strong>5. Recommendation output</strong><span>Deliver appointment routing, guidance, and follow-ups.</span></div>
      </div>
    </section>

    <section class="diagram-panel">
      <h2>Reference system diagram</h2>
      <p>The UI is designed around the same component flow shown in your project diagram.</p>
      <img src="/diagram.png" alt="NeuroHealth architecture diagram" class="diagram-image" />
    </section>

    {error_html}

    <main class="layout">
      <section class="panel">
        <h2>Patient Intake + Context</h2>
        <p class="muted">Provide natural-language symptoms and optional context for personalization.</p>
        <form method="post" action="/recommend">
          <label for="query">Health question or symptom narrative *</label>
          <textarea id="query" name="query" required placeholder="Example: I have mild fever and cough for 2 days and want to know if I should book an appointment.">{_escape(_value(form, "query"))}</textarea>

          <p class="section-title">Profile</p>
          <div class="form-grid">
            <div>
              <label for="age">Age</label>
              <input id="age" name="age" type="number" min="1" value="{_escape(_value(form, "age"))}" />
            </div>
            <div>
              <label for="health_literacy">Health literacy</label>
              <select id="health_literacy" name="health_literacy">
                <option value="basic"{_selected(_value(form, "health_literacy", "intermediate"), "basic")}>basic</option>
                <option value="intermediate"{_selected(_value(form, "health_literacy", "intermediate"), "intermediate")}>intermediate</option>
                <option value="advanced"{_selected(_value(form, "health_literacy", "intermediate"), "advanced")}>advanced</option>
              </select>
            </div>
          </div>

          <label for="preferences">Preferences (comma-separated)</label>
          <input id="preferences" name="preferences" placeholder="vegetarian, low-impact exercise" value="{_escape(_value(form, "preferences"))}" />
          <label for="medical_constraints">Medical constraints (comma-separated)</label>
          <input id="medical_constraints" name="medical_constraints" placeholder="knee pain, hypertension" value="{_escape(_value(form, "medical_constraints"))}" />
          <label for="chronic_conditions">Chronic conditions (comma-separated)</label>
          <input id="chronic_conditions" name="chronic_conditions" placeholder="asthma, diabetes" value="{_escape(_value(form, "chronic_conditions"))}" />

          <p class="section-title">Symptoms + vital context</p>
          <div class="form-grid">
            <div>
              <label for="symptoms">Symptoms (comma-separated)</label>
              <input id="symptoms" name="symptoms" placeholder="fever, dry cough, fatigue" value="{_escape(_value(form, "symptoms"))}" />
            </div>
            <div>
              <label for="duration_hours">Duration (hours)</label>
              <input id="duration_hours" name="duration_hours" type="number" min="0" value="{_escape(_value(form, "duration_hours"))}" />
            </div>
            <div>
              <label for="pain_level">Pain level (0-10)</label>
              <input id="pain_level" name="pain_level" type="number" min="0" max="10" value="{_escape(_value(form, "pain_level"))}" />
            </div>
            <div>
              <label for="biometrics">Biometrics key=value, comma-separated</label>
              <input id="biometrics" name="biometrics" placeholder="temperature_c=38.1,oxygen_saturation=97" value="{_escape(_value(form, "biometrics"))}" />
            </div>
          </div>

          <p class="section-title">Feedback loop (optional)</p>
          <div class="form-grid">
            <div>
              <label for="feedback_rating">Feedback rating (1-5)</label>
              <input id="feedback_rating" name="feedback_rating" type="number" min="1" max="5" value="{_escape(_value(form, "feedback_rating"))}" />
            </div>
            <div>
              <label for="feedback_comment">Feedback comment</label>
              <input id="feedback_comment" name="feedback_comment" placeholder="Was the recommendation clear?" value="{_escape(_value(form, "feedback_comment"))}" />
            </div>
          </div>

          <div class="submit-row">
            <button type="submit">Generate clinical guidance</button>
          </div>
          <p class="hint">Prototype only: this does not replace professional medical diagnosis or emergency services.</p>
        </form>
      </section>

      <section class="panel">
        {result_html}
      </section>
    </main>
  </div>
</body>
</html>
"""


def _parse_form(body: bytes) -> dict[str, str]:
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[0] if values else "" for key, values in parsed.items()}


def _build_request(form: dict[str, str]) -> NeuroHealthRequest:
    query = _value(form, "query").strip()
    if not query:
        raise ValueError("Health question is required.")

    health_literacy_raw = _value(form, "health_literacy", "intermediate")
    if health_literacy_raw not in {"basic", "intermediate", "advanced"}:
        raise ValueError("Health literacy must be one of: basic, intermediate, advanced.")
    health_literacy: Literal["basic", "intermediate", "advanced"] = health_literacy_raw

    biometrics_raw = _value(form, "biometrics")
    biometrics_items = [item.strip() for item in biometrics_raw.split(",") if item.strip()]

    profile = UserProfile(
        age=parse_optional_int(_value(form, "age")),
        preferences=parse_csv(_value(form, "preferences")),
        medical_constraints=parse_csv(_value(form, "medical_constraints")),
        chronic_conditions=parse_csv(_value(form, "chronic_conditions")),
        health_literacy=health_literacy,
    )
    symptoms = SymptomReport(
        symptoms=parse_csv(_value(form, "symptoms")),
        biometrics=parse_biometrics(biometrics_items),
        duration_hours=parse_optional_int(_value(form, "duration_hours")),
        pain_level=parse_optional_int(_value(form, "pain_level")),
    )
    return NeuroHealthRequest(user_input=query, user_profile=profile, symptom_report=symptoms, history=[])


def _record_feedback_if_present(engine: NeuroHealthEngine, form: dict[str, str]) -> None:
    rating_text = _value(form, "feedback_rating").strip()
    if not rating_text:
        return
    feedback = RecommendationFeedback(
        conversation_id=str(uuid4()),
        rating=int(rating_text),
        comment=_value(form, "feedback_comment").strip(),
    )
    engine.record_feedback(feedback)


def _create_handler(engine_factory: Callable[[], NeuroHealthEngine]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        _engine: NeuroHealthEngine | None = None

        def _get_engine(self) -> NeuroHealthEngine:
            if type(self)._engine is None:
                type(self)._engine = engine_factory()
            return type(self)._engine

        def do_GET(self) -> None:
            if self.path == "/diagram.png":
                diagram_path = DIAGRAM_PATH if DIAGRAM_PATH.exists() else LEGACY_DIAGRAM_PATH
                if not diagram_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND, "Diagram image not found")
                    return
                self._send_binary(diagram_path.read_bytes(), "image/png")
                return
            if self.path not in {"/", "/index.html"}:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self._send_html(_render_page({}))

        def do_POST(self) -> None:
            if self.path != "/recommend":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            form = _parse_form(body)
            try:
                request_payload = _build_request(form)
                engine = self._get_engine()
                recommendation = engine.generate(request_payload)
                _record_feedback_if_present(engine, form)
                self._send_html(_render_page(form, recommendation))
            except Exception as exc:
                self._send_html(_render_page(form, error=str(exc)), status=HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_html(self, payload: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_binary(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroHealth interactive web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    def _engine_factory() -> NeuroHealthEngine:
        settings = Settings.from_env()
        return build_engine(settings)

    handler_cls = _create_handler(_engine_factory)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(
        json.dumps(
            {
                "message": "NeuroHealth UI server running",
                "url": f"http://{args.host}:{args.port}",
            }
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
