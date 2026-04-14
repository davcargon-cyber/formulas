"""
IOL Advisor — Backend
Flask server that orchestrates PDF extraction, ESCRS automation, and AI recommendation.
"""

import os
import re
import json
import time
import base64
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import anthropic
from playwright.sync_api import sync_playwright

# ─── Config ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("iol")

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ESCRS_URL = "https://iolcalculator.escrs.org/"
FORMULA_NAMES = [
    "Barrett Universal II", "Cooke K6", "EVO",
    "Hill-RBF", "Hoffer QST", "Kane", "PEARL-DGS",
]


def get_client():
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "has_key": bool(ANTHROPIC_KEY)})


@app.route("/extract", methods=["POST"])
def extract_biometry():
    """Extract biometry data from uploaded PDF/image using Claude."""
    if not ANTHROPIC_KEY:
        return jsonify({"error": "API key no configurada en el servidor"}), 500

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No se recibió ningún archivo"}), 400

    try:
        raw = file.read()
        b64 = base64.standard_b64encode(raw).decode("utf-8")
        fname = file.filename.lower()

        if fname.endswith(".pdf"):
            block = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}}
        elif any(fname.endswith(e) for e in (".png", ".jpg", ".jpeg", ".webp")):
            mt_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
            ext = "." + fname.rsplit(".", 1)[-1]
            block = {"type": "image", "source": {"type": "base64", "media_type": mt_map.get(ext, "image/png"), "data": b64}}
        else:
            return jsonify({"error": f"Formato no soportado: {fname}. Usa PDF, PNG, JPG o WEBP."}), 400

        client = get_client()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": [
                block,
                {"type": "text", "text": """Extract biometry from this ophthalmology report. Return ONLY valid JSON:
{"axl":number,"acd":number,"k1":number,"k2":number,"lt":number|null,"wtw":number|null,"cct":number|null,"a_const":number|null,"patient":""|null,"eye":"OD"|"OS"|null,"iol_model":""|null}
k1 = flat K (lower), k2 = steep K. Standard units (mm, D, μm). No markdown."""}
            ]}],
        )

        text = msg.content[0].text.strip()
        text = re.sub(r"```json\s*|```\s*", "", text).strip()
        parsed = json.loads(text)

        # Clean nulls
        result = {k: v for k, v in parsed.items() if v is not None}
        log.info(f"Extracted: AL={result.get('axl')} K1={result.get('k1')} K2={result.get('k2')}")
        return jsonify(result)

    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        return jsonify({"error": "No se pudieron extraer los datos del documento. Introduce los valores manualmente."}), 422
    except Exception as e:
        log.error(f"Extraction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/calculate", methods=["POST"])
def calculate_escrs():
    """Automate ESCRS calculator with Playwright and return real results."""
    bio = request.json
    if not bio:
        return jsonify({"error": "No se recibieron datos biométricos"}), 400

    required = ["axl", "k1", "k2"]
    missing = [f for f in required if not bio.get(f)]
    if missing:
        return jsonify({"error": f"Faltan campos obligatorios: {', '.join(missing)}"}), 400

    log.info(f"Starting ESCRS calculation: AL={bio['axl']} K1={bio['k1']} K2={bio['k2']}")

    try:
        results = run_escrs_automation(bio)
        return jsonify(results)
    except Exception as e:
        log.error(f"ESCRS error: {e}")
        return jsonify({"error": f"Error en la automatización ESCRS: {str(e)}"}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    """Generate AI clinical recommendation based on biometry + ESCRS results."""
    if not ANTHROPIC_KEY:
        return jsonify({"error": "API key no configurada"}), 500

    data = request.json
    bio = data.get("bio", {})
    results = data.get("results", {})

    try:
        rec = generate_recommendation(bio, results)
        return jsonify({"recommendation": rec})
    except Exception as e:
        log.error(f"Recommendation error: {e}")
        return jsonify({"error": str(e)}), 500


# ─── ESCRS Automation ────────────────────────────────────────────────────────

def run_escrs_automation(bio):
    """
    Open ESCRS calculator, fill in data, calculate, and extract results.
    Returns dict with formula results and metadata.
    """
    results = {"formulas": {}, "status": "ok", "errors": [], "timestamp": datetime.now().isoformat()}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        try:
            # 1. Navigate
            log.info("Opening ESCRS...")
            page.goto(ESCRS_URL, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(3000)

            # 2. Accept terms
            log.info("Accepting terms...")
            for sel in ["text=I Agree", "button:has-text('Agree')", "text=Agree"]:
                try:
                    el = page.wait_for_selector(sel, timeout=5000)
                    if el:
                        el.click()
                        page.wait_for_timeout(2000)
                        break
                except:
                    continue

            # 3. Fill biometry
            log.info("Filling biometry...")
            filled = fill_form(page, bio)
            results["fields_filled"] = filled
            page.wait_for_timeout(1000)

            # 4. IOL / A-constant
            if bio.get("iol_model"):
                try_select_iol(page, bio["iol_model"])
            set_a_constant(page, bio.get("a_const", 118.7))
            page.wait_for_timeout(500)

            # 5. Click Calculate
            log.info("Clicking Calculate...")
            calc_clicked = False
            for sel in ["button:has-text('Calculate')", "text=Calculate", "button:has-text('Calcul')"]:
                try:
                    btn = page.wait_for_selector(sel, timeout=3000)
                    if btn:
                        btn.click()
                        calc_clicked = True
                        break
                except:
                    continue

            if not calc_clicked:
                results["errors"].append("No se encontró el botón Calculate")
                results["status"] = "partial"

            # 6. Wait for results
            log.info("Waiting for results...")
            start = time.time()
            max_wait = 120
            best_count = 0

            while time.time() - start < max_wait:
                body_text = page.inner_text("body").lower()
                count = sum(1 for f in FORMULA_NAMES if f.lower() in body_text)
                if count > best_count:
                    best_count = count
                    log.info(f"  {count}/7 formulas received")
                if count >= 5:
                    page.wait_for_timeout(3000)
                    break
                page.wait_for_timeout(2000)

            # 7. Extract
            log.info("Extracting results...")
            results["formulas"] = extract_results(page)
            results["formulas_found"] = len(results["formulas"])

            # Screenshot
            try:
                ss_path = "/tmp/escrs_results.png"
                page.screenshot(path=ss_path, full_page=True)
            except:
                pass

        except Exception as e:
            log.error(f"Playwright error: {e}")
            results["status"] = "error"
            results["errors"].append(str(e))
            try:
                page.screenshot(path="/tmp/escrs_error.png", full_page=True)
            except:
                pass
        finally:
            browser.close()

    return results


def fill_form(page, bio):
    """Fill ESCRS form fields by matching label text."""
    field_map = [
        (r"axial|length|\bal\b", bio.get("axl")),
        (r"flat.*k|k1|kf\b", bio.get("k1")),
        (r"steep.*k|k2|ks\b", bio.get("k2")),
        (r"acd|anterior.*chamber|chamber.*depth", bio.get("acd")),
        (r"lens.*thick|\blt\b|thickness", bio.get("lt")),
        (r"corneal.*thick|cct|pachym", bio.get("cct")),
        (r"white.*white|wtw|corneal.*diam", bio.get("wtw")),
    ]

    labels = page.query_selector_all("label")
    filled = 0

    for pattern, value in field_map:
        if not value:
            continue
        regex = re.compile(pattern, re.IGNORECASE)

        for label in labels:
            text = label.inner_text().strip()
            if regex.search(text):
                inp = find_input_for_label(page, label)
                if inp:
                    try:
                        inp.click()
                        inp.fill(str(value))
                        inp.dispatch_event("input")
                        inp.dispatch_event("change")
                        inp.evaluate("el => el.dispatchEvent(new Event('blur',{bubbles:true}))")
                        filled += 1
                        log.info(f"  Filled: {text} = {value}")
                        page.wait_for_timeout(80)
                        break
                    except Exception as e:
                        log.warning(f"  Failed to fill {text}: {e}")

    return filled


def find_input_for_label(page, label):
    """Find the input element associated with a label."""
    for_id = label.get_attribute("for")
    if for_id:
        inp = page.query_selector(f"#{for_id}")
        if inp:
            return inp

    inp = label.query_selector("input")
    if inp:
        return inp

    try:
        handle = label.evaluate_handle(
            "el => el.parentElement?.querySelector?.('input') || el.nextElementSibling?.querySelector?.('input')"
        )
        return handle.as_element()
    except:
        return None


def try_select_iol(page, model):
    """Try to select an IOL model."""
    for sel in ["input[placeholder*='IOL' i]", "input[placeholder*='search' i]", "input[placeholder*='lens' i]"]:
        try:
            el = page.query_selector(sel)
            if el:
                el.fill(model)
                el.dispatch_event("input")
                page.wait_for_timeout(1000)
                opt = page.query_selector(f"text={model}")
                if opt:
                    opt.click()
                return
        except:
            continue


def set_a_constant(page, value):
    """Set A-constant field."""
    labels = page.query_selector_all("label")
    for label in labels:
        text = label.inner_text().strip().lower()
        if re.search(r"a.const|constant|optimized", text):
            inp = find_input_for_label(page, label)
            if inp:
                inp.fill(str(value))
                inp.dispatch_event("input")
                inp.dispatch_event("change")
                return


def extract_results(page):
    """Extract formula results from the ESCRS page."""
    formulas = {}

    # Strategy 1: Tables
    for table in page.query_selector_all("table"):
        for row in table.query_selector_all("tr"):
            cells = row.query_selector_all("td, th")
            texts = [c.inner_text().strip() for c in cells]
            row_text = " ".join(texts).lower()

            for name in FORMULA_NAMES:
                if name.lower() in row_text and name not in formulas:
                    nums = []
                    for t in texts:
                        try:
                            nums.append(float(t.replace(",", ".")))
                        except:
                            pass
                    if nums:
                        formulas[name] = {"power": nums[0], "rx": nums[1] if len(nums) > 1 else 0.0}

    # Strategy 2: Regex fallback
    if len(formulas) < 3:
        body = page.inner_text("body")
        for name in FORMULA_NAMES:
            if name not in formulas:
                pat = re.escape(name) + r"[\s:=]*([-+]?\d+\.?\d*)"
                m = re.search(pat, body, re.IGNORECASE)
                if m:
                    formulas[name] = {"power": float(m.group(1)), "rx": 0.0}

    return formulas


# ─── AI Recommendation ──────────────────────────────────────────────────────

def generate_recommendation(bio, results):
    """Generate clinical recommendation using Claude."""
    client = get_client()

    al = float(bio.get("axl", 0))
    k1 = float(bio.get("k1", 0))
    k2 = float(bio.get("k2", 0))
    k_avg = (k1 + k2) / 2
    astig = abs(k2 - k1)
    eye_type = "corto" if al < 22 else "largo" if al > 25.5 else "normal"

    formulas = results.get("formulas", {})
    formula_lines = "\n".join(
        f"  • {name}: {d['power']:.1f} D (Rx: {d.get('rx', 0):+.2f})"
        for name, d in formulas.items()
    ) or "  (Sin resultados)"

    powers = [d["power"] for d in formulas.values() if d.get("power")]
    rng = (max(powers) - min(powers)) if len(powers) >= 2 else 0

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": f"""Eres oftalmólogo experto en catarata y biometría. Analiza estos RESULTADOS REALES de la calculadora ESCRS.

BIOMETRÍA: AL={al}mm ({eye_type}) · ACD={bio.get('acd','?')}mm · K1={k1}D K2={k2}D K̄={k_avg:.2f}D · ΔK={astig:.2f}D
LT={bio.get('lt','N/D')} · WTW={bio.get('wtw','N/D')} · CCT={bio.get('cct','N/D')} · A={bio.get('a_const',118.7)} · LIO={bio.get('iol_model','?')}

RESULTADOS REALES (7 FÓRMULAS ESCRS):
{formula_lines}
Rango: {rng:.2f}D · Concordancia: {'Alta' if rng <= 0.5 else 'Moderada' if rng <= 1.0 else 'Baja'}

Basándote en Melles 2019, Darcy 2020, Kane 2020, Aristodemou 2011, Wang-Koch 2018, proporciona:

1. FÓRMULA RECOMENDADA para este ojo y por qué
2. POTENCIA a implantar (emetropía vs miopía residual)
3. CONCORDANCIA — interpretar dispersión entre fórmulas
4. PRECAUCIONES específicas
5. CONFIANZA (Alta/Media/Baja)

Español, conciso, clínico."""}],
    )
    return msg.content[0].text


# ─── Run ─────────────────────────────────────────────────────────────────────
"""
Kane Formula Test — Prueba de scraping con Playwright
Añade este endpoint a tu app.py en Render para testear.
Accede a: https://tu-app.onrender.com/test-kane
"""

# ─── Pega esto al final de tu app.py, ANTES de la línea "if __name__" ────

@app.route("/test-kane")
def test_kane():
    """
    Test: intenta automatizar la calculadora Kane con datos de ejemplo.
    Devuelve un log detallado de qué pasa en cada paso.
    """
    import time
    from playwright.sync_api import sync_playwright

    # Datos de prueba
    test_data = {
        "a_const": 118.7,
        "sex": "Male",
        "axl": 23.50,
        "k1": 43.25,
        "k2": 44.00,
        "acd": 3.12,
        "lt": 4.56,
        "cct": 545,
    }

    log_entries = []
    def log(msg):
        log_entries.append(f"{time.strftime('%H:%M:%S')} — {msg}")
        app.logger.info(msg)

    results = {"status": "started", "log": log_entries, "data": None}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"]
        )
        # Simular navegador real
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            java_script_enabled=True,
        )
        page = context.new_page()

        try:
            # ── 1. Abrir la web ──
            log("1. Abriendo iolformula.com...")
            page.goto("https://www.iolformula.com/", wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000)
            log(f"   URL actual: {page.url}")
            log(f"   Título: {page.title()}")

            # ── 2. Buscar el botón "I Agree" ──
            log("2. Buscando botón 'I Agree'...")
            body_text = page.inner_text("body")[:500]
            log(f"   Texto visible (primeros 500 chars): {body_text[:200]}...")

            agree_btn = None
            for sel in [
                "text=I Agree",
                "button:has-text('I Agree')",
                "a:has-text('I Agree')",
                "input[value='I Agree']",
                "#agree",
                ".agree",
            ]:
                try:
                    agree_btn = page.wait_for_selector(sel, timeout=3000)
                    if agree_btn:
                        log(f"   ✓ Encontrado con selector: {sel}")
                        break
                except:
                    continue

            if not agree_btn:
                log("   ✗ No se encontró botón I Agree. Buscando todos los botones/links...")
                buttons = page.query_selector_all("button, a, input[type='submit']")
                for i, btn in enumerate(buttons):
                    txt = btn.inner_text().strip()[:50] if btn.inner_text() else ""
                    href = btn.get_attribute("href") or ""
                    log(f"   [{i}] text='{txt}' href='{href}'")

            # ── 3. Hacer clic en I Agree ──
            if agree_btn:
                log("3. Haciendo clic en 'I Agree'...")
                agree_btn.click()
                page.wait_for_timeout(3000)
                log(f"   URL después del clic: {page.url}")
            else:
                log("3. SKIP — No hay botón, intentando navegar directo al calculator...")
                page.goto("https://www.iolformula.com/calculator/", wait_until="networkidle", timeout=15000)
                page.wait_for_timeout(2000)
                log(f"   URL: {page.url}")

            # ── 4. Detectar reCAPTCHA ──
            log("4. Buscando reCAPTCHA...")
            page_html = page.content()
            has_recaptcha = "recaptcha" in page_html.lower()
            has_grecaptcha = "grecaptcha" in page_html.lower()
            has_captcha_frame = bool(page.query_selector("iframe[src*='recaptcha']"))
            has_captcha_div = bool(page.query_selector(".g-recaptcha"))
            log(f"   recaptcha en HTML: {has_recaptcha}")
            log(f"   grecaptcha en HTML: {has_grecaptcha}")
            log(f"   iframe recaptcha: {has_captcha_frame}")
            log(f"   div .g-recaptcha: {has_captcha_div}")

            if has_captcha_frame:
                log("   ⚠ CAPTCHA IFRAME detectado — probablemente v2 (checkbox)")
            elif has_recaptcha and not has_captcha_frame:
                log("   ℹ reCAPTCHA detectado pero sin iframe — puede ser v3 (invisible)")

            # ── 5. Listar todos los inputs del formulario ──
            log("5. Escaneando formulario...")
            all_inputs = page.query_selector_all("input, select, textarea")
            log(f"   Total inputs encontrados: {len(all_inputs)}")
            for i, inp in enumerate(all_inputs):
                tag = inp.evaluate("el => el.tagName")
                inp_type = inp.get_attribute("type") or ""
                inp_name = inp.get_attribute("name") or ""
                inp_id = inp.get_attribute("id") or ""
                placeholder = inp.get_attribute("placeholder") or ""
                inp_class = (inp.get_attribute("class") or "")[:40]
                visible = inp.is_visible()
                log(f"   [{i}] <{tag}> type='{inp_type}' name='{inp_name}' id='{inp_id}' ph='{placeholder}' class='{inp_class}' visible={visible}")

            # ── 6. Listar labels ──
            log("6. Escaneando labels...")
            labels = page.query_selector_all("label")
            for i, lbl in enumerate(labels):
                txt = lbl.inner_text().strip()[:50]
                for_attr = lbl.get_attribute("for") or ""
                log(f"   [{i}] for='{for_attr}' text='{txt}'")

            # ── 7. Intentar rellenar campos ──
            log("7. Intentando rellenar datos de prueba...")
            field_map = {
                "a_const": str(test_data["a_const"]),
                "axl": str(test_data["axl"]),
                "k1": str(test_data["k1"]),
                "k2": str(test_data["k2"]),
                "acd": str(test_data["acd"]),
                "lt": str(test_data["lt"]),
                "cct": str(test_data["cct"]),
            }
            filled = 0
            for inp in all_inputs:
                inp_name = (inp.get_attribute("name") or "").lower()
                inp_id = (inp.get_attribute("id") or "").lower()
                placeholder = (inp.get_attribute("placeholder") or "").lower()
                combined = f"{inp_name} {inp_id} {placeholder}"

                for key, value in field_map.items():
                    if key.lower() in combined or key.replace("_","") in combined:
                        try:
                            if inp.is_visible():
                                inp.click()
                                inp.fill(value)
                                inp.dispatch_event("input")
                                inp.dispatch_event("change")
                                filled += 1
                                log(f"   ✓ Rellenado '{key}' = {value} (via name/id/placeholder match)")
                                break
                        except Exception as e:
                            log(f"   ✗ Error rellenando '{key}': {e}")

            # Try by label matching too
            label_patterns = {
                r"a.const|constant": "a_const",
                r"axial|length|\bal\b": "axl",
                r"flat|k1": "k1",
                r"steep|k2": "k2",
                r"anterior|acd|chamber": "acd",
                r"lens.*thick|\blt\b": "lt",
                r"corneal.*thick|cct": "cct",
            }
            import re as regex
            for lbl in labels:
                txt = lbl.inner_text().strip().lower()
                for pattern, key in label_patterns.items():
                    if regex.search(pattern, txt) and key in field_map:
                        for_id = lbl.get_attribute("for")
                        inp = page.query_selector(f"#{for_id}") if for_id else lbl.query_selector("input, select")
                        if inp and inp.is_visible():
                            try:
                                inp.click()
                                inp.fill(field_map[key])
                                inp.dispatch_event("input")
                                inp.dispatch_event("change")
                                filled += 1
                                log(f"   ✓ Rellenado '{key}' = {field_map[key]} (via label '{txt}')")
                            except Exception as e:
                                log(f"   ✗ Error via label '{txt}': {e}")

            log(f"   Total campos rellenados: {filled}")

            # ── 8. Buscar botón Calculate ──
            log("8. Buscando botón Calculate/Submit...")
            for sel in [
                "button:has-text('Calculate')",
                "input[type='submit']",
                "button[type='submit']",
                "text=Calculate",
                "button:has-text('Submit')",
            ]:
                try:
                    calc_btn = page.wait_for_selector(sel, timeout=2000)
                    if calc_btn:
                        log(f"   ✓ Botón encontrado: {sel}")
                        log("   Haciendo clic...")
                        calc_btn.click()
                        page.wait_for_timeout(5000)
                        log(f"   URL después de Calculate: {page.url}")
                        break
                except:
                    continue

            # ── 9. Extraer resultados ──
            log("9. Buscando resultados...")
            page.wait_for_timeout(3000)
            result_text = page.inner_text("body")

            # Look for IOL power numbers
            import re as regex
            power_matches = regex.findall(r'(\d{1,2}\.\d{1,2})\s*(?:D|diopt)', result_text)
            log(f"   Posibles potencias encontradas: {power_matches[:10]}")

            # Look for tables
            tables = page.query_selector_all("table")
            log(f"   Tablas encontradas: {len(tables)}")
            for i, table in enumerate(tables):
                rows = table.query_selector_all("tr")
                log(f"   Tabla [{i}]: {len(rows)} filas")
                for j, row in enumerate(rows[:5]):
                    cells = [c.inner_text().strip()[:20] for c in row.query_selector_all("td, th")]
                    log(f"     Fila {j}: {cells}")

            # Save page text for analysis
            results["page_text"] = result_text[:3000]
            results["status"] = "completed"

            # Screenshot
            try:
                page.screenshot(path="/tmp/kane_test.png", full_page=True)
                log("   Screenshot guardado en /tmp/kane_test.png")
            except:
                pass

        except Exception as e:
            log(f"ERROR: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            try:
                page.screenshot(path="/tmp/kane_error.png", full_page=True)
            except:
                pass
        finally:
            browser.close()

    results["log"] = log_entries
    return jsonify(results)


# ─── También un endpoint para ver el screenshot ──────────────────────────

@app.route("/screenshot/<name>")
def screenshot(name):
    """Serve a screenshot file."""
    from flask import send_file
    path = f"/tmp/{name}.png"
    try:
        return send_file(path, mimetype="image/png")
    except:
        return "No screenshot available", 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
