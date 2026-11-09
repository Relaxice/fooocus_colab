#!/usr/bin/env python3
"""
Fooocus Colab — Image Generation Service
  Run:  uvicorn app:app --host 0.0.0.0 --port 7865 --reload
  API:  POST /api/generate  { prompt, negative_prompt, steps, width, height, seed }
        GET  /api/status
        GET  /outputs/<filename>
"""
import asyncio, base64, io, json, os, random, time
from pathlib import Path

ROOT        = Path(__file__).parent
FOOOCUS_DIR = ROOT.parent / "Fooocus"
MODELS_DIR  = FOOOCUS_DIR / "models" / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Capability detection ──────────────────────────────────
HAS_TORCH = HAS_DIFFUSERS = False
_pipe = None

def _probe_caps():
    global HAS_TORCH, HAS_DIFFUSERS
    try:
        import torch; HAS_TORCH = True  # noqa: E401
        from diffusers import DiffusionPipeline; HAS_DIFFUSERS = True  # noqa: F401,E401
    except ImportError:
        pass

_probe_caps()

def _get_models():
    return sorted(
        p.name for p in MODELS_DIR.iterdir()
        if p.suffix.lower() in {".safetensors", ".ckpt", ".pt"}
    ) if MODELS_DIR.exists() else []

def _load_pipe(model_name: str):
    global _pipe
    if not HAS_DIFFUSERS:
        return False
    try:
        from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
        import torch
        path = MODELS_DIR / model_name
        if not path.exists():
            return False
        is_xl = "xl" in model_name.lower() or "sdxl" in model_name.lower()
        cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
        _pipe = cls.from_single_file(str(path), torch_dtype=torch.float32)
        _pipe = _pipe.to("cpu")
        return True
    except Exception as e:
        print(f"[app] load_pipe failed: {e}")
        return False

# ── Placeholder generator (Pillow, always available) ─────
def _placeholder(prompt: str, width: int, height: int, seed: int) -> bytes:
    from PIL import Image, ImageDraw
    import hashlib
    rng = random.Random(seed)
    h = int(hashlib.md5(prompt.encode()).hexdigest()[:6], 16)
    base_r = (h >> 16) & 0xFF
    base_g = (h >> 8)  & 0xFF
    base_b =  h        & 0xFF
    img  = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / height
        r = int(base_r * (0.3 + 0.7 * (1 - t)))
        g = int(base_g * (0.3 + 0.7 * (1 - t)))
        b = int(base_b * (0.4 + 0.6 * t))
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    # Text overlay
    words  = (prompt or "empty prompt").split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) > 32:
            lines.append(" ".join(cur[:-1])); cur = [w]
    if cur: lines.append(" ".join(cur))
    text = "\n".join(lines[:5])
    bx = draw.textbbox((0,0), text)
    tw, th = bx[2]-bx[0], bx[3]-bx[1]
    x, y = (width-tw)//2, (height-th)//2
    draw.text((x+2, y+2), text, fill=(0,0,0))
    draw.text((x, y),     text, fill=(255,255,255))
    draw.text((8, 8), "[ No model — placeholder ]", fill=(255,220,80))
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()

def _real_generate(prompt, negative, steps, width, height, seed):
    import torch
    gen = torch.Generator().manual_seed(seed)
    result = _pipe(
        prompt, negative_prompt=negative,
        num_inference_steps=steps,
        width=width, height=height,
        generator=gen,
    )
    img = result.images[0]
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()

# ── ASGI helpers ──────────────────────────────────────────
async def _body(receive):
    b = b""
    while True:
        ev = await receive()
        b += ev.get("body", b"")
        if not ev.get("more_body", False): break
    return b

async def _send(send, status: int, ctype: str, body):
    if isinstance(body, str): body = body.encode()
    await send({"type":"http.response.start","status":status,"headers":[
        [b"content-type", ctype.encode()],
        [b"content-length", str(len(body)).encode()],
        [b"access-control-allow-origin", b"*"],
    ]})
    await send({"type":"http.response.body","body":body})

async def _json(send, data, status=200):
    await _send(send, status, "application/json", json.dumps(data))

# ── Route handlers ────────────────────────────────────────
async def route_status(scope, receive, send):
    models = _get_models()
    await _json(send, {
        "torch":     HAS_TORCH,
        "diffusers": HAS_DIFFUSERS,
        "models":    models,
        "loaded":    _pipe is not None,
        "device":    "cpu",
        "placeholder_mode": _pipe is None,
    })

async def route_generate(scope, receive, send):
    raw = await _body(receive)
    try:   req = json.loads(raw)
    except Exception: await _json(send, {"error":"invalid JSON"}, 400); return

    prompt   = str(req.get("prompt","")).strip()
    negative = str(req.get("negative_prompt","")).strip()
    steps    = max(1, min(int(req.get("steps", 20)), 100))
    width    = max(64, min(int(req.get("width",  512)), 2048))
    height   = max(64, min(int(req.get("height", 512)), 2048))
    # Round to multiples of 64
    width    = (width  // 64) * 64
    height   = (height // 64) * 64
    seed     = int(req.get("seed") or random.randint(0, 2**31-1))

    if not prompt: await _json(send, {"error":"prompt required"}, 400); return

    t0 = time.time()
    placeholder = True
    try:
        if _pipe is not None:
            png = _real_generate(prompt, negative, steps, width, height, seed)
            placeholder = False
        else:
            png = _placeholder(prompt, width, height, seed)
    except Exception as e:
        print(f"[app] generate error: {e}")
        png = _placeholder(prompt, width, height, seed)

    duration = round(time.time() - t0, 2)
    fname = f"{int(time.time())}_{seed}.png"
    (OUTPUTS_DIR / fname).write_bytes(png)

    await _json(send, {
        "image":       base64.b64encode(png).decode(),
        "seed":        seed,
        "duration":    duration,
        "placeholder": placeholder,
        "filename":    fname,
    })

async def route_load_model(scope, receive, send):
    raw = await _body(receive)
    try:   req = json.loads(raw)
    except Exception: await _json(send, {"error":"invalid JSON"}, 400); return
    model = req.get("model","")
    if not model: await _json(send, {"error":"model required"}, 400); return
    ok = _load_pipe(model)
    await _json(send, {"ok": ok, "model": model if ok else None})

async def route_outputs(scope, receive, send):
    fname = scope["path"].split("/outputs/")[-1]
    p = OUTPUTS_DIR / fname
    if not p.exists() or not p.suffix == ".png":
        await _send(send, 404, "text/plain", "not found"); return
    await _send(send, 200, "image/png", p.read_bytes())

async def route_home(scope, receive, send):
    await _send(send, 200, "text/html; charset=utf-8", HTML)

# ── ASGI app ──────────────────────────────────────────────
async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    p, m = scope["path"], scope["method"]
    if p == "/":                       await route_home(scope, receive, send)
    elif p == "/api/status":           await route_status(scope, receive, send)
    elif p == "/api/generate":         await route_generate(scope, receive, send)
    elif p == "/api/load":             await route_load_model(scope, receive, send)
    elif p.startswith("/outputs/"):    await route_outputs(scope, receive, send)
    else:                              await _send(send, 404, "text/plain", "Not found")

# ── Embedded web UI ───────────────────────────────────────
HTML = """<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fooocus Colab</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--bg-p:#161b22;--bg-h:rgba(177,186,196,.1);--bd:#30363d;
  --fg:#e6edf3;--fg2:#7d8590;--fg3:#484f58;
  --acc:#58a6ff;--grn:#3fb950;--org:#d29922;--red:#f85149;
  color-scheme:dark;
}
html,body{height:100%;background:var(--bg);color:var(--fg);font-family:'Inter',sans-serif;font-size:14px}
body{display:flex;flex-direction:column;overflow:hidden}

.hdr{
  display:flex;align-items:center;gap:10px;padding:10px 20px;
  background:var(--bg-p);border-bottom:1px solid var(--bd);flex-shrink:0;
}
.logo{font-weight:600;font-size:15px;display:flex;align-items:center;gap:8px}
.logo svg{flex-shrink:0}
.spc{flex:1}
.badge{
  display:flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;
  font-size:12px;border:1px solid var(--bd);
}
.badge.ok{color:var(--grn);border-color:var(--grn)}
.badge.warn{color:var(--org);border-color:var(--org)}
.badge .dot{width:7px;height:7px;border-radius:50%;background:currentColor}

.main{flex:1;display:flex;overflow:hidden}

/* LEFT PANEL */
.panel{
  width:340px;flex-shrink:0;display:flex;flex-direction:column;
  background:var(--bg-p);border-right:1px solid var(--bd);overflow-y:auto;
}
.section{padding:16px 20px;border-bottom:1px solid var(--bd)}
.section-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--fg2);margin-bottom:10px}
label{display:block;font-size:12px;color:var(--fg2);margin-bottom:4px}
textarea,input,select{
  width:100%;background:var(--bg);border:1px solid var(--bd);color:var(--fg);
  border-radius:6px;padding:8px 10px;font-size:13px;font-family:'Inter',sans-serif;
  outline:none;transition:border-color .15s;resize:vertical;
}
textarea:focus,input:focus,select:focus{border-color:var(--acc)}
textarea{line-height:1.5}
.row{display:flex;gap:8px}
.row>*{flex:1}
input[type=number]{font-family:'JetBrains Mono',monospace}

.gen-btn{
  width:100%;padding:11px;border-radius:8px;border:none;cursor:pointer;
  background:var(--acc);color:#fff;font-size:14px;font-weight:600;
  font-family:'Inter',sans-serif;
  transition:opacity .15s,transform .1s;display:flex;align-items:center;justify-content:center;gap:8px;
}
.gen-btn:hover{opacity:.88}
.gen-btn:active{transform:scale(.98)}
.gen-btn:disabled{opacity:.45;cursor:not-allowed}

.model-row{display:flex;gap:6px;align-items:flex-end}
.model-row select{flex:1}
.load-btn{
  padding:7px 12px;border-radius:6px;border:1px solid var(--bd);background:none;
  color:var(--fg);font-size:12px;cursor:pointer;white-space:nowrap;font-family:inherit;
  transition:background .15s;
}
.load-btn:hover{background:var(--bg-h)}

/* RIGHT PANEL */
.canvas{flex:1;display:flex;flex-direction:column;overflow:hidden}
.canvas-top{
  display:flex;align-items:center;gap:10px;padding:10px 20px;
  border-bottom:1px solid var(--bd);font-size:13px;color:var(--fg2);flex-shrink:0;
}
.canvas-body{
  flex:1;overflow-y:auto;display:flex;flex-wrap:wrap;
  align-content:flex-start;gap:12px;padding:16px;
}

.empty-state{
  flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:10px;color:var(--fg3);
}
.empty-state svg{opacity:.25}
.empty-state p{font-size:13px;text-align:center;max-width:260px;line-height:1.5;color:var(--fg2)}

.img-card{
  position:relative;border-radius:8px;overflow:hidden;
  border:1px solid var(--bd);flex-shrink:0;cursor:pointer;
  transition:transform .15s,border-color .15s;
}
.img-card:hover{transform:scale(1.02);border-color:var(--acc)}
.img-card img{display:block;width:100%;height:auto}
.img-meta{
  position:absolute;bottom:0;left:0;right:0;
  background:linear-gradient(transparent,rgba(0,0,0,.8));
  padding:20px 8px 6px;font-size:11px;color:rgba(255,255,255,.8);
  display:flex;align-items:center;justify-content:space-between;
}
.placeholder-tag{
  position:absolute;top:6px;left:6px;background:rgba(210,153,34,.85);
  color:#fff;font-size:10px;padding:2px 6px;border-radius:4px;font-weight:600;
}

/* PROGRESS */
.progress{
  display:none;position:fixed;inset:0;z-index:100;
  background:rgba(0,0,0,.7);backdrop-filter:blur(4px);
  flex-direction:column;align-items:center;justify-content:center;gap:14px;
}
.progress.on{display:flex}
.spinner{
  width:40px;height:40px;border:3px solid var(--bd);border-top-color:var(--acc);
  border-radius:50%;animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.progress p{font-size:14px;color:var(--fg2)}

/* Toast */
.toast{
  position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
  background:var(--bg-p);border:1px solid var(--bd);border-radius:7px;
  padding:7px 16px;font-size:13px;box-shadow:0 8px 24px rgba(0,0,0,.5);
  z-index:200;pointer-events:none;opacity:0;transition:opacity .25s;white-space:nowrap;
}
.toast.on{opacity:1}

@media(max-width:680px){
  .main{flex-direction:column}
  .panel{width:100%;border-right:none;border-bottom:1px solid var(--bd);max-height:55vh}
}
</style>
</head>
<body>
<div class="hdr">
  <div class="logo">
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <rect width="20" height="20" rx="4" fill="#58a6ff" opacity=".2"/>
      <path d="M4 15V5l5.5 5L4 15z" fill="#58a6ff"/>
      <path d="M10.5 15V5L16 10l-5.5 5z" fill="#58a6ff" opacity=".5"/>
    </svg>
    Fooocus Colab
  </div>
  <div class="spc"></div>
  <div class="badge warn" id="statusBadge">
    <span class="dot"></span>
    <span id="statusText">Checking…</span>
  </div>
</div>

<div class="main">
  <!-- LEFT: controls -->
  <div class="panel">
    <div class="section">
      <div class="section-title">Model</div>
      <div class="model-row">
        <select id="modelSel"><option value="">No models found</option></select>
        <button class="load-btn" id="loadBtn">Load</button>
      </div>
    </div>

    <div class="section" style="flex:1">
      <div class="section-title">Prompt</div>
      <label>Prompt</label>
      <textarea id="prompt" rows="4" placeholder="a cinematic photo of a mountain lake at sunset, golden hour, 4k"></textarea>
      <label style="margin-top:10px">Negative prompt</label>
      <textarea id="neg" rows="2" placeholder="lowres, bad anatomy, blurry, watermark"></textarea>
    </div>

    <div class="section">
      <div class="section-title">Settings</div>
      <div class="row">
        <div>
          <label>Width</label>
          <input type="number" id="width" value="512" min="64" max="2048" step="64">
        </div>
        <div>
          <label>Height</label>
          <input type="number" id="height" value="512" min="64" max="2048" step="64">
        </div>
      </div>
      <div class="row" style="margin-top:8px">
        <div>
          <label>Steps</label>
          <input type="number" id="steps" value="20" min="1" max="100">
        </div>
        <div>
          <label>Seed</label>
          <input type="number" id="seed" value="" placeholder="random">
        </div>
      </div>
    </div>

    <div class="section">
      <button class="gen-btn" id="genBtn">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 1a.5.5 0 01.5.5v5H14a.5.5 0 010 1H8.5V13a.5.5 0 01-1 0V7.5H2a.5.5 0 010-1h5.5V1.5A.5.5 0 018 1z"/>
        </svg>
        Generate
      </button>
    </div>
  </div>

  <!-- RIGHT: output -->
  <div class="canvas">
    <div class="canvas-top">
      <span id="countLabel">0 images</span>
      <span style="flex:1"></span>
      <button onclick="clearOutputs()" style="background:none;border:none;color:var(--fg2);cursor:pointer;font-size:12px;font-family:inherit">Clear</button>
    </div>
    <div class="canvas-body" id="gallery">
      <div class="empty-state" id="emptyState">
        <svg width="56" height="56" viewBox="0 0 64 64" fill="none">
          <rect x="6" y="10" width="52" height="44" rx="4" stroke="currentColor" stroke-width="2"/>
          <circle cx="22" cy="24" r="5" stroke="currentColor" stroke-width="2"/>
          <path d="M6 42l14-10 10 8 10-12 18 16" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
        </svg>
        <p>Enter a prompt and click Generate to create images.<br>Without a loaded model, placeholder images are returned.</p>
      </div>
    </div>
  </div>
</div>

<div class="progress" id="progress">
  <div class="spinner"></div>
  <p id="progressMsg">Generating…</p>
</div>
<div class="toast" id="toast"></div>

<script>
let images = [];
const $ = id => document.getElementById(id);

// ── Status polling ────────────────────────────────────────
async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const badge = $('statusBadge'), text = $('statusText');
    const sel = $('modelSel');

    if (d.models.length) {
      sel.innerHTML = d.models.map(m => `<option value="${m}"${d.loaded && m === d.loaded_model ? ' selected':''}">${m}</option>`).join('');
    } else {
      sel.innerHTML = '<option value="">No models found</option>';
    }

    if (d.loaded) {
      badge.className = 'badge ok'; text.textContent = 'Model loaded';
    } else if (d.models.length) {
      badge.className = 'badge warn'; text.textContent = `${d.models.length} model(s) — none loaded`;
    } else {
      badge.className = 'badge warn';
      text.textContent = d.torch ? 'No models in models/checkpoints' : 'PyTorch installing…';
    }
  } catch {
    $('statusBadge').className = 'badge warn';
    $('statusText').textContent = 'Server error';
  }
}
refreshStatus();
setInterval(refreshStatus, 8000);

// ── Load model ────────────────────────────────────────────
$('loadBtn').onclick = async () => {
  const model = $('modelSel').value;
  if (!model) { showToast('No model selected'); return; }
  $('loadBtn').disabled = true; $('loadBtn').textContent = 'Loading…';
  try {
    const r = await fetch('/api/load', {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({model})});
    const d = await r.json();
    if (d.ok) showToast(`Loaded ${model}`);
    else showToast('Failed to load model');
  } catch { showToast('Load failed'); }
  finally { $('loadBtn').disabled = false; $('loadBtn').textContent = 'Load'; refreshStatus(); }
};

// ── Generate ──────────────────────────────────────────────
$('genBtn').onclick = async () => {
  const prompt = $('prompt').value.trim();
  if (!prompt) { showToast('Enter a prompt'); return; }

  const prog = $('progress');
  const msg  = $('progressMsg');
  prog.classList.add('on');
  msg.textContent = 'Generating…';
  $('genBtn').disabled = true;

  const t0 = Date.now();
  const timer = setInterval(() => {
    msg.textContent = `Generating… ${((Date.now()-t0)/1000).toFixed(1)}s`;
  }, 200);

  try {
    const r = await fetch('/api/generate', {
      method: 'POST',
      headers: {'content-type':'application/json'},
      body: JSON.stringify({
        prompt,
        negative_prompt: $('neg').value,
        steps:  parseInt($('steps').value) || 20,
        width:  parseInt($('width').value) || 512,
        height: parseInt($('height').value) || 512,
        seed:   $('seed').value ? parseInt($('seed').value) : null,
      })
    });
    const d = await r.json();
    if (d.error) { showToast('Error: ' + d.error); return; }
    addImage(d, prompt);
    if (d.placeholder) showToast('Placeholder (no model loaded)');
  } catch(e) {
    showToast('Request failed');
  } finally {
    clearInterval(timer);
    prog.classList.remove('on');
    $('genBtn').disabled = false;
  }
};

function addImage(d, prompt) {
  images.unshift({...d, prompt});
  renderGallery();
}

function renderGallery() {
  const gallery = $('gallery');
  const empty = $('emptyState');
  $('countLabel').textContent = `${images.length} image${images.length!==1?'s':''}`;
  if (!images.length) { empty.style.display=''; return; }
  empty.style.display = 'none';

  gallery.innerHTML = images.map((img, i) => `
    <div class="img-card" style="width:calc(50% - 6px);min-width:180px" onclick="openImage(${i})">
      <img src="data:image/png;base64,${img.image}" alt="Generated image">
      ${img.placeholder ? '<div class="placeholder-tag">PLACEHOLDER</div>' : ''}
      <div class="img-meta">
        <span title="${img.prompt}">${img.prompt.slice(0,30)}${img.prompt.length>30?'…':''}</span>
        <span>${img.duration}s · #${img.seed}</span>
      </div>
    </div>`).join('');
}

function openImage(i) {
  const img = images[i];
  const w = window.open('', '_blank');
  if (w) { w.document.write(`<img src="data:image/png;base64,${img.image}" style="max-width:100%">`); }
}

function clearOutputs() {
  images = [];
  renderGallery();
}

// ── Toast ─────────────────────────────────────────────────
let toastTimer;
function showToast(msg) {
  const el = $('toast');
  el.textContent = msg; el.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('on'), 2500);
}

// ── Keyboard shortcut ─────────────────────────────────────
document.addEventListener('keydown', e => {
  if ((e.ctrlKey||e.metaKey) && e.key==='Enter') $('genBtn').click();
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7865))
    print(f"Starting Fooocus Colab service on http://0.0.0.0:{port}")
    print(f"Models dir: {MODELS_DIR}")
    print(f"PyTorch available: {HAS_TORCH}")
    print(f"Diffusers available: {HAS_DIFFUSERS}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True, log_level="info")
