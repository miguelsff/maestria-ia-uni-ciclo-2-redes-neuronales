/**
 * PeruFoodNet en el navegador: carga el modelo ONNX, clasifica una foto y dibuja el mapa CAM.
 * Todo el cómputo ocurre en el dispositivo; la imagen no se envía a ningún servidor.
 */
import { decode, resizeLikePillow, toTensor, SIZE } from "./preprocess.js";

const MODEL_URL = "model/perufoodnet.onnx";
const MODEL_BYTES = 41619516; // tamaño conocido, para el progreso si el servidor no envía Content-Length
const UNCERTAIN_BELOW = 0.5;  // mismo umbral que el artículo

const $ = (id) => document.getElementById(id);
const ui = {
  plato: $("plato"), lienzo: $("lienzo"), camara: $("camara"), archivo: $("archivo"),
  estadoModelo: $("estado-modelo"), estadoTexto: $("estado-texto"), progreso: $("progreso"),
  bienvenida: $("bienvenida"), resultado: $("resultado"), intro: $("resultado-intro"),
  nombre: $("resultado-plato"), relleno: $("confianza-relleno"), valor: $("confianza-valor"),
  etiqueta: $("confianza-etiqueta"), aviso: $("aviso-incierto"), verMapa: $("ver-mapa"),
  opciones: $("opciones"), ingredientes: $("ingredientes"), notaIngr: $("nota-ingredientes"),
  tecnico: $("tecnico"), error: $("error"),
};

const state = { session: null, classes: [], names: {}, catalog: null, last: null };

/* ---------- Carga ---------- */

/** Descarga el modelo mostrando el progreso. */
async function fetchModel() {
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`No se pudo descargar el modelo (HTTP ${res.status}).`);
  const total = Number(res.headers.get("Content-Length")) || MODEL_BYTES;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    setProgress(Math.min(1, received / total));
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  return bytes;
}

function setProgress(f) {
  const pct = Math.min(100, Math.round(f * 100));
  ui.progreso.style.width = `${pct}%`;
  ui.progreso.parentElement.setAttribute("aria-valuenow", String(pct));
  ui.estadoTexto.textContent = `Descargando el modelo (${(f * MODEL_BYTES / 1e6).toFixed(0)} de ${(MODEL_BYTES / 1e6).toFixed(0)} MB). Solo la primera vez.`;
}

async function loadAll() {
  const [classes, names, catalog] = await Promise.all([
    fetch("data/classes.json").then((r) => r.json()),
    fetch("data/display_names.json").then((r) => r.json()),
    fetch("data/ingredients.json").then((r) => r.json()),
  ]);
  Object.assign(state, { classes, names, catalog });

  const bytes = await fetchModel();
  ui.estadoTexto.textContent = "Iniciando la red neuronal…";
  ort.env.wasm.wasmPaths = new URL("vendor/ort/", location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  state.session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"], graphOptimizationLevel: "all",
  });
  // Primera pasada de calentamiento para que la primera foto real no tarde más.
  await state.session.run({ image: new ort.Tensor("float32", new Float32Array(3 * SIZE * SIZE), [1, 3, SIZE, SIZE]) });
  ui.estadoModelo.hidden = true;
  document.body.dataset.listo = "true";
}

/* ---------- Inferencia ---------- */

function softmax(logits) {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/**
 * Clasifica una imagen.
 * @param {Blob} blob
 * @returns {Promise<{probs: number[], order: number[], cam: Float32Array, image: object, ms: number}>}
 */
async function classify(blob) {
  const image = await decode(blob);
  const t0 = performance.now();
  const rgb = resizeLikePillow(image.rgba, image.width, image.height);
  const input = new ort.Tensor("float32", toTensor(rgb), [1, 3, SIZE, SIZE]);
  const out = await state.session.run({ image: input });
  const ms = performance.now() - t0;
  const probs = softmax(Array.from(out.logits.data));
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).map(([, i]) => i);
  const [, , gh, gw] = out.cam.dims;
  return { probs, order, cam: out.cam.data, grid: [gh, gw], image, ms };
}

/* ---------- Dibujo ---------- */

/** Rectángulo para cubrir el lienzo cuadrado conservando proporciones (como object-fit: cover). */
function coverRect(w, h, size) {
  const s = Math.max(size / w, size / h);
  return { x: (size - w * s) / 2, y: (size - h * s) / 2, w: w * s, h: h * s };
}

/**
 * Efecto "foco": las zonas que más aportan a la decisión quedan iluminadas y el resto se
 * cubre de morado. Se lee bien incluso sobre comida amarilla, donde un mapa de calor se pierde.
 * @param {number} v activación normalizada en [0, 1]
 */
function spotlight(v) {
  const a = Math.round(215 * Math.pow(1 - v, 1.6));
  return [36, 18, 47, a];
}

function drawPlate(result, showCam) {
  const { lienzo } = ui;
  const ctx = lienzo.getContext("2d");
  const size = lienzo.width;
  const r = coverRect(result.image.width, result.image.height, size);
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(result.image.bitmap, r.x, r.y, r.w, r.h);
  if (!showCam) return;
  const [gh, gw] = result.grid;
  const k = result.order[0];
  const plane = result.cam.subarray(k * gh * gw, (k + 1) * gh * gw);
  let max = 0;
  for (const v of plane) max = Math.max(max, v);
  const small = new OffscreenCanvas(gw, gh);
  const sctx = small.getContext("2d");
  const img = sctx.createImageData(gw, gh);
  for (let i = 0; i < plane.length; i++) {
    const v = max > 0 ? Math.max(0, plane[i]) / max : 0;
    img.data.set(spotlight(v), i * 4);
  }
  sctx.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.globalCompositeOperation = "source-over";
  // El modelo ve la foto completa estirada a 224×224: el mapa cubre toda la foto.
  ctx.drawImage(small, r.x, r.y, r.w, r.h);
  ctx.restore();
}

const pct = (p) => `${(p * 100).toFixed(1).replace(".", ",")} %`;
const pretty = (cls) => state.names[cls] || cls;

function recipeFor(cls) {
  const entry = state.catalog.dishes[cls];
  if (!entry) return { codes: [], empty: true };
  const codes = [...new Set([...(entry.efsa || []), ...(entry.usda || [])])].sort();
  return { codes, empty: entry.efsa === null && entry.usda === null };
}

function render(result) {
  state.last = result;
  const top = result.order[0];
  const p = result.probs[top];
  const uncertain = p < UNCERTAIN_BELOW;

  ui.bienvenida.hidden = true;
  ui.resultado.hidden = false;
  ui.resultado.dataset.incierto = String(uncertain);
  ui.intro.textContent = uncertain ? "Podría ser" : "Es probable que sea";
  ui.nombre.textContent = pretty(state.classes[top]);
  ui.valor.textContent = pct(p);
  ui.etiqueta.textContent = uncertain ? "de puntuación: incierto" : "de puntuación";
  ui.aviso.hidden = !uncertain;
  ui.relleno.style.setProperty("--valor", String(p));

  ui.opciones.replaceChildren(...result.order.slice(1, 4).map((i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="opcion-nombre"></span><span class="opcion-barra"><span></span></span><span class="opcion-valor"></span>`;
    li.querySelector(".opcion-nombre").textContent = pretty(state.classes[i]);
    li.querySelector(".opcion-barra span").style.setProperty("--valor", String(result.probs[i]));
    li.querySelector(".opcion-valor").textContent = pct(result.probs[i]);
    return li;
  }));

  const { codes, empty } = recipeFor(state.classes[top]);
  ui.ingredientes.replaceChildren(...codes.map((c) => {
    const li = document.createElement("li");
    li.textContent = state.catalog.ingredients[c]?.es || c;
    return li;
  }));
  ui.notaIngr.textContent = codes.length
    ? `Según la receta documentada de ${pretty(state.classes[top]).toLowerCase()} en PeruFoodNet (tablas 1 y 2). No se detectan en la foto.`
    : empty
      ? "El estudio no enumera ingredientes alergénicos para este plato. Eso no garantiza que no tenga ninguno."
      : "La receta documentada no registra ingredientes alergénicos de la lista.";

  ui.tecnico.textContent = `Inferencia en tu dispositivo: ${Math.round(result.ms)} ms. Entrada ${result.image.width}×${result.image.height} px redimensionada a 224×224 con el mismo algoritmo del entrenamiento.`;

  ui.plato.dataset.estado = "servido";
  ui.plato.classList.remove("aterriza");
  void ui.plato.offsetWidth; // reinicia la animación
  ui.plato.classList.add("aterriza");
  drawPlate(result, ui.verMapa.checked);
  // En el celular el resultado queda debajo del plato: lo acercamos a la vista.
  if (window.matchMedia("(max-width: 899px)").matches) {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    setTimeout(() => ui.resultado.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" }), 450);
  }
}

/* ---------- Eventos ---------- */

async function handle(blob) {
  ui.error.hidden = true;
  if (!blob || !blob.type.startsWith("image/")) {
    showError("Ese archivo no es una imagen. Elige una foto en JPG, PNG o WebP.");
    return;
  }
  if (!state.session) {
    showError("El modelo aún se está preparando. Espera unos segundos y vuelve a intentarlo.");
    return;
  }
  ui.plato.dataset.estado = "pensando";
  try {
    render(await classify(blob));
  } catch (err) {
    console.error(err);
    ui.plato.dataset.estado = state.last ? "servido" : "vacio";
    showError("No se pudo leer esa imagen. Prueba con otra foto.");
  }
}

function showError(msg) {
  ui.error.textContent = msg;
  ui.error.hidden = false;
}

for (const input of [ui.camara, ui.archivo]) {
  input.addEventListener("change", () => { handle(input.files?.[0]); input.value = ""; });
}

document.querySelectorAll("[data-ejemplo]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const blob = await fetch(btn.dataset.ejemplo).then((r) => r.blob());
    handle(blob);
  });
});

ui.verMapa.addEventListener("change", () => state.last && drawPlate(state.last, ui.verMapa.checked));

["dragenter", "dragover"].forEach((t) => ui.plato.addEventListener(t, (e) => { e.preventDefault(); ui.plato.classList.add("sobre"); }));
["dragleave", "drop"].forEach((t) => ui.plato.addEventListener(t, () => ui.plato.classList.remove("sobre")));
ui.plato.addEventListener("drop", (e) => { e.preventDefault(); handle(e.dataTransfer.files?.[0]); });
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) handle(item.getAsFile());
});

if (location.protocol === "file:") {
  // Abierto con doble clic: el navegador bloquea la carga del modelo desde archivos locales.
  ui.estadoTexto.innerHTML = "Esta página se abrió como archivo y el navegador no deja cargar el modelo así. " +
    "Ciérrala y abre <strong>Iniciar_demo_Windows.bat</strong> (o <strong>iniciar_demo.sh</strong> en Mac o Linux), " +
    "o usa la versión publicada en GitHub Pages.";
  ui.estadoModelo.dataset.error = "true";
  ui.progreso.parentElement.hidden = true;
} else loadAll().catch((err) => {
  console.error(err);
  ui.estadoTexto.textContent = "No se pudo cargar el modelo. Revisa tu conexión y recarga la página.";
  ui.estadoModelo.dataset.error = "true";
});

if ("serviceWorker" in navigator && location.protocol === "https:") {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

// Gancho para pruebas automáticas (no afecta a la interfaz).
window.__perufoodnet = { classify: async (blob) => { const r = await classify(blob); return { probs: r.probs, top: state.classes[r.order[0]] }; }, ready: () => !!state.session };
