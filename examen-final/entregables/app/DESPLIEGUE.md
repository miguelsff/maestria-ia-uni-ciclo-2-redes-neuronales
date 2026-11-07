# Web de inferencia y presentación (GitHub Pages)

Esta carpeta agrega al repositorio:

| Ruta | Qué es |
|---|---|
| `docs/index.html` | Portada del sitio con enlaces a la app y a la presentación |
| `docs/app/` | Web de inferencia: cámara o foto, top-3, aviso de incertidumbre, mapa CAM y ficha de ingredientes |
| `docs/slides/` | Presentación en HTML (20 diapositivas, mismo contenido y tiempos que `Equipo3_Slides.pdf`) |
| `src/export_onnx.py` | Exporta `perufoodnet_best.pt` a `docs/app/model/perufoodnet.onnx` (logits + mapa CAM) |
| `src/verify_onnx.py` | Comprueba que el ONNX reproduce los logits de PyTorch |

## Publicar en GitHub Pages

1. Copien `docs/` y los dos scripts de `src/` a la raíz de su repositorio
   (`src/` va junto a `models.py` y `datasets.py`, porque los importa).
2. `git add docs src/export_onnx.py src/verify_onnx.py && git commit -m "Web de inferencia y presentación" && git push`
   El modelo pesa 41,6 MB, por debajo del límite de 100 MB de GitHub: no hace falta Git LFS.
3. En GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch**,
   rama `main`, carpeta `/docs`. Guardar.
4. En uno o dos minutos queda en `https://<usuario>.github.io/<repositorio>/`.
   Para su repo: `https://miguelsff.github.io/maestria-ia-uni-ciclo-2-redes-neuronales/`.

El QR de la diapositiva 18 se genera solo con la URL donde se publique, así que no hay que editarlo.

## Demo en su PC (sin publicar)

- **Windows:** doble clic en `Iniciar_demo_Windows.bat`. **No necesita Python**: usa PowerShell, que viene con Windows 10 y 11
  (`servidor_demo.ps1`). Abre el navegador en la portada; no cierren la ventana negra mientras usan la demo. Si el puerto 8000
  está ocupado, toma el siguiente libre y lo muestra en la ventana.
- **Mac o Linux:** `bash iniciar_demo.sh` (usa Python 3).

Descompriman **todo** el ZIP antes de abrir el lanzador: necesita la carpeta `docs` a su lado.
Si alguien abre `docs/app/index.html` con doble clic, la web avisa que debe usar el lanzador, porque el navegador bloquea la
carga del modelo desde archivos locales.

## Video de respaldo

`PeruFoodNet_demo.mp4` (54 s, 1280×810, con subtítulos) muestra la web clasificando aguadito, el error de ají de gallina → locro
y una foto subida de lomo saltado. También está dentro de la diapositiva 18, que lo reproduce sola: la demo se ve aunque falle
internet en la sustentación.

## Probar en su PC antes de publicar

```bash
cd docs
python -m http.server 8000
```

Abrir `http://localhost:8000/app/` y `http://localhost:8000/slides/`. (Abrir el HTML con doble clic no funciona:
el navegador bloquea la carga del modelo desde `file://`.) La cámara del celular solo se habilita por HTTPS, es decir,
en GitHub Pages.

## Regenerar el modelo de la web

```bash
pip install onnx onnxruntime
python src/export_onnx.py --checkpoint perufoodnet_best.pt --output docs/app/model/perufoodnet.onnx
python src/verify_onnx.py --checkpoint perufoodnet_best.pt --onnx docs/app/model/perufoodnet.onnx \
       --images data/ejemplo_aguadito.jpg results/test/ejemplos/error.jpg
```

Resultado verificado: diferencia máxima de logits 3,8 × 10⁻⁶; aguadito 0,997145 y ají de gallina → locro 0,305457,
iguales a `predict.py`. En el navegador se obtienen exactamente los mismos valores.

## Cómo funciona la web

- **Inferencia en el dispositivo** con ONNX Runtime Web (WebAssembly, incluido en `docs/app/vendor/ort/`). La foto no se envía a ningún servidor.
- **Preprocesamiento idéntico al del entrenamiento.** `preprocess.js` reimplementa el redimensionamiento bilineal de Pillow;
  da los mismos píxeles, bit a bit, que `transforms.Resize((224, 224))`. Sin normalización ImageNet, como YOLO en `datasets.py`.
- **Mapa CAM.** La cabeza de YOLO26m-cls es `conv 1×1 → promedio global → lineal`, así que el mapa
  `M_c = Σ_k w_ck A_k + b_c` cumple exactamente `promedio(M_c) = logit_c` (comprobado en `verify_onnx.py`). En esta cabeza coincide con Grad-CAM.
  No cambia el modelo ni sus métricas: es interpretación adicional (R5).
- **Mismo umbral que el artículo:** puntuación menor que 0,50 se marca como incierta.
- **Ficha de ingredientes** desde `data/ingredients.json` (tablas 1 y 2 de Arzola et al., 2025), igual que `predict.py`.
- **Funciona sin internet** tras la primera visita (service worker) y se puede instalar en el celular como app.

## Presentación

La barra de controles flota abajo y se oculta sola a los 2,5 s; reaparece al mover el ratón.
Teclas: **→ / espacio** avanzar, **←** retroceder, **F** pantalla completa, **T** iniciar o pausar el cronómetro,
**R** reiniciar el cronómetro. El cronómetro se pone en rojo si pasan del tiempo previsto para la diapositiva.
En celular o tablet se desliza con el dedo. Para exportar a PDF: imprimir desde Chrome con "Guardar como PDF".

Cambios frente a `Equipo3_Slides.pdf`: mismo contenido, presentadores y tiempos. Se agregaron el mapa CAM en la
diapositiva 15, la latencia de la web en la 17 y la demo (video + QR) en la 18.

## Usar la app desde el celular en la misma red Wi-Fi (sin internet)

1. Conectar la laptop y el celular a la **misma red Wi-Fi**.
2. En la laptop, doble clic en `Iniciar_demo_red_local.bat` y aceptar el permiso de administrador
   (abre el puerto 8000 en el firewall de Windows con la regla «PeruFoodNet demo»).
3. La ventana muestra una dirección como `http://192.168.1.25:8000/app/`. Escribirla en el navegador del celular.
4. Al terminar, cerrar la ventana y ejecutar `Cerrar_acceso_red.bat` para quitar la regla del firewall.

Si el celular no conecta: comprobar que la red de Windows esté como «Privada», que no sea una red de invitados o
universitaria con aislamiento de clientes (bloquean la comunicación entre equipos) y que no haya una VPN activa.
En ese caso, usar la web publicada en GitHub Pages o compartir internet desde el celular a la laptop.
