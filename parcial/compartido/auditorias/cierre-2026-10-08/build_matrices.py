"""Build the exam review ledger from verified evidence and final QA receipts."""

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARCIAL = HERE.parents[2]
VALID_STATES = {"cumple", "parcial", "falta", "no verificado"}
ROWS = []


def add(question, deliverable, requirement, evidence, limitation="", pending="",
        state="cumple", mandatory=True):
    assert state in VALID_STATES
    ROWS.append({"pregunta": question, "entregable": deliverable,
                 "subrequisito": requirement, "evidencia": evidence,
                 "estado": state, "limitacion": limitation or "Ninguna adicional.",
                 "pendiente": pending or "Ninguno técnico.", "obligatorio": mandatory})


MRI = "E:/Kaggle/exam-output-mri/stratified-v2/mri/"
FLOOD = "E:/Kaggle/exam-output-flood/flood/"
WILD = "E:/Kaggle/exam-output-wildlife-v6/wildlife/"
OCC = "E:/Kaggle/exam-output-wildlife-v6-occlusion-sample-600/"
MR = "mri-scientific.json; pregunta_1_mri/entregable/informe_mri.pdf"
FR = "flood-scientific.json; pregunta_2_inundaciones/entregable/informe_inundaciones.pdf"
WR = "wildlife-scientific.json; pregunta_3_fauna/entregable/informe_fauna.pdf"
OR = "occlusion-scientific.json; " + OCC + "accepted_evaluation/6b2983c33abccbc1/results.json"


def scientific_rows():
    add(1, 1, "Doce etiquetas coincidentes con el esquema original",
        MR + "; " + MRI + "manifest.json; compartido/src/neural_exam/mri/labels.py: LABELS",
        "El examen presenta nombres provisionales; se cotejaron las doce columnas originales.")
    add(1, 1, "Extracción multilingüe, negación, incertidumbre y abstención",
        MR + "; 600 reportes reextraídos; lexicón de 13 idiomas en labels.py",
        "Las reglas no garantizan igual calidad entre idiomas.")
    add(1, 1, "Comparación contra etiquetas expertas",
        MR + "; " + MRI + "label_agreement.json: 58 estudios, 24 filas por etiqueta/confianza",
        "La abstención cambia el soporte observado, que se informa.")
    add(1, 1, "Acuerdo y ruido estimado recalculados",
        MR + "; agreement y estimated_disagreement coinciden a tolerancia 1e-12",
        "Desacuerdo observado no equivale a tasa causal de errores del radiólogo.")
    add(1, 2, "CNN propia entrenada desde cero",
        MR + "; " + MRI + "scratch_bce_weighted/history.csv, complete.json y best.pt: 48 épocas")
    add(1, 2, "Backbone ImageNet con ajuste y doce salidas independientes",
        MR + "; compartido/src/neural_exam/mri/model.py; resnet50_bce_weighted: 12 épocas",
        "Los presupuestos y parada temprana difieren entre arquitecturas.")
    add(1, 2, "Configuración, historiales y checkpoints de ambas arquitecturas",
        MR + "; ocho variantes con execution.json, history.csv, complete.json y best.pt",
        "run.json histórico conserva running; los recibos de finalización y checkpoints demuestran cierre.")
    add(1, 2, "Separación de estudios y selección del modelo",
        "compartido/auditorias/revision-2026-10-08/structure.json; " + MRI + "manifest.json: 434 entrenamiento, 108 validación débil, 58 expertos",
        "Separación comprobada por estudio; no existe identificador global fiable para demostrar separación por paciente.")
    add(1, 3, "AUC macro oficial sobre las doce etiquetas",
        MR + "; 464 predicciones expertas en ocho variantes; incluye Synovitis; CNN/BCE/ponderada=0.577642622 y ResNet/BCE/ponderada=0.618907985")
    add(1, 3, "AUC por etiqueta",
        MR + "; metrics.json y predicciones guardadas cotejados para las doce etiquetas")
    add(1, 3, "Precisión, recall y F1",
        MR + "; métricas macro y por etiqueta recalculadas a tolerancia 1e-12")
    add(1, 3, "Conjunto experto, umbral elegido y justificación",
        MR + "; gold=58, umbral 0.5 declarado y justificado en informe",
        "Los 58 expertos fueron reutilizados tras particiones anteriores; no constituyen prueba externa intacta.")
    add(1, 4, "Al menos dos factores de ablación",
        MR + "; diseño 2 arquitecturas × BCE/focal × confianza ponderada/filtrada: ocho corridas")
    add(1, 4, "Comparaciones identificables y conclusiones sustentadas",
        MR + "; tabla por variante; máximo AUC 0.619867594 en resnet50_focal_weighted",
        "Una semilla y presupuestos diferentes limitan atribuciones causales.")
    add(1, 5, "Grad-CAM para al menos tres hallazgos distintos",
        MR + "; mapas ACL, Medial Meniscus y Effusion; arrays no constantes normalizados [0,1]")
    add(1, 5, "Plausibilidad anatómica discutida",
        "pregunta_1_mri/fuentes_informe/discussion.tex; informe_mri.pdf y mapas revisados",
        "Mapas difusos o periféricos se describen como limitación; no validación de localización clínica.")
    add(1, 6, "Prevalencia de las doce etiquetas",
        MR + "; 36 filas de prevalencia por etiqueta y partición con positivos y denominadores observados")
    add(1, 6, "Mitigación implementada y evaluada",
        MR + "; pérdidas BCE/focal implementadas y comparación cuantitativa en ocho corridas",
        "La prevalencia débil está afectada por ruido y abstenciones.")
    add(1, 7, "Discusión de diferencias geográficas y lingüísticas",
        "pregunta_1_mri/entregable/informe_mri.pdf; discussion.tex: cobertura, lexicón, negación e idiomas",
        "No se presentan métricas fiables de equidad por centro/idioma por falta de soporte.")
    add(1, 7, "Implicancias clínicas y validación local",
        "informe_mri.pdf: discusión de sesgo y riesgo de despliegue, aproximadamente media página",
        "No se afirma aptitud para uso clínico.")

    add(2, 1, "U-Net personalizada implementada en PyTorch",
        "compartido/src/neural_exam/flood/model.py; " + FR)
    add(2, 1, "Entrada SAR con las dos bandas VV/VH",
        FR + "; dataset, configuración y carga GeoTIFF de dos canales")
    add(2, 2, "Etiquetas doradas frente a doradas más débiles",
        FR + "; seis corridas gold_* y gold_weak_*; entrenamiento gold=251, weak=4150",
        "Se respetan las particiones oficiales; no se usan todos los dorados para entrenar.")
    add(2, 2, "Comparación de pérdidas o arquitectura",
        FR + "; BCE, Dice y combinada cruzadas con las dos fuentes de supervisión")
    add(2, 2, "Configuración y finalización de las seis corridas",
        FR + "; execution.json, history.csv, complete.json, best.pt y predicciones guardadas",
        "Una semilla; los presupuestos efectivos difieren por la cantidad de etiquetas débiles.")
    add(2, 3, "IoU, F1, precisión y recall en distribución",
        FR + "; 88 chips test; baseline mean-chip IoU=0.393926867, F1 global=0.798229203")
    add(2, 3, "Definición y agregación de IoU",
        FR + "; IoU por chip con unión no vacía; global=0.664210850 para baseline",
        "Media por chip, IoU global y convención oficial con epsilon en chips vacíos son cantidades distintas.")
    add(2, 3, "Tratamiento de píxeles sin datos",
        FR + "; 618 GeoTIFF, nodata=-1, máscaras válidas y georreferenciación comprobados")
    add(2, 3, "Matriz de confusión y comentario",
        FR + "; baseline TP1884956 FP322880 FN630052 TN17606071; figura y discusión")
    add(2, 4, "Bolivia excluida del entrenamiento y selección",
        FR + "; cuatro particiones oficiales; Bolivia=15, train251/val86/test88 sin intersecciones")
    add(2, 4, "Evaluación externa y brecha frente a ID",
        FR + "; seis variantes test/Bolivia; baseline mean-chip Bolivia=0.455477085",
        "La brecha es firmada: algunos modelos mejoran en Bolivia. No se fuerza una caída inexistente.")
    add(2, 4, "Al menos dos hipótesis explicativas",
        "pregunta_2_inundaciones/entregable/informe_inundaciones.pdf: distribución de agua/cobertura y condiciones SAR",
        "Hipótesis plausibles; no causas demostradas experimentalmente.")
    add(2, 5, "Tres tipos de fallo con ejemplos visuales concretos",
        "flood-qualitative.json; Pakistan_70625, Pakistan_849790, Bolivia_379434; figuras del informe")
    add(2, 5, "Explicaciones sustentadas por observaciones",
        "flood-qualitative.json: backscatter, FP/FN y errores de borde recalculados; 81.25% cerca del borde en tercer caso",
        "Sombra y doble rebote son interpretaciones posibles; sin información adicional no se certifica mecanismo físico.")
    add(2, 6, "Bono de fusión Sentinel-2 frente a SAR",
        "No hay corrida SAR+óptico en los seis experimentos preservados.",
        "Bono opcional; no bloquea los cinco entregables obligatorios.",
        "Ninguno obligatorio; no se añade entrenamiento nuevo.", state="falta", mandatory=False)

    add(3, 1, "Entrenamiento YOLO ajustado terminado y modelo seleccionado",
        WR + "; training_complete.json:39 épocas; mejor época29 por cis_val; checkpoint b2a47c3b0b22d927…")
    add(3, 1, "Resolución y estrategia de aumentos documentadas",
        WR + "; YOLOv8s, imgsz640, AdamW, batch16; args.yaml y configuración V6")
    add(3, 1, "Fotografías vacías tratadas explícitamente como fondo",
        "structure.json; cohorte de 1000 vacías verificadas en metadata/split oficial de entrenamiento",
        "Proceden de ocho cámaras de la colección completa, adicionales al benchmark; la distinción está declarada.")
    add(3, 2, "mAP@0.5 y mAP@0.5:0.95 recalculados",
        WR + "; cis_test=0.687751884/0.435752672; trans_test=0.423057069/0.262485537")
    add(3, 2, "Desglose y soporte por especie",
        WR + "; 14 especies benchmark, soporte12461 objetos conocidos/17620 nuevos",
        "Categorías sin positivos elegibles tienen AP no definido; no se sustituyen por cero.")
    add(3, 2, "Discusión del desbalance y colección frente a benchmark",
        "informe_fauna.pdf: clases raras, fox/badger y 14 especies del benchmark frente a colección completa",
        "Los resultados del benchmark no cubren todas las categorías de la colección completa.")
    add(3, 3, "Particiones contrastadas con anotaciones oficiales",
        "structure.json: cinco particiones ECCV y 45 305 cajas cotejadas; tres correcciones bird con revisión humana")
    add(3, 3, "Separación de cámaras, secuencias e imágenes",
        "structure.json: imágenes disjuntas, secuencias train/holdouts disjuntas y cámaras nuevas ausentes de train")
    add(3, 3, "Brecha conocidos/nuevos cuantificada",
        WR + "; diferencia mAP50:95=0.173267135; diferencia mAP50=0.264694815")
    add(3, 3, "Evidencia visual y discusión del aprendizaje del fondo",
        "informe_fauna.pdf:12 Grad-CAM y ejemplos de cámaras conocidas/nuevas",
        "Tres mapas planos y atención periférica declarados; Grad-CAM no demuestra causalmente memorización del fondo.")
    add(3, 4, "Grupos de tamaño, soportes y desempeño",
        WR + "; grupos <1%,1–5%,>5%; trans1922/12169/3529; AP50:95=0.161862/0.322044/0.286956")
    add(3, 4, "Trazabilidad de la revisión humana aceptada",
        "structure.json; " + OCC + "review_acceptance.json y CSV aceptado:600 originales,582 aceptados,18 excluidos",
        "398 casos sin nota, permitidos; caso121 conserva medium de Miguel; no se fabricaron explicaciones humanas.")
    add(3, 4, "Grupos de oclusión y soportes",
        OR + "; low340,medium69,high48,ambiguous125; últimos125 fuera de comparación de niveles",
        "Submuestra independiente aceptada, sin garantía de representatividad completa tras 18 exclusiones.")
    add(3, 4, "Métricas, intervalos y diferencias de oclusión",
        OR + "; AP50:95 low0.087523278,medium0.027141706,high0.024376059;200 bootstrap por secuencia/nivel",
        "Todos los 18647 frames, incluidos vacíos; otras cajas se ignoran según protocolo. AP de submuestra no comparable directamente con mAP global V6.")
    add(3, 4, "Resultados incorporados al informe y notebook finales",
        "pregunta_3_fauna/entregable/informe_fauna.pdf,03_wildlife.ipynb,anexo_occlusion.pdf y04_occlusion_582.ipynb",
        "El anexo no reemplaza la discusión requerida dentro de las 6–10 páginas.",
        "Verificación final del documento integrado.", state="no verificado")
    add(3, 5, "Cámara de demostración ausente del entrenamiento",
        WR + "; camera7 pertenece a cámaras trans y no al entrenamiento")
    add(3, 5, "Conteos por especie y periodo",
        WR + "; activity.json:1853 frames,211 días observados,319 filas especie/día,1565 detecciones")
    add(3, 5, "Especies observadas y fotografías sin detección",
        WR + ";11 especies detectadas,1420 fotos con detección,433/1853 sin detección=23.3675%",
        "Sin detección no garantiza vacío real; conteos son detecciones, no abundancia ni individuos únicos.")
    add(3, 6, "Falsos negativos y falsos positivos en conservación",
        "informe_fauna.pdf: discusión aproximadamente media página sobre costos y revisión especializada")
    add(3, 6, "Riesgo de despliegue sin validación local",
        "informe_fauna.pdf: caída en cámaras nuevas, validación local y restricciones de uso")


def common_rows(final):
    for q in (1, 2, 3):
        for name, flag in (
            ("Informe NeurIPS de 6–10 páginas", "report_format"),
            ("Cinco autores y al menos cinco referencias", "authors_references"),
            ("Motivación, antecedentes, metodología, experimentos, resultados y discusión", "report_content"),
            ("ODT editable y PDF exportado desde él", "odt_edit_export"),
            ("Notebook final ejecutado sin errores", "notebooks"),
        ):
            passed = final.get("questions", {}).get(str(q), {}).get(flag, False)
            add(q, "común", name, "validation-final.json; entregable de la pregunta correspondiente",
                pending="Ninguno técnico." if passed else "Completar comprobación final.",
                state="cumple" if passed else "no verificado")
    for name, flag in (
        ("Cifras y hashes consistentes entre originales y entregas", "scientific_integrity"),
        ("Código publicado y accesible con credenciales del propietario", "published"),
        ("Entorno y comando completo documentados", "commands"),
        ("Pruebas de empaquetado y ejecución desde copia limpia", "clean_checkout"),
        ("Migración sin pérdida de contenido único y con manifiesto", "migration"),
        ("Preservación de cambios anteriores y secretos", "preservation"),
    ):
        passed = final.get(flag, False)
        add("transversal", "común", name, "validation-final.json; manifiesto.json; publicación y recibos de pruebas",
            pending="Ninguno técnico." if passed else "Completar comprobación final.",
            state="cumple" if passed else "no verificado")
    add("transversal", "común", "Componentes propios frente al uso específico de YOLO existente",
        "CNN/heads MRI y U-Net propios; YOLOv8s existente y módulo de monitoreo propio declarados en fauna",
        "Tensión entre instrucción general y entregable específico de YOLO ajustado; no consta aceptación docente.",
        "Obtener interpretación docente si se necesita resolver el criterio académico.", state="parcial")
    for name, limitation, pending in (
        ("Reproducción científica completa desde entrenamiento", "Se comprobaron predicciones y empaquetado, no nuevos entrenamientos.", "Solo si se solicita, ejecutar todas las corridas originales con datos y GPU."),
        ("Acceso del docente al repositorio privado", "El acceso del propietario no demuestra acceso del docente.", "Conservar constancia de acceso docente."),
        ("Aceptación de reglas Kaggle por los cinco integrantes", "No existe constancia individual completa.", "Conservar aceptación y condiciones aplicables al uso académico."),
        ("Envío por la plataforma del curso", "No existe recibo de envío.", "Conservar recibo al entregar."),
        ("Presentación del 30 de octubre de 2026", "Fecha futura al cierre técnico del 8 de octubre.", "Conservar constancia tras la presentación."),
    ):
        add("transversal", "común", name, "No hay constancia comprobada en los archivos revisados.",
            limitation, pending, state="no verificado")


def markdown(rows, title):
    lines = [f"# {title}", "", "Cada fila corresponde a una comprobación independiente. Los originales científicos permanecen inalterados.", "",
             "Las rutas de entrega se interpretan desde `parcial/`; los recibos científicos y `validation-final.json`, desde `compartido/auditorias/cierre-2026-10-08/`. `structure.json` corresponde a `compartido/auditorias/revision-2026-10-08/structure.json`.", "",
             "| Pregunta | Entregable | Subrequisito | Evidencia concreta | Estado | Limitación | Pendiente |",
             "|---|---|---|---|---|---|---|"]
    keys = ("pregunta", "entregable", "subrequisito", "evidencia", "estado", "limitacion", "pendiente")
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]).replace("|", "\\|").replace("\n", " ") for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main():
    final_path = HERE / "validation-final.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else {}
    scientific_rows()
    for row in ROWS:
        if row["pregunta"] == 3 and row["entregable"] == 4 and row["estado"] == "no verificado":
            if final.get("questions", {}).get("3", {}).get("occlusion_integrated", False):
                row["estado"] = "cumple"
                row["pendiente"] = "Ninguno técnico."
    common_rows(final)
    mandatory = {(r["pregunta"], r["entregable"]) for r in ROWS if r["obligatorio"] and isinstance(r["entregable"], int)}
    assert len(mandatory) == 18
    ledger = {"review_date_peru": "2026-10-08", "updated_at_utc": datetime.now(timezone.utc).isoformat(),
              "mandatory_deliverables": 18, "rows": ROWS,
              "scope": "Recálculo de predicciones guardadas y verificación de entrega; no reproducción científica desde entrenamiento."}
    (HERE / "matrices.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    folders = {1: "pregunta_1_mri", 2: "pregunta_2_inundaciones", 3: "pregunta_3_fauna"}
    for q, folder in folders.items():
        own = [r for r in ROWS if r["pregunta"] == q]
        out = PARCIAL / folder / "evidencias"
        out.mkdir(parents=True, exist_ok=True)
        (out / "matriz_revision.md").write_text(markdown(own, f"Revisión de la pregunta {q}"), encoding="utf-8")
        (out / "estado.json").write_text(json.dumps({"pregunta": q, "rows": own,
            "mandatory_deliverables": sum(1 for question, _ in mandatory if question == q)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "matrices_completas.md").write_text(markdown(ROWS, "Matrices completas de los 18 entregables"), encoding="utf-8")
    print(json.dumps({"subrequirements": len(ROWS), "mandatory_deliverables": len(mandatory),
                      "states": {s: sum(r["estado"] == s for r in ROWS) for s in sorted(VALID_STATES)}}))


if __name__ == "__main__":
    main()
