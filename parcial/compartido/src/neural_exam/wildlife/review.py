"""Persist visual review galleries without inventing human judgments."""

import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neural_exam.common.io import fingerprint, read_csv, read_json, write_csv, write_json


def _draw(record: dict, names: list[str], destination: Path, selected: dict | None = None) -> None:
    with Image.open(record["image"]) as image:
        image = image.convert("RGB")
        image.thumbnail((960, 704))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=15)
        for box in [selected] if selected else record["boxes"]:
            cx, cy, width, height = box["box"]
            bounds = (
                (cx - width / 2) * image.width,
                (cy - height / 2) * image.height,
                (cx + width / 2) * image.width,
                (cy + height / 2) * image.height,
            )
            draw.rectangle(bounds, outline="#ffb000", width=3)
            draw.text(
                (max(0, bounds[0]), max(0, bounds[1] - 17)),
                names[box["class_id"]],
                font=font,
                fill="#ffb000",
                stroke_width=1,
                stroke_fill="black",
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)


def _gallery(rows: list[dict], path: Path, title: str, occlusion: bool = False) -> None:
    field = "level" if occlusion else "status"
    choices = (
        [
            ["", "Sin revisar"],
            ["low", "Baja (>75 % visible)"],
            ["medium", "Media (25-75 %)"],
            ["high", "Alta (<25 %)"],
            ["ambiguous", "Ambigua o truncada"],
        ]
        if occlusion
        else [["pending", "Sin revisar"], ["reviewed", "Correcta"], ["needs_changes", "Requiere corrección"]]
    )
    payload = [{**row, "preview": str(Path(row["gallery"]).relative_to(path.parent))} for row in rows]
    # JSON escapes keep category names from becoming executable HTML.
    script_data = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")
    criteria = (
        "Evaluar visibilidad corporal, no tamaño. Marcar como ambigua la visibilidad incierta o el truncamiento. Mínimo: 30 objetos por nivel."
        if occlusion
        else "Comprobar que cada caja cubra el animal correcto, que las categorías coincidan y que los fondos estén vacíos. Una fila correcta requiere revisión visual."
    )
    document = r"""<!doctype html><html lang="es"><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:16px system-ui;background:#f3f4f6;color:#111827;margin:24px}header{position:sticky;top:0;background:#fff;padding:16px;z-index:1;border:1px solid #ddd}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}article{background:white;padding:12px;border:1px solid #ccc}.photo{position:relative}img{display:block;width:100%;cursor:zoom-in}.box{position:absolute;border:3px solid #ffb000;box-sizing:border-box;pointer-events:none}textarea{width:95%;min-height:60px}select,input,button{font:inherit;padding:8px}small{overflow-wrap:anywhere}</style>
<header><h1>__TITLE__</h1><p>__CRITERIA__</p><label>Evaluador: <input id="reviewer" placeholder="Nombre de quien revisa"></label> <button id="save">Guardar CSV revisado</button><p id="counts"></p><button id="previous">Anterior</button> <span id="page"></span> <button id="next">Siguiente</button> <button id="pending">Ir a la primera pendiente</button><p>Las imágenes se pueden abrir en otra pestaña para ampliar. Guardar el CSV con su nombre original en la carpeta de resultados. La exportación conserva la autoría de las filas que no se modificaron.</p></header><main id="cards"></main>
<script>const rows=__DATA__, choices=__CHOICES__, field=__FIELD__;const root=document.getElementById('cards');let page=0;const pageSize=24;
function counts(){const c={};rows.forEach(r=>c[r[field]||'pending']=(c[r[field]||'pending']||0)+1);document.getElementById('counts').textContent=JSON.stringify(c)}
function render(){root.replaceChildren();document.getElementById('page').textContent=(page+1)+' / '+Math.max(1,Math.ceil(rows.length/pageSize));
rows.slice(page*pageSize,(page+1)*pageSize).forEach(r=>{const card=document.createElement('article'),link=document.createElement('a'),img=document.createElement('img');img.src=r.preview;img.loading='lazy';link.href=r.preview;link.target='_blank';link.className='photo';link.style.display='block';link.append(img);if(r.normalized_box){const [cx,cy,w,h]=r.normalized_box,b=document.createElement('span');b.className='box';Object.assign(b.style,{left:(cx-w/2)*100+'%',top:(cy-h/2)*100+'%',width:w*100+'%',height:h*100+'%'});link.append(b)}card.append(link);const caption=document.createElement('p');caption.textContent=(r.species||r.split||'')+' | '+r.image_id;card.append(caption);const select=document.createElement('select');choices.forEach(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;select.append(o)});select.value=r[field]||choices[0][0];select.onchange=()=>{r[field]=select.value;r._dirty=true;counts()};card.append(select);const note=document.createElement('textarea');note.placeholder='Evidencia visual o problema observado';note.value=r.visual_evidence||'';note.oninput=()=>{r.visual_evidence=note.value;r._dirty=true};card.append(note);root.append(card)});counts()}
document.getElementById('previous').onclick=()=>{page=Math.max(0,page-1);render()};document.getElementById('next').onclick=()=>{page=Math.min(Math.max(0,Math.ceil(rows.length/pageSize)-1),page+1);render()};document.getElementById('pending').onclick=()=>{const i=rows.findIndex(r=>!r[field]||r[field]==='pending'||r[field]==='needs_changes');if(i>=0){page=Math.floor(i/pageSize);render()}};
document.getElementById('save').onclick=()=>{const name=document.getElementById('reviewer').value.trim();if(!name){alert('Ingresar el nombre del evaluador');return}const fields=Object.keys(rows[0]).filter(k=>!['preview','normalized_box','_dirty'].includes(k));const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const data=[fields.join(','),...rows.map(r=>{if(r._dirty){r.reviewer=name;r._dirty=false}return fields.map(k=>quote(r[k])).join(',')})].join('\r\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\ufeff'+data],{type:'text/csv;charset=utf-8'}));a.download='__CSV__';a.click();URL.revokeObjectURL(a.href)};render();</script></html>"""
    document = (
        document.replace("__TITLE__", html.escape(title))
        .replace("__CRITERIA__", html.escape(criteria))
        .replace("__DATA__", script_data)
        .replace("__CHOICES__", json.dumps(choices))
        .replace("__FIELD__", json.dumps(field))
        .replace("__CSV__", "occlusion_review.csv" if occlusion else "label_review.csv")
    )
    path.write_text(document, encoding="utf-8")


def prepare_label_review(manifest: dict, output: Path, seed: int = 42) -> Path:
    """Sample five positives per class and twenty backgrounds, balancing train cameras when available."""
    path = output / "label_review.csv"
    sample_path = output / "label_review_sample.json"
    if sample_path.exists() and read_json(sample_path)["manifest_identity"] != manifest["identity"]:
        raise ValueError("Label review belongs to another data manifest; use a separate output directory.")
    if path.exists():
        rows = read_csv(path)
        if not sample_path.exists():
            raise ValueError("The label review has no original sample receipt.")
    else:
        candidates = [(split, record) for split in ("train", "val") for record in manifest["records"][split]]
        generator = np.random.default_rng(seed)
        candidates = [candidates[index] for index in generator.permutation(len(candidates))]
        selected = {}
        for class_id in range(len(manifest["names"])):
            group = [
                (split, record)
                for split, record in candidates
                if any(box["class_id"] == class_id for box in record["boxes"])
            ]
            selected.update({record["image_id"]: (split, record) for split, record in group[:5]})
        backgrounds = [(split, record) for split, record in candidates if not record["boxes"]]
        training_backgrounds = [(split, record) for split, record in backgrounds if split == "train"]
        if training_backgrounds:
            grouped = {}
            for item in training_backgrounds:
                grouped.setdefault(item[1]["camera_id"], []).append(item)
            backgrounds = []
            while grouped and len(backgrounds) < 20:
                for camera in sorted(list(grouped)):
                    backgrounds.append(grouped[camera].pop(0))
                    if not grouped[camera]:
                        del grouped[camera]
                    if len(backgrounds) == 20:
                        break
        selected.update({record["image_id"]: (split, record) for split, record in backgrounds[:20]})
        rows = []
        for identifier, (split, record) in sorted(selected.items()):
            gallery = output / "label_gallery" / f"{fingerprint(identifier)[:16]}.png"
            _draw(record, manifest["names"], gallery)
            rows.append(
                {
                    "image_id": identifier,
                    "split": split,
                    "gallery": str(gallery),
                    "status": "pending",
                    "visual_evidence": "",
                    "reviewer": "",
                }
            )
        write_csv(path, rows)
        write_json(
            sample_path,
            {
                "manifest_identity": manifest["identity"],
                "seed": seed,
                "image_ids": sorted(row["image_id"] for row in rows),
            },
        )
    _gallery(rows, output / "label_review.html", "Revisión de etiquetas CCT20")
    lookup = {record["image_id"]: record for split in ("train", "val") for record in manifest["records"][split]}
    font = ImageFont.load_default(size=18)
    for start in range(0, len(rows), 12):
        group = rows[start : start + 12]
        sheet = Image.new("RGB", (1440, 900), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(group):
            x, y = index % 4 * 360, index // 4 * 300
            with Image.open(row["gallery"]) as image:
                image = image.convert("RGB")
                image.thumbnail((352, 250))
                sheet.paste(image, (x + 4, y + 4))
            species = (
                ", ".join(sorted({manifest["names"][box["class_id"]] for box in lookup[row["image_id"]]["boxes"]}))
                or "empty"
            )
            draw.text((x + 4, y + 255), f"{start + index + 1}: {species}", font=font, fill="black")
            draw.text((x + 4, y + 278), f"{row['split']} | {row['image_id'][:12]}", font=font, fill="black")
        sheet.save(output / f"label_contact_{start // 12 + 1:02d}.png")
    status = label_review_status(path)
    write_json(output / "label_review_status.json", status)
    if "train_class_boxes" in manifest:
        from neural_exam.common.figures import bar_plot

        bar_plot(
            manifest["names"],
            [manifest["train_class_boxes"][name] for name in manifest["names"]],
            output / "class_balance.png",
            "Distribución de animales en entrenamiento",
            "Cajas anotadas",
        )
    return path


def label_review_status(path: Path, require_evidence: bool = False) -> dict:
    """Keep human review pending until every sampled image has been approved."""
    rows = read_csv(path) if path.exists() else []
    sample_path = path.with_name("label_review_sample.json")
    if rows and (
        not sample_path.exists() or sorted(row["image_id"] for row in rows) != read_json(sample_path)["image_ids"]
    ):
        raise ValueError("Label review rows do not match the original visual sample.")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate images in label review.")
    if any(row["status"] not in {"pending", "reviewed", "needs_changes"} for row in rows):
        raise ValueError("Invalid label review status.")
    complete = bool(rows) and all(row["status"] == "reviewed" and row.get("reviewer", "").strip() for row in rows)
    require_evidence = require_evidence or (
        sample_path.exists() and read_json(sample_path).get("require_evidence", False)
    )
    missing_evidence = sum(not row.get("visual_evidence", "").strip() for row in rows)
    if require_evidence and missing_evidence:
        complete = False
    return {
        "status": "reviewed" if complete else "human_review_pending",
        "images": len(rows),
        "counts": dict(Counter(row["status"] for row in rows)),
        "missing_visual_evidence": missing_evidence,
        "evidence_required": require_evidence,
    }


def architecture_review_status(path: Path) -> dict:
    """Require a documented human resolution of the exam's implementation rule."""
    if not path.is_file():
        return {"status": "human_review_pending", "reason": "No documented architecture decision", "receipt": str(path)}
    decision = read_json(path)
    complete = (
        decision.get("reviewed") is True
        and decision.get("decision") in {"pretrained_detector_permitted", "student_components_verified"}
        and bool(str(decision.get("reviewer", "")).strip())
        and bool(str(decision.get("evidence", "")).strip())
    )
    return {"status": "reviewed" if complete else "human_review_pending", "receipt": str(path), "decision": decision}


def prepare_occlusion_review(manifest: dict, output: Path, seed: int = 42, extend: int = 0, full: bool = False) -> Path:
    """Extend a deterministic object sample while preserving all existing human evidence."""
    if extend < 0:
        raise ValueError("Occlusion extension must be nonnegative.")
    path = output / "occlusion_review.csv"
    rows = read_csv(path) if path.exists() else []
    objects = {
        box["annotation_id"]: (record, box) for record in manifest["records"]["trans_test"] for box in record["boxes"]
    }
    reviewed = {row["annotation_id"] for row in rows}
    if len(reviewed) != len(rows) or not reviewed <= objects.keys():
        raise ValueError("Occlusion sheet has duplicate or unknown annotation IDs.")
    available = sorted(objects.keys() - reviewed)
    generator = np.random.default_rng(seed + len(rows))
    count = len(available) if full else min(extend if rows else 120, len(available))
    if rows and not extend and not full:
        count = 0
    for index in generator.choice(len(available), count, replace=False):
        identifier = available[index]
        record, box = objects[identifier]
        gallery = (
            Path(record["image"]) if full else output / "occlusion_gallery" / f"{fingerprint(identifier)[:16]}.png"
        )
        rows.append(
            {
                "annotation_id": identifier,
                "image_id": record["image_id"],
                "species": manifest["names"][box["class_id"]],
                "camera_id": record["camera_id"],
                "gallery": str(gallery),
                "level": "",
                "visual_evidence": "",
                "reviewer": "",
            }
        )
    for row in rows:
        record, box = objects[row["annotation_id"]]
        if row["image_id"] != record["image_id"] or row["species"] != manifest["names"][box["class_id"]]:
            raise ValueError("Occlusion object/image/species mismatch.")
        gallery = Path(row["gallery"])
        if full:
            row["gallery"] = record["image"]
        elif not gallery.exists():
            _draw(record, manifest["names"], gallery, box)
    write_csv(
        path,
        rows,
        ["annotation_id", "image_id", "species", "camera_id", "gallery", "level", "visual_evidence", "reviewer"],
    )
    display_rows = (
        [{**row, "normalized_box": objects[row["annotation_id"]][1]["box"]} for row in rows] if full else rows
    )
    _gallery(display_rows, output / "occlusion_review.html", "Oclusión en cámaras nuevas", occlusion=True)
    return path
