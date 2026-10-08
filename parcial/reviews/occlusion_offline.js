/* CSV portability and conservative merging, independent of the browser UI. */
const ReviewData = (() => {
  const fields = ['annotation_id', 'image_id', 'species', 'camera_id', 'sequence_id',
    'size', 'gallery', 'stratum_population', 'stratum_sample', 'sample_weight',
    'level', 'visual_evidence', 'reviewer'];
  const humanFields = ['level', 'visual_evidence', 'reviewer'];
  const levels = ['', 'low', 'medium', 'high', 'ambiguous'];
  const complete = row => Boolean(row.level && row.visual_evidence.trim() && row.reviewer.trim());
  const sameReview = (a, b) => humanFields.every(field => a[field] === b[field]);
  const hasReview = row => humanFields.some(field => row[field].trim());

  function parseCsv(text) {
    text = text.replace(/^\uFEFF/, '');
    const records = [];
    let record = [], value = '', quoted = false, closed = false;
    const cell = () => { record.push(value); value = ''; closed = false; };
    const line = () => { cell(); records.push(record); record = []; };
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (quoted) {
        if (char === '"') {
          if (text[i + 1] === '"') { value += '"'; i++; }
          else { quoted = false; closed = true; }
        } else value += char;
      } else if (char === '"') {
        if (value || closed) throw Error('CSV inválido: comillas fuera de una celda.');
        quoted = true;
      } else if (char === ',') cell();
      else if (char === '\r' || char === '\n') {
        line();
        if (char === '\r' && text[i + 1] === '\n') i++;
      } else {
        if (closed) throw Error('CSV inválido: texto después de las comillas.');
        value += char;
      }
    }
    if (quoted) throw Error('CSV incompleto: hay comillas sin cerrar.');
    if (value || record.length || closed) line();
    while (records.length && records[records.length - 1].every(value => !value)) records.pop();
    if (!records.length || records[0].join(',') !== fields.join(',')) {
      throw Error('Use el CSV descargado de esta galería, con sus 13 columnas originales.');
    }
    return records.slice(1).map((record, index) => {
      if (record.length !== fields.length) throw Error(`CSV inválido en la fila ${index + 2}.`);
      return Object.fromEntries(fields.map((field, index) => [field, record[index]]));
    });
  }

  function validate(rows, reference) {
    if (!Array.isArray(rows) || rows.length !== reference.length) {
      throw Error(`El CSV debe contener los ${reference.length} casos originales, incluso los pendientes.`);
    }
    const originals = new Map(reference.map(row => [row.annotation_id, row]));
    const seen = new Set();
    for (const row of rows) {
      const original = row && originals.get(row.annotation_id);
      if (!original || seen.has(row.annotation_id)) throw Error('Hay casos duplicados o ajenos a esta muestra.');
      seen.add(row.annotation_id);
      if (fields.some(field => typeof row[field] !== 'string')) throw Error('Hay celdas faltantes o inválidas.');
      if (fields.some(field => !humanFields.includes(field) && row[field] !== original[field])) {
        throw Error('El CSV cambió identificadores, imágenes o datos de la muestra. No se cargó.');
      }
      if (!levels.includes(row.level)) throw Error('El CSV contiene un nivel de oclusión desconocido.');
    }
    return rows;
  }

  function csv(rows) {
    const quote = value => /[",\r\n]/.test(value) ? '"' + value.replace(/"/g, '""') + '"' : value;
    return '\uFEFF' + [fields.join(','), ...rows.map(row => fields.map(field => quote(row[field])).join(','))]
      .join('\r\n') + '\r\n';
  }

  function merge(current, incoming, baseline) {
    validate(incoming, baseline);
    validate(current, baseline);
    const byId = new Map(incoming.map(row => [row.annotation_id, row]));
    const base = new Map(baseline.map(row => [row.annotation_id, row]));
    const conflicts = [], changes = [];
    current.forEach((row, index) => {
      const next = byId.get(row.annotation_id);
      if (!hasReview(next) || sameReview(next, base.get(row.annotation_id)) || sameReview(next, row)) return;
      if (!sameReview(row, base.get(row.annotation_id)) && hasReview(row)) conflicts.push(index + 1);
      else changes.push([index, next]);
    });
    if (conflicts.length) {
      throw Error(`No se cargó el CSV: ${conflicts.length} casos tienen revisiones diferentes ` +
        `(casos ${conflicts.slice(0, 8).join(', ')}${conflicts.length > 8 ? ', …' : ''}). ` +
        'Conserve ambos CSV y envíelos al responsable para resolver las diferencias.');
    }
    const result = current.map(row => ({...row}));
    for (const [index, next] of changes) {
      for (const field of humanFields) result[index][field] = next[field];
    }
    return {rows: result, changed: changes.length};
  }
  return {fields, humanFields, complete, parseCsv, validate, csv, merge};
})();
