// ─────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────
let currentMode = 'describe';
let keywords    = [];

// ─────────────────────────────────────────────────────────────
// PATIENT DETAILS HELPERS
// ─────────────────────────────────────────────────────────────
function getPatientDetails() {
  return {
    name:        document.getElementById('p-name')?.value.trim()        || '',
    age:         document.getElementById('p-age')?.value                || '',
    gender:      document.getElementById('p-gender')?.value             || '',
    bp:          document.getElementById('p-bp')?.value.trim()          || '',
    sugar:       document.getElementById('p-sugar')?.value              || '',
    heart_rate:  document.getElementById('p-hr')?.value                 || '',
    temperature: document.getElementById('p-temp')?.value               || '',
    weight:      document.getElementById('p-weight')?.value             || '',
    height:      document.getElementById('p-height')?.value             || '',
    spo2:        document.getElementById('p-spo2')?.value               || '',
    cholesterol: document.getElementById('p-cholesterol')?.value        || '',
    medications: document.getElementById('p-meds')?.value.trim()        || '',
    family_history: document.getElementById('p-family')?.value.trim()   || '',
  };
}

// Flag vitals as high / low / ok for display
function vitalFlag(field, value) {
  const v = parseFloat(value);
  if (isNaN(v)) return '';
  const ranges = {
    sugar:       { low: 70,  high: 140 },
    heart_rate:  { low: 60,  high: 100 },
    temperature: { low: 36,  high: 37.5 },
    spo2:        { low: 95,  high: 101 },
    cholesterol: { low: 0,   high: 200 },
  };
  const r = ranges[field];
  if (!r) return '';
  if (v > r.high) return 'flag-high';
  if (v < r.low)  return 'flag-low';
  return 'flag-ok';
}

// ─────────────────────────────────────────────────────────────
// MODE SWITCHING
// ─────────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  document.getElementById('mode-describe').style.display  = mode === 'describe' ? 'block' : 'none';
  document.getElementById('mode-keywords').style.display  = mode === 'keywords' ? 'block' : 'none';
  document.getElementById('btn-describe').classList.toggle('active', mode === 'describe');
  document.getElementById('btn-keywords').classList.toggle('active', mode === 'keywords');
}

// ─────────────────────────────────────────────────────────────
// KEYWORD MODE
// ─────────────────────────────────────────────────────────────
const kwInput = document.getElementById('symptom-input');
if (kwInput) {
  kwInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      addKeyword(kwInput.value);
    }
  });
}
function addKeyword(raw) {
  const val = raw.trim().replace(/,$/, '').toLowerCase();
  if (val && !keywords.includes(val)) { keywords.push(val); renderKeywordTags(); }
  if (kwInput) kwInput.value = '';
}
function removeKeyword(s) {
  keywords = keywords.filter(x => x !== s);
  renderKeywordTags();
}
function renderKeywordTags() {
  const row = document.getElementById('tags-row');
  if (!row) return;
  row.innerHTML = keywords.map(s =>
    `<div class="tag">${s} <span onclick="removeKeyword('${s}')">×</span></div>`
  ).join('');
}

// ─────────────────────────────────────────────────────────────
// EXAMPLES
// ─────────────────────────────────────────────────────────────
function loadExample(mode, data) {
  setMode(mode);
  if (mode === 'describe') {
    document.getElementById('nlp-input').value = data;
    document.getElementById('extracted-tags-section').style.display = 'none';
  } else {
    keywords = [...data];
    renderKeywordTags();
  }
}

// ─────────────────────────────────────────────────────────────
// MAIN PREDICTION HANDLER
// ─────────────────────────────────────────────────────────────
async function runPrediction() {
  const topN    = parseInt(document.getElementById('topN').value);
  const patient = getPatientDetails();
  const btn     = document.querySelector('.predict-btn');
  const loader  = document.getElementById('loader');
  const results = document.getElementById('results');

  btn.disabled          = true;
  loader.style.display  = 'block';
  results.style.display = 'none';

  try {
    let data;

    if (currentMode === 'describe') {
      const text = document.getElementById('nlp-input').value.trim();
      if (!text) { alert('Please describe your symptoms.'); btn.disabled = false; loader.style.display = 'none'; return; }

      const res = await fetch('/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, top_n: topN, patient })
      });
      data = await res.json();

      if (data.extracted?.length > 0) {
        document.getElementById('extracted-tags-section').style.display = 'block';
        document.getElementById('extracted-tags').innerHTML =
          data.extracted.map(s => `<div class="tag extracted">${s}</div>`).join('');
      }

    } else {
      if (keywords.length === 0) { alert('Please enter at least one symptom.'); btn.disabled = false; loader.style.display = 'none'; return; }

      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: keywords, top_n: topN, patient })
      });
      data = await res.json();
      data.extracted = keywords;
    }

    renderResults(data, patient);

  } catch (err) {
    results.innerHTML     = `<div class="card" style="color:#ef4444">Error: ${err.message}</div>`;
    results.style.display = 'block';
  }

  loader.style.display = 'none';
  btn.disabled         = false;
}

// ─────────────────────────────────────────────────────────────
// RENDER RESULTS
// ─────────────────────────────────────────────────────────────
function renderResults(data, patient) {
  const el = document.getElementById('results');

  if (data.message) {
    el.innerHTML = `<div class="card" style="color:#fbbf24">⚠️ ${data.message}</div>`;
    el.style.display = 'block';
    return;
  }
  if (!data.predictions?.length) {
    el.innerHTML = `<div class="card">No predictions found. Try describing symptoms differently.</div>`;
    el.style.display = 'block';
    return;
  }

  const extracted = data.extracted || data.matched || [];
  const unmatched = data.unmatched || [];

  // ── Patient summary panel ──
  let patientHtml = '';
  const hasPatient = patient.name || patient.age || patient.gender || patient.bp ||
                     patient.sugar || patient.heart_rate || patient.temperature;
  if (hasPatient) {
    const bmiVal  = (patient.weight && patient.height)
      ? (parseFloat(patient.weight) / Math.pow(parseFloat(patient.height)/100, 2)).toFixed(1)
      : null;

    const fields = [
      patient.name        && { label: 'Patient',      value: patient.name,        flag: '' },
      patient.age         && { label: 'Age',           value: patient.age + ' yrs', flag: '' },
      patient.gender      && { label: 'Gender',        value: patient.gender,      flag: '' },
      patient.bp          && { label: 'Blood Pressure',value: patient.bp,          flag: '' },
      patient.sugar       && { label: 'Blood Sugar',   value: patient.sugar + ' mg/dL', flag: vitalFlag('sugar', patient.sugar) },
      patient.heart_rate  && { label: 'Heart Rate',    value: patient.heart_rate + ' bpm', flag: vitalFlag('heart_rate', patient.heart_rate) },
      patient.temperature && { label: 'Temperature',   value: patient.temperature + ' °C', flag: vitalFlag('temperature', patient.temperature) },
      patient.spo2        && { label: 'SpO₂',          value: patient.spo2 + '%',  flag: vitalFlag('spo2', patient.spo2) },
      patient.cholesterol && { label: 'Cholesterol',   value: patient.cholesterol + ' mg/dL', flag: vitalFlag('cholesterol', patient.cholesterol) },
      bmiVal              && { label: 'BMI',            value: bmiVal,              flag: parseFloat(bmiVal) > 30 ? 'flag-high' : parseFloat(bmiVal) < 18.5 ? 'flag-low' : 'flag-ok' },
      patient.medications && { label: 'Medications',   value: patient.medications, flag: '' },
      patient.family_history && { label: 'Family History', value: patient.family_history, flag: '' },
    ].filter(Boolean);

    patientHtml = `
      <div class="patient-summary">
        <h3>📋 Patient Profile</h3>
        ${fields.map(f => `
          <div class="p-detail">
            <span class="p-label">${f.label}</span>
            <span class="p-value ${f.flag}">${f.value}</span>
          </div>`).join('')}
      </div>`;
  }

  // ── Result header ──
  let html = `
    <div class="result-header">
      <h2>Top ${data.predictions.length} Predictions</h2>
      <small>${extracted.length} symptom(s) analysed</small>
    </div>
    ${patientHtml}`;

  if (extracted.length > 0) {
    html += `<div class="extracted-summary">
      ✅ <strong>Symptoms analysed:</strong> ${extracted.join(' · ')}
    </div>`;
  }
  if (unmatched.length > 0) {
    html += `<div class="unmatched-warning">
      ⚠️ Not recognised (try medical terms): <strong>${unmatched.join(', ')}</strong>
    </div>`;
  }

  // ── Disease cards ──
  data.predictions.forEach((p, i) => {
    const isRare    = p.DiseaseType === 'rare';
    const typeClass = isRare ? 'rare'    : 'common';
    const typeLabel = isRare ? '🔴 Rare' : '🟡 Common';
    const conf      = parseFloat(p.Confidence) || 0;
    const barW      = Math.min(conf * 6, 100);
    const confColor = conf > 10 ? '#a855f7' : conf > 3 ? '#f59e0b' : '#6b7280';

    let metaHtml = `<span class="meta-pill pill-${typeClass}">${typeLabel}</span>`;
    if (p.OrphaCode)    metaHtml += `<span class="meta-pill">OrphaCode: ${p.OrphaCode}</span>`;
    if (p.AgeOfOnset && p.AgeOfOnset !== 'Unknown')         metaHtml += `<span class="meta-pill">Onset: ${p.AgeOfOnset}</span>`;
    if (p.TypeOfInheritance && p.TypeOfInheritance !== 'Unknown') metaHtml += `<span class="meta-pill">${p.TypeOfInheritance}</span>`;
    if (p.PrevalenceClass && p.PrevalenceClass !== 'Unknown') metaHtml += `<span class="meta-pill">Prevalence: ${p.PrevalenceClass}</span>`;
    if (p.GeneSymbols && p.GeneSymbols !== 'Unknown' && p.GeneSymbols !== '')
      metaHtml += `<span class="meta-pill">Genes: ${p.GeneSymbols.split('|').slice(0,3).join(', ')}</span>`;

    const desc = (p.Description && p.Description !== '')
      ? `<div class="dc-desc">${p.Description.substring(0,160)}...</div>` : '';

    html += `
      <div class="disease-card ${typeClass}">
        <div class="dc-top">
          <div class="dc-name">#${i+1} ${p.DiseaseName}</div>
          <div class="dc-conf" style="color:${confColor}">${conf.toFixed(2)}%</div>
        </div>
        <div class="conf-bar-wrap"><div class="conf-bar" style="width:${barW}%"></div></div>
        <div class="dc-meta">${metaHtml}</div>
        ${desc}
      </div>`;
  });

  el.innerHTML     = html;
  el.style.display = 'block';
  el.scrollIntoView({ behavior: 'smooth' });
}