let sessionId = null;
let currentIndex = 0;
let currentItem = null;
let itemStartedAt = null;
let previousRenderedFields = null;
let k = 25;

const startCard = document.getElementById('start-card');
const surveyCard = document.getElementById('survey-card');
const doneCard = document.getElementById('done-card');
const startBtn = document.getElementById('start-btn');
const submitBtn = document.getElementById('submit-btn');
const backBtn = document.getElementById('back-btn');

function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }
function setError(id, message) {
  const el = document.getElementById(id);
  if (!message) { hide(el); el.textContent = ''; return; }
  el.textContent = message;
  show(el);
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed: ${res.status}`);
  return data;
}

async function getJSON(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed: ${res.status}`);
  return data;
}

startBtn.addEventListener('click', async () => {
  setError('start-error', '');
  startBtn.disabled = true;
  try {
    const participantCode = document.getElementById('participant-code').value.trim();
    if (!participantCode) {
      throw new Error('Please enter your participant code before starting.');
    }
    const data = await postJSON('/api/start', {participant_code: participantCode});
    sessionId = data.session_id;
    k = data.k;
    currentIndex = 0;
    hide(startCard);
    show(surveyCard);
    await loadItem(currentIndex);
  } catch (err) {
    setError('start-error', err.message);
  } finally {
    startBtn.disabled = false;
  }
});

backBtn.addEventListener('click', async () => {
  if (currentIndex <= 0) return;
  currentIndex -= 1;
  await loadItem(currentIndex);
});

submitBtn.addEventListener('click', async () => {
  setError('submit-error', '');
  const rating = document.querySelector('input[name="rating"]:checked');
  if (!rating) {
    setError('submit-error', 'Please choose an acceptability rating before continuing.');
    return;
  }
  const attention = currentItem && currentItem.attention_check;
  let attentionAnswer = '';
  if (attention && attention.required) {
    const attentionSelect = document.getElementById('attention-check-answer');
    attentionAnswer = attentionSelect ? attentionSelect.value : '';
    if (!attentionAnswer) {
      setError('submit-error', 'Please answer the attention-check question before continuing.');
      return;
    }
  }
  submitBtn.disabled = true;
  try {
    const elapsed = itemStartedAt ? Date.now() - itemStartedAt : null;
    await postJSON(`/api/session/${sessionId}/submit`, {
      index: currentIndex,
      rating: rating.value,
      confidence: document.getElementById('confidence').value,
      free_text: document.getElementById('free-text').value,
      elapsed_ms: elapsed,
      attention_check_answer: attentionAnswer
    });
    if (currentIndex + 1 >= k) {
      hide(surveyCard);
      show(doneCard);
    } else {
      currentIndex += 1;
      await loadItem(currentIndex);
    }
  } catch (err) {
    setError('submit-error', err.message);
  } finally {
    submitBtn.disabled = false;
  }
});

async function loadItem(index) {
  setError('submit-error', '');
  currentItem = await getJSON(`/api/session/${sessionId}/item/${index}`);
  itemStartedAt = Date.now();
  renderItem(currentItem);
}

function formatExample(example) {
  const text = String(example || '').trim();
  if (!text) return '';
  return /^example\s*:/i.test(text) ? text : `Example: ${text}`;
}

function renderAttentionCheck(item, table) {
  const check = item.attention_check;
  if (!check || !table) return;

  const tr = document.createElement('tr');
  tr.className = 'attention-check-row';

  const tdLabel = document.createElement('td');
  tdLabel.textContent = 'Attention check';

  const tdValue = document.createElement('td');
  const question = document.createElement('div');
  question.className = 'attention-check-question';
  question.textContent = check.question || 'Which value is shown in the scenario details above?';
  tdValue.appendChild(question);

  const helpText = check.note || 'Please choose the matching value from the scenario details above.';
  if (helpText) {
    const help = document.createElement('div');
    help.className = 'field-help';
    help.textContent = helpText;
    tdValue.appendChild(help);
  }

  const select = document.createElement('select');
  select.id = 'attention-check-answer';
  select.name = 'attention_check_answer';
  select.required = true;
  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = 'Select the value shown above';
  select.appendChild(blank);

  const options = check.options || [];
  for (const opt of options) {
    const option = document.createElement('option');
    option.value = String(opt);
    option.textContent = String(opt);
    select.appendChild(option);
  }
  const existing = item.existing_response && item.existing_response.attention_check_answer;
  if (existing) select.value = String(existing);
  tdValue.appendChild(select);

  tr.appendChild(tdLabel);
  tr.appendChild(tdValue);
  table.appendChild(tr);
}

function fieldChangeKey(row) {
  return String((row && (row.ci_field_label || row.label)) || '');
}

function fieldChangeValue(row) {
  return String((row && row.value) || '').trim();
}

function previousFieldMap(fields) {
  const out = {};
  for (const row of fields || []) {
    const key = fieldChangeKey(row);
    if (key) out[key] = fieldChangeValue(row);
  }
  return out;
}

function renderItem(item) {
  document.getElementById('progress-text').textContent = `Question ${item.index + 1} of ${item.total}`;
  document.getElementById('progress-fill').style.width = `${((item.index + 1) / item.total) * 100}%`;

  const flow = item.flow || {};
  const participantVisible = item.participant_visible || {};
  const taskTitleEl = document.getElementById('task-title');
  if (taskTitleEl) {
    taskTitleEl.textContent = flow.task_label || participantVisible.task_title || flow.task || 'Smart-space scenario';
  }
  document.getElementById('vignette').textContent = item.vignette || participantVisible.vignette || '';

  const table = document.getElementById('ci-table');
  table.innerHTML = '';
  const fields = item.display_fields || participantVisible.display_fields || [];
  const prevMap = previousFieldMap(previousRenderedFields);
  const showChanges = previousRenderedFields && item.index !== 0;
  for (const row of fields) {
    const tr = document.createElement('tr');
    const tdLabel = document.createElement('td');
    const tdValue = document.createElement('td');
    tdLabel.textContent = row.label || '';

    const key = fieldChangeKey(row);
    const changed = showChanges && key && prevMap[key] !== undefined && prevMap[key] !== fieldChangeValue(row);
    if (changed && row.label !== 'Scenario overview') {
      tr.classList.add('changed-field');
      const pill = document.createElement('span');
      pill.className = 'changed-pill';
      pill.textContent = 'changed';
      tdLabel.appendChild(pill);
      window.setTimeout(() => {
        tr.classList.remove('changed-field');
        const existingPill = tr.querySelector('.changed-pill');
        if (existingPill) existingPill.remove();
      }, 3200);
    }

    const main = document.createElement('div');
    main.textContent = row.value || '';
    tdValue.appendChild(main);

    const helpText = row.help || row.description || '';
    if (helpText) {
      const help = document.createElement('div');
      help.className = 'field-help';
      help.textContent = helpText;
      tdValue.appendChild(help);
    }

    const exampleText = formatExample(row.example);
    if (exampleText) {
      const example = document.createElement('div');
      example.className = 'field-help field-example';
      example.textContent = exampleText;
      tdValue.appendChild(example);
    }

    tr.appendChild(tdLabel);
    tr.appendChild(tdValue);
    table.appendChild(tr);
  }

  renderAttentionCheck(item, table);
  previousRenderedFields = fields.map(row => ({...row}));

  document.querySelectorAll('input[name="rating"]').forEach(input => {
    input.checked = false;
    input.closest('label').classList.remove('selected');
  });
  document.getElementById('confidence').value = '';
  document.getElementById('free-text').value = '';

  if (item.existing_response) {
    const r = document.querySelector(`input[name="rating"][value="${item.existing_response.rating}"]`);
    if (r) {
      r.checked = true;
      r.closest('label').classList.add('selected');
    }
    if (item.existing_response.confidence !== null && item.existing_response.confidence !== undefined) {
      document.getElementById('confidence').value = String(item.existing_response.confidence);
    }
    document.getElementById('free-text').value = item.existing_response.free_text || '';
  }

  backBtn.disabled = item.index === 0;
  submitBtn.textContent = item.index + 1 >= item.total ? 'Submit and finish' : 'Submit and continue';
}

document.querySelectorAll('input[name="rating"]').forEach(input => {
  input.addEventListener('change', () => {
    document.querySelectorAll('input[name="rating"]').forEach(i => i.closest('label').classList.remove('selected'));
    input.closest('label').classList.add('selected');
  });
});
