let sessionId = null;
let currentIndex = 0;
let currentItem = null;
let itemStartedAt = null;
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

  const comprehension = currentItem && currentItem.comprehension_check;
  let comprehensionAnswer = '';
  if (comprehension && comprehension.required) {
    const comprehensionSelect = document.getElementById('comprehension-check-answer');
    comprehensionAnswer = comprehensionSelect ? comprehensionSelect.value : '';
    if (!comprehensionAnswer) {
      setError('submit-error', 'Please answer the scenario-check question before continuing.');
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
      attention_check_answer: attentionAnswer,
      comprehension_check_answer: comprehensionAnswer
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

function renderQualityCheck(check, table, idPrefix, existingAnswer) {
  if (!check || !table) return;

  const tr = document.createElement('tr');
  tr.className = `${idPrefix}-row quality-check-row`;

  const tdLabel = document.createElement('td');
  tdLabel.textContent = check.field_label || 'Check question';

  const tdValue = document.createElement('td');
  const question = document.createElement('div');
  question.className = 'quality-check-question';
  question.textContent = check.question || 'Please choose the requested answer.';
  tdValue.appendChild(question);

  const helpText = check.note || '';
  if (helpText) {
    const help = document.createElement('div');
    help.className = 'field-help';
    help.textContent = helpText;
    tdValue.appendChild(help);
  }

  const select = document.createElement('select');
  select.id = `${idPrefix}-answer`;
  select.name = `${idPrefix}_answer`;
  select.required = true;
  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = check.placeholder || 'Select an answer';
  select.appendChild(blank);

  const options = check.options || [];
  for (const opt of options) {
    const option = document.createElement('option');
    option.value = String(opt);
    option.textContent = String(opt);
    select.appendChild(option);
  }
  if (existingAnswer) select.value = String(existingAnswer);
  tdValue.appendChild(select);

  tr.appendChild(tdLabel);
  tr.appendChild(tdValue);
  table.appendChild(tr);
}

function renderAttentionCheck(item, table) {
  const existing = item.existing_response && item.existing_response.attention_check_answer;
  renderQualityCheck(item.attention_check, table, 'attention-check', existing);
}

function renderComprehensionCheck(item, table) {
  const existing = item.existing_response && item.existing_response.comprehension_check_answer;
  renderQualityCheck(item.comprehension_check, table, 'comprehension-check', existing);
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
  for (const row of fields) {
    const tr = document.createElement('tr');
    if (row.emphasis === 'primary' || row.emphasis === 'variable' || row.emphasis === 'changed') tr.classList.add('key-detail-row');
    if (row.emphasis === 'changed') tr.classList.add('changed-row');
    if (row.ci_field_label === 'Output') tr.classList.add('data-shared-row');
    const tdLabel = document.createElement('td');
    const tdValue = document.createElement('td');
    tdLabel.textContent = row.label || '';
    if (row.change_label) {
      const pill = document.createElement('span');
      pill.className = 'changed-pill';
      pill.textContent = row.change_label;
      tdLabel.appendChild(pill);
    }

    const main = document.createElement('div');
    if (row.value_html) {
      // value_html is generated by the survey server from controlled text only;
      // it is used to bold key variables in repeated vignette text.
      main.innerHTML = row.value_html;
    } else {
      main.textContent = row.value || '';
    }
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
  renderComprehensionCheck(item, table);

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
