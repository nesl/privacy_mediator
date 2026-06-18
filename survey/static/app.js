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
    const requestedK = parseInt(document.getElementById('k-input').value || '25', 10);
    const data = await postJSON('/api/start', {participant_code: participantCode, k: requestedK});
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
  submitBtn.disabled = true;
  try {
    const elapsed = itemStartedAt ? Date.now() - itemStartedAt : null;
    await postJSON(`/api/session/${sessionId}/submit`, {
      index: currentIndex,
      rating: rating.value,
      confidence: document.getElementById('confidence').value,
      free_text: document.getElementById('free-text').value,
      elapsed_ms: elapsed
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

function renderItem(item) {
  document.getElementById('progress-text').textContent = `Question ${item.index + 1} of ${item.total}`;
  document.getElementById('progress-fill').style.width = `${((item.index + 1) / item.total) * 100}%`;
  document.getElementById('task-title').textContent = item.flow.task_label || item.flow.task || 'Smart-space scenario';
  document.getElementById('vignette').textContent = item.vignette;

  const table = document.getElementById('ci-table');
  table.innerHTML = '';
  for (const row of item.display_fields) {
    const tr = document.createElement('tr');
    const tdLabel = document.createElement('td');
    const tdValue = document.createElement('td');
    tdLabel.textContent = row.label;
    const main = document.createElement('div');
    main.textContent = row.value;
    tdValue.appendChild(main);
    if (row.help) {
      const help = document.createElement('div');
      help.className = 'field-help';
      help.textContent = row.example ? `${row.help} Example: ${row.example}` : row.help;
      tdValue.appendChild(help);
    }
    tr.appendChild(tdLabel);
    tr.appendChild(tdValue);
    table.appendChild(tr);
  }

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
