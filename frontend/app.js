const question = document.querySelector('#question');
const askButton = document.querySelector('#ask-button');
const status = document.querySelector('#status');
const answer = document.querySelector('#answer');
const answerMode = document.querySelector('#answer-mode');
const answerSource = document.querySelector('#answer-source');
const answerCount = document.querySelector('#answer-count');
const evidenceList = document.querySelector('#evidence-list');
const sourceName = document.querySelector('#source-name');
const fileInput = document.querySelector('#file-input');

function setStatus(message, isError = false) {
  status.textContent = message;
  status.style.color = isError ? '#a42a20' : '';
}

function showEvidence(items) {
  evidenceList.replaceChildren();
  if (!items.length) {
    evidenceList.innerHTML = '<p class="empty">No relevant passages were found.</p>';
    return;
  }
  items.forEach((item) => {
    const card = document.createElement('article');
    card.className = 'evidence-card';
    const pages = item.pages.length === 1 ? `Page ${item.pages[0]}` : `Pages ${item.pages.join(', ')}`;
    card.innerHTML = `<header><span>${pages}</span><span class="source-name">${item.source || 'Report'}</span></header><p>${item.excerpt}</p>`;
    evidenceList.append(card);
  });
}

async function ask() {
  const value = question.value.trim();
  if (!value) { setStatus('Write a question first.', true); return; }
  askButton.disabled = true;
  setStatus('Finding the most relevant passages…');
  try {
    const response = await fetch('/api/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: value }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Search failed.');
    answer.textContent = data.answer;
    answerMode.textContent = data.mode === 'grounded generation' ? 'Based on sources' : 'Source preview';
    answerSource.textContent = data.source;
    answerCount.textContent = `${data.evidence.length} supporting passage${data.evidence.length === 1 ? '' : 's'}`;
    showEvidence(data.evidence);
    setStatus('Answer and sources ready.');
  } catch (error) { setStatus(error.message, true); }
  finally { askButton.disabled = false; }
}

async function upload(file) {
  if (!file) return;
  askButton.disabled = true;
  setStatus(`Preparing ${file.name}…`);
  try {
    const response = await fetch('/api/upload', { method: 'POST', headers: { 'Content-Type': 'application/pdf', 'X-Filename': encodeURIComponent(file.name) }, body: file });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Upload failed.');
    sourceName.textContent = `${data.source} · ${data.pages} pages · ${data.chunks} passages`;
    answer.textContent = 'Your document is ready. Ask a question to find the relevant passages.';
    answerMode.textContent = 'Ready';
    answerSource.textContent = data.source;
    answerCount.textContent = `${data.chunks} passages available`;
    showEvidence([]);
    setStatus('PDF is ready to search.');
  } catch (error) { setStatus(error.message, true); }
  finally { askButton.disabled = false; fileInput.value = ''; }
}

askButton.addEventListener('click', ask);
question.addEventListener('keydown', (event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') ask(); });
fileInput.addEventListener('change', () => upload(fileInput.files[0]));

fetch('/api/suggestions').then((response) => response.json()).then((data) => {
  sourceName.textContent = data.source;
  const list = document.querySelector('#suggestion-list');
  data.questions.forEach((text) => {
    const button = document.createElement('button');
    button.className = 'suggestion'; button.textContent = text;
    button.addEventListener('click', () => { question.value = text; question.focus(); });
    list.append(button);
  });
}).catch(() => setStatus('Unable to load suggested questions.', true));
