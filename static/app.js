const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const ragOutput = document.getElementById('ragOutput');
const graphOutput = document.getElementById('graphOutput');
const traceSummary = document.getElementById('traceSummary');
const traceList = document.getElementById('traceList');
const querySummary = document.getElementById('querySummary');
const queryList = document.getElementById('queryList');
const ragMetaSummary = document.getElementById('ragMetaSummary');
const ragMetaList = document.getElementById('ragMetaList');
const chatInput = document.getElementById('chatInput');
const askButton = document.getElementById('askButton');

const supplierOutage = document.getElementById('supplierOutage');
const carbonTaxEnabled = document.getElementById('carbonTaxEnabled');
const carbonTaxRate = document.getElementById('carbonTaxRate');
const maxLeadTime = document.getElementById('maxLeadTime');
const llmProvider = document.getElementById('llmProvider');
const llmModel = document.getElementById('llmModel');
const llmEmbedModel = document.getElementById('llmEmbedModel');

const uploadLabel = document.querySelector('.upload-label');

uploadLabel.addEventListener('dragover', (event) => {
  event.preventDefault();
  uploadLabel.classList.add('dragging');
});

uploadLabel.addEventListener('dragleave', () => {
  uploadLabel.classList.remove('dragging');
});

uploadLabel.addEventListener('drop', (event) => {
  event.preventDefault();
  uploadLabel.classList.remove('dragging');
  if (event.dataTransfer.files.length > 0) {
    fileInput.files = event.dataTransfer.files;
    handleUpload();
  }
});

fileInput.addEventListener('change', handleUpload);
askButton.addEventListener('click', handleAsk);
chatInput.addEventListener('keypress', (event) => {
  if (event.key === 'Enter') {
    handleAsk();
  }
});

function getScenario() {
  return {
    supplierBOutage: supplierOutage.checked,
    carbonTaxEnabled: carbonTaxEnabled.checked,
    carbonTaxRate: parseFloat(carbonTaxRate.value || '0.1'),
    maxLeadTimeDays: parseInt(maxLeadTime.value || '0', 10),
  };
}

async function handleUpload() {
  if (!fileInput.files.length) {
    return;
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);
  formData.append('provider', llmProvider.value);
  formData.append('model', llmModel.value);
  formData.append('embed_model', llmEmbedModel.value);
  uploadStatus.textContent = 'Uploading and rebuilding world...';
  try {
    const response = await fetch('/data/upload', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Upload failed');
    }
    uploadStatus.textContent = `Upload complete. ${data.rows_loaded} rows loaded. LLM: ${data.llm_provider} (${data.llm_model}).`;
    ragOutput.textContent = 'Upload data and ask a question.';
    graphOutput.textContent = 'Upload data and ask a question.';
    traceSummary.textContent = 'World rebuilt. No trace yet.';
    traceList.innerHTML = '';
  } catch (error) {
    uploadStatus.textContent = `Upload failed: ${error.message}`;
  }
}

async function handleAsk() {
  const question = chatInput.value.trim();
  if (!question) {
    return;
  }
  ragOutput.textContent = '';
  graphOutput.textContent = '';
  traceSummary.textContent = '';
  traceList.innerHTML = '';
  querySummary.textContent = '';
  queryList.innerHTML = '';
  ragMetaSummary.textContent = '';
  ragMetaList.innerHTML = '';
  setLoading(ragOutput, true);
  setLoading(graphOutput, true);
  const scenario = getScenario();

  try {
    const payload = {
      question,
      scenario,
      provider: llmProvider.value,
      model: llmModel.value,
      embed_model: llmEmbedModel.value,
    };

    const ragPromise = streamEndpoint(
      '/chat/rag/stream',
      payload,
      (event) => {
        if (event.type === 'token') {
          ragOutput.textContent += event.content;
        } else if (event.type === 'status') {
          ragOutput.textContent = event.message || 'Working...';
        } else if (event.type === 'meta') {
          renderRagMeta(event.retrieval || [], event.scenario);
        } else if (event.type === 'error') {
          ragOutput.textContent = `Error: ${event.message}`;
          setLoading(ragOutput, false);
        } else if (event.type === 'done') {
          setLoading(ragOutput, false);
        }
      },
      (error) => {
        ragOutput.textContent = `Error: ${error.message}`;
        setLoading(ragOutput, false);
      }
    );

    const graphPromise = streamEndpoint(
      '/chat/graphrag/stream',
      payload,
      (event) => {
        if (event.type === 'queries') {
          renderQueryResults(event.results || []);
        } else if (event.type === 'status') {
          traceSummary.textContent = event.message || 'Working...';
        } else if (event.type === 'token') {
          graphOutput.textContent += event.content;
        } else if (event.type === 'error') {
          graphOutput.textContent = `Error: ${event.message}`;
          setLoading(graphOutput, false);
        } else if (event.type === 'done') {
          setLoading(graphOutput, false);
        }
      },
      (error) => {
        graphOutput.textContent = `Error: ${error.message}`;
        setLoading(graphOutput, false);
      }
    );

    await Promise.all([ragPromise, graphPromise]);
  } catch (error) {
    ragOutput.textContent = `Error: ${error.message}`;
    graphOutput.textContent = `Error: ${error.message}`;
    querySummary.textContent = `Error: ${error.message}`;
    queryList.innerHTML = '';
    setLoading(ragOutput, false);
    setLoading(graphOutput, false);
  }
}

function setLoading(element, isLoading) {
  if (isLoading) {
    element.classList.add('loading');
  } else {
    element.classList.remove('loading');
  }
}

function renderRagMeta(matches, scenarioText) {
  ragMetaList.innerHTML = '';
  if (scenarioText) {
    ragMetaSummary.textContent = scenarioText;
  }
  if (!matches.length) {
    ragMetaList.innerHTML = '<div class="query-empty">No retrieved documents.</div>';
    return;
  }
  ragMetaSummary.textContent = `Retrieved ${matches.length} documents.`;
  matches.forEach((match, index) => {
    const wrapper = document.createElement('div');
    wrapper.classList.add('query-block');
    const heading = document.createElement('div');
    heading.classList.add('query-heading');
    const score = typeof match.score === 'number' ? match.score.toFixed(4) : 'n/a';
    heading.textContent = `Doc ${index + 1} | score ${score} | id ${match.id}`;
    const body = document.createElement('pre');
    body.classList.add('query-rows');
    body.textContent = match.text || '';
    wrapper.appendChild(heading);
    wrapper.appendChild(body);
    ragMetaList.appendChild(wrapper);
  });
}

function renderQueryResults(results) {
  queryList.innerHTML = '';
  if (!results.length) {
    querySummary.textContent = 'No Cypher results returned.';
    queryList.innerHTML = '<div class="query-empty">No query details available.</div>';
    return;
  }
  querySummary.textContent = `Executed ${results.length} Cypher queries.`;
  results.forEach((result, index) => {
    const wrapper = document.createElement('div');
    wrapper.classList.add('query-block');
    const heading = document.createElement('div');
    heading.classList.add('query-heading');
    heading.textContent = `Query ${index + 1}: ${result.reason || 'No reason provided.'}`;
    const cypher = document.createElement('pre');
    cypher.classList.add('query-cypher');
    cypher.textContent = result.cypher || '';
    const params = document.createElement('pre');
    params.classList.add('query-params');
    params.textContent = `Params: ${JSON.stringify(result.params || {}, null, 2)}`;
    wrapper.appendChild(heading);
    wrapper.appendChild(cypher);
    wrapper.appendChild(params);

    const meta = document.createElement('div');
    meta.classList.add('query-meta');
    const rowCount = typeof result.row_count === 'number' ? result.row_count : 0;
    meta.textContent = `Rows: ${rowCount}`;
    wrapper.appendChild(meta);

    if (result.error) {
      const error = document.createElement('pre');
      error.classList.add('query-error');
      error.textContent = `Error: ${result.error}`;
      wrapper.appendChild(error);
    } else {
      const rows = document.createElement('pre');
      rows.classList.add('query-rows');
      rows.textContent = JSON.stringify(result.rows || [], null, 2);
      wrapper.appendChild(rows);
    }

    queryList.appendChild(wrapper);
  });
}

async function streamEndpoint(url, payload, onEvent, onError) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    let detail = 'Streaming failed';
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch (err) {
      // ignore
    }
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';
      for (const part of parts) {
        const lines = part.split('\n');
        const dataLines = lines
          .filter((line) => line.startsWith('data: '))
          .map((line) => line.slice(6))
          .join('\n');
        if (!dataLines) {
          continue;
        }
        try {
          const event = JSON.parse(dataLines);
          onEvent(event);
        } catch (err) {
          // ignore parsing errors
        }
      }
    }
  } catch (error) {
    onError(error);
  }
}
