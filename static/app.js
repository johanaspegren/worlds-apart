const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const ragOutput = document.getElementById('ragOutput');
const graphOutput = document.getElementById('graphOutput');
const traceSummary = document.getElementById('traceSummary');
const traceList = document.getElementById('traceList');
const querySummary = document.getElementById('querySummary');
const queryList = document.getElementById('queryList');
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
  ragOutput.textContent = 'Thinking...';
  graphOutput.textContent = 'Thinking...';
  traceSummary.textContent = '';
  traceList.innerHTML = '';
  querySummary.textContent = '';
  queryList.innerHTML = '';
  const scenario = getScenario();

  try {
    const [ragResponse, graphResponse] = await Promise.all([
      fetch('/chat/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          scenario,
          provider: llmProvider.value,
          model: llmModel.value,
          embed_model: llmEmbedModel.value,
        }),
      }),
      fetch('/chat/graphrag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          scenario,
          provider: llmProvider.value,
          model: llmModel.value,
          embed_model: llmEmbedModel.value,
        }),
      }),
    ]);

    const ragData = await ragResponse.json();
    const graphData = await graphResponse.json();

    if (!ragResponse.ok) {
      throw new Error(ragData.detail || 'RAG failed');
    }
    if (!graphResponse.ok) {
      throw new Error(graphData.detail || 'GraphRAG failed');
    }

    ragOutput.textContent = ragData.answer;
    graphOutput.textContent = graphData.answer;
    traceSummary.textContent = graphData.trace_summary || 'Trace updated.';
    if (graphData.trace && graphData.trace.length) {
      graphData.trace.forEach((trace) => {
        const item = document.createElement('li');
        if (trace.summary) {
          item.textContent = trace.summary;
        } else {
          item.textContent = `${trace.rel_type || 'REL'} ${trace.source_span || ''}`.trim();
        }
        traceList.appendChild(item);
      });
    } else {
      traceList.innerHTML = '<li>No graph relationships returned.</li>';
    }

    if (graphData.queries && graphData.queries.length) {
      querySummary.textContent = `Executed ${graphData.queries.length} Cypher queries.`;
      graphData.queries.forEach((query, index) => {
        const wrapper = document.createElement('div');
        wrapper.classList.add('query-block');
        const heading = document.createElement('div');
        heading.classList.add('query-heading');
        heading.textContent = `Query ${index + 1}: ${query.reason || 'No reason provided.'}`;
        const cypher = document.createElement('pre');
        cypher.classList.add('query-cypher');
        cypher.textContent = query.cypher || '';
        const params = document.createElement('pre');
        params.classList.add('query-params');
        params.textContent = `Params: ${JSON.stringify(query.params || {}, null, 2)}`;
        wrapper.appendChild(heading);
        wrapper.appendChild(cypher);
        wrapper.appendChild(params);
        const result = (graphData.results || []).find((entry) => entry.cypher === query.cypher);
        if (result) {
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
        }
        queryList.appendChild(wrapper);
      });
    } else {
      querySummary.textContent = 'No Cypher queries returned.';
      queryList.innerHTML = '<div class="query-empty">No query details available.</div>';
    }
  } catch (error) {
    ragOutput.textContent = `Error: ${error.message}`;
    graphOutput.textContent = `Error: ${error.message}`;
    querySummary.textContent = `Error: ${error.message}`;
    queryList.innerHTML = '';
  }
}
