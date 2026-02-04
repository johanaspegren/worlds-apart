const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const ragOutput = document.getElementById('ragOutput');
const graphOutput = document.getElementById('graphOutput');
const traceSummary = document.getElementById('traceSummary');
const traceList = document.getElementById('traceList');
const querySummary = document.getElementById('querySummary');
const queryList = document.getElementById('queryList');
const verifyQuery = document.getElementById('verifyQuery');
const ragMetaSummary = document.getElementById('ragMetaSummary');
const ragMetaList = document.getElementById('ragMetaList');
const chatInput = document.getElementById('chatInput');
const askButton = document.getElementById('askButton');
const graphView = document.getElementById('graphView');
const domainSelect = document.getElementById('domainSelect');
const faqTitle = document.querySelector('.faq-title');
const faqButtonsContainer = document.querySelector('.faq-buttons');

const llmProvider = document.getElementById('llmProvider');
const llmModel = document.getElementById('llmModel');
const llmEmbedModel = document.getElementById('llmEmbedModel');
const DOMAIN_UI = {
  supplychain: {
    label: 'Supply Chain Assistant',
    placeholder: 'Ask a question about the supply chain...',
    faqs: [
      {
        label: 'Supplier B outage impact',
        question: 'What happens if Supplier B is offline for two weeks?',
      },
      {
        label: 'Carbon tax sensitivity',
        question: 'How would a $0.10/kg CO₂ carbon tax change total cost by product?',
      },
      {
        label: 'Lead time cap risk',
        question: 'Which products break first if max lead time is capped at 10 days?',
      },
      {
        label: 'Vietnam disruption',
        question: 'Why does a disruption of supply in Vietnam affect Product Gamma but not Product Delta? Consider the BOM dependencies!',
      },
    ],
  },
  fraudfinder: {
    label: 'Fraud Finder',
    placeholder: 'Ask a question about fraud risk...',
    faqs: [
      {
        label: 'Shared device risk',
        question: 'Which accounts share a device with Account A17?',
      },
      {
        label: 'Ring detection',
        question: 'Show accounts connected through the same merchant and device.',
      },
      {
        label: 'IP clustering',
        question: 'Which accounts log in from the same IP as flagged accounts?',
      },
      {
        label: 'Account explanation',
        question: 'Why was Account A17 flagged as high risk?',
      },
    ],
  },
  drhouse: {
    label: 'Dr House',
    placeholder: 'Ask a question about patient patterns...',
    faqs: [
      {
        label: 'Symptom clusters',
        question: 'Which symptoms co-occur most often for Patient P12?',
      },
      {
        label: 'Diagnosis rationale',
        question: 'What evidence supports the diagnosis for Patient P12?',
      },
      {
        label: 'Medication patterns',
        question: 'Which medications are most common for this diagnosis?',
      },
      {
        label: 'Lab anomalies',
        question: 'Show abnormal lab patterns for Patient P12.',
      },
    ],
  },
};

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

function currentDomain() {
  const value = domainSelect?.value || 'supplychain';
  return DOMAIN_UI[value] ? value : 'supplychain';
}

function renderFaqs(domainKey) {
  const config = DOMAIN_UI[domainKey] || DOMAIN_UI.supplychain;
  if (faqTitle) {
    faqTitle.textContent = `${config.label} FAQs`;
  }
  if (!faqButtonsContainer) {
    return;
  }
  faqButtonsContainer.innerHTML = '';
  config.faqs.forEach((item) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.classList.add('faq-button');
    button.dataset.question = item.question;
    button.textContent = item.label;
    button.addEventListener('click', () => {
      chatInput.value = item.question;
      chatInput.focus();
    });
    faqButtonsContainer.appendChild(button);
  });
}

function applyDomain(domainKey) {
  const config = DOMAIN_UI[domainKey] || DOMAIN_UI.supplychain;
  if (chatInput) {
    chatInput.placeholder = config.placeholder;
  }
  renderFaqs(domainKey);
}

if (domainSelect) {
  const storedDomain = localStorage.getItem('wa_domain');
  if (storedDomain && DOMAIN_UI[storedDomain]) {
    domainSelect.value = storedDomain;
  }
  applyDomain(currentDomain());
  domainSelect.addEventListener('change', () => {
    const value = currentDomain();
    localStorage.setItem('wa_domain', value);
    applyDomain(value);
  });
} else {
  applyDomain('supplychain');
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
  formData.append('domain', currentDomain());
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
  if (verifyQuery) {
    verifyQuery.innerHTML = '';
  }
  ragMetaSummary.textContent = '';
  ragMetaList.innerHTML = '';
  renderGraph({ nodes: [], edges: [] });
  setLoading(ragOutput, true);
  setLoading(graphOutput, true);
  try {
    const payload = {
      question,
      scenario: {},
      provider: llmProvider.value,
      model: llmModel.value,
      embed_model: llmEmbedModel.value,
      domain: currentDomain(),
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
          renderQueryResults(event.results || [], event.graph);
        } else if (event.type === 'status') {
          traceSummary.textContent = event.message || 'Working...';
        } else if (event.type === 'token') {
          graphOutput.textContent += event.content;
        } else if (event.type === 'verify') {
          renderVerification(event.verification);
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

function renderQueryResults(results, graph) {
  queryList.innerHTML = '';
  if (verifyQuery) {
    verifyQuery.innerHTML = '';
  }
  if (!results.length) {
    querySummary.textContent = 'No Cypher results returned.';
    queryList.innerHTML = '<div class="query-empty">No query details available.</div>';
    renderGraph(graph || { nodes: [], edges: [] });
    return;
  }
  querySummary.textContent = `Executed ${results.length} Cypher queries.`;
  renderGraph(graph || { nodes: [], edges: [] });
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

function renderVerification(verification) {
  if (!verifyQuery) {
    return;
  }
  verifyQuery.innerHTML = '';
  if (!verification || !verification.result) {
    verifyQuery.innerHTML = '<div class="query-empty">No verification query generated.</div>';
    return;
  }
  const block = document.createElement('div');
  block.classList.add('verify-block');
  const heading = document.createElement('div');
  heading.classList.add('query-heading');
  heading.textContent = 'Verification query';
  const cypher = document.createElement('pre');
  cypher.classList.add('query-cypher');
  cypher.textContent = verification.result.cypher || '';
  const params = document.createElement('pre');
  params.classList.add('query-params');
  params.textContent = `Params: ${JSON.stringify(verification.result.params || {}, null, 2)}`;
  const meta = document.createElement('div');
  meta.classList.add('query-meta');
  if (verification.result.error) {
    meta.textContent = `Error: ${verification.result.error}`;
  } else {
    const rowCount = typeof verification.result.row_count === 'number' ? verification.result.row_count : 0;
    meta.textContent = `Rows: ${rowCount}`;
  }
  block.appendChild(heading);
  block.appendChild(cypher);
  block.appendChild(params);
  block.appendChild(meta);
  verifyQuery.appendChild(block);
  renderGraph(verification.graph || { nodes: [], edges: [] });
}

function renderGraph(graph) {
  if (!graphView) {
    return;
  }
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  graphView.innerHTML = '';
  if (!nodes.length) {
    graphView.innerHTML = '<div class="graph-empty">No graph data returned yet.</div>';
    return;
  }

  const width = Math.max(graphView.clientWidth, 320);
  const height = Math.max(graphView.clientHeight, 240);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.classList.add('graph-svg');

  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) / 2 - 40;
  const nodePositions = new Map();
  nodes.forEach((node, index) => {
    const angle = (index / nodes.length) * Math.PI * 2;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    nodePositions.set(node.id, { x, y, node });
  });

  edges.forEach((edge) => {
    const source = nodePositions.get(edge.source);
    const target = nodePositions.get(edge.target);
    if (!source || !target) {
      return;
    }
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', source.x);
    line.setAttribute('y1', source.y);
    line.setAttribute('x2', target.x);
    line.setAttribute('y2', target.y);
    line.setAttribute('stroke', '#98aee0');
    line.setAttribute('stroke-width', '1.6');
    line.setAttribute('opacity', '0.8');
    svg.appendChild(line);

    const midX = (source.x + target.x) / 2;
    const midY = (source.y + target.y) / 2;
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', midX);
    label.setAttribute('y', midY);
    label.setAttribute('fill', '#52606d');
    label.setAttribute('font-size', '10');
    label.setAttribute('text-anchor', 'middle');
    label.textContent = edge.type;
    svg.appendChild(label);
  });

  nodes.forEach((node) => {
    const pos = nodePositions.get(node.id);
    if (!pos) {
      return;
    }
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', pos.x);
    circle.setAttribute('cy', pos.y);
    circle.setAttribute('r', '16');
    circle.setAttribute('fill', nodeColor(node.label));
    circle.setAttribute('stroke', '#102a43');
    circle.setAttribute('stroke-width', '1');
    svg.appendChild(circle);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', pos.x);
    label.setAttribute('y', pos.y + 30);
    label.setAttribute('fill', '#102a43');
    label.setAttribute('font-size', '10');
    label.setAttribute('text-anchor', 'middle');
    label.textContent = node.name || node.id;
    svg.appendChild(label);
  });

  graphView.appendChild(svg);
}

function nodeColor(label) {
  switch ((label || '').toLowerCase()) {
    case 'supplier':
      return '#f8b4b4';
    case 'component':
      return '#c4b5fd';
    case 'product':
      return '#fcd34d';
    case 'factory':
      return '#93c5fd';
    case 'port':
      return '#86efac';
    case 'country':
      return '#fca5a5';
    default:
      return '#cbd5e1';
  }
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
