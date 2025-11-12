async function loadData() {
  const sel = document.getElementById('category');
  const q = sel.value ? `?category=${encodeURIComponent(sel.value)}` : '';
  const res = await fetch(`/dashboard/data${q}`);
  const data = await res.json();
  const rows = document.getElementById('rows');
  rows.innerHTML = '';
  (data.items || []).forEach((it) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${it.expense_id}</td>
      <td>${it.category}</td>
      <td>R$ ${it.amount.toFixed(2)}</td>
      <td>${it.department}</td>
      <td>${it.date}</td>
      <td>${it.score.toFixed(3)}</td>
      <td class="risk-${it.risk}">${it.risk}</td>
      <td>${it.model || '—'}</td>
    `;
    rows.appendChild(tr);
  });
}

document.getElementById('refresh').addEventListener('click', loadData);
window.addEventListener('load', loadData);
window.addEventListener('load', loadMetrics);
function pad2(n) { return n.toString().padStart(2, '0'); }
function todayStr() {
  const d = new Date();
  return `${d.getFullYear()}-${pad2(d.getMonth()+1)}-${pad2(d.getDate())}`;
}

window.addEventListener('load', () => {
  const reqInput = document.getElementById('in_reqdate');
  const travelInput = document.getElementById('in_traveldate');
  if (reqInput && !reqInput.value) reqInput.value = todayStr();
  if (travelInput && !travelInput.value) travelInput.value = todayStr();
});

async function calculateScore() {
  const expenseId = `E${Math.floor(Math.random()*1e7)}`;
  const requestId = `R${Math.floor(Math.random()*1e7)}`;
  const payload = {
    expense_id: expenseId,
    request_id: requestId,
    requester_id: document.getElementById('in_requester').value || 'U100',
    traveller_id: document.getElementById('in_traveller').value || 'U100',
    approver_id: document.getElementById('in_approver').value || 'A10',
    request_date: document.getElementById('in_reqdate').value,
    travel_date: document.getElementById('in_traveldate').value,
    category: document.getElementById('in_category').value,
    description: document.getElementById('in_description').value || '',
    amount: parseFloat(document.getElementById('in_amount').value || '0'),
    currency: document.getElementById('in_currency').value || 'BRL',
    job_title: document.getElementById('in_jobtitle').value || 'Analista',
    department: document.getElementById('in_department').value || 'Financeiro',
    approval_status: document.getElementById('in_status').value || 'Aprovado',
  };
  const resultBox = document.getElementById('calc-result');
  const flagsBox = document.getElementById('calc-flags');
  resultBox.textContent = 'Calculando...';
  flagsBox.textContent = '';
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || `HTTP ${res.status}`);
    }
    const data = await res.json();
    resultBox.textContent = `Score: ${data.score.toFixed(3)} — Risco: ${data.risk}`;
    if (data.flags && Array.isArray(data.flags) && data.flags.length > 0) {
      flagsBox.textContent = `Flags: ${data.flags.join(', ')}`;
    } else {
      flagsBox.textContent = '';
    }
    const rows = document.getElementById('rows');
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${expenseId}</td>
      <td>${payload.category}</td>
      <td>R$ ${payload.amount.toFixed(2)}</td>
      <td>${payload.department}</td>
      <td>${payload.request_date}</td>
      <td>${data.score.toFixed(3)}</td>
      <td class="risk-${data.risk}">${data.risk}</td>
      <td>${data.model || '—'}</td>
    `;
    rows.prepend(tr);
  } catch (err) {
    resultBox.textContent = `Erro: ${err.message}`;
    flagsBox.textContent = '';
  }
}

document.getElementById('calc-btn').addEventListener('click', calculateScore);

async function loadMetrics() {
  try {
    const res = await fetch('/dashboard/metrics');
    const data = await res.json();
    const m = data.metrics || {};
    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.textContent = val ?? '—';
    };
    set('m-size', m.size);
    set('m-test-acc', fmt(m.test_acc));
    set('m-test-f1', fmt(m.test_f1));
    set('m-test-roc', fmt(m.test_roc_auc));
    set('m-test-pr', fmt(m.test_pr_auc));
  } catch (e) {
  }
}

function fmt(v) {
  if (typeof v === 'string') {
    const f = parseFloat(v);
    if (!isNaN(f)) return f.toFixed(3);
    return v;
  }
  if (typeof v === 'number') return v.toFixed(3);
  return '—';
}