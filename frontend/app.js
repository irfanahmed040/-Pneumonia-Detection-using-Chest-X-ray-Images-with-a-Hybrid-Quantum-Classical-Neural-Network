const form = document.getElementById('predict-form');
const input = document.getElementById('image-input');
const result = document.getElementById('result');
const preview = document.getElementById('preview');
const submitBtn = document.getElementById('submit-btn');
const clearBtn = document.getElementById('clear-btn');

let currentPreviewUrl = null;

const clearSelection = () => {
  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = null;
  }

  input.value = '';
  preview.removeAttribute('src');
  preview.hidden = true;
  result.hidden = true;
  result.textContent = '';
  clearBtn.disabled = true;
};

input.addEventListener('change', () => {
  const file = input.files?.[0];
  if (!file) {
    clearSelection();
    return;
  }

  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
  }

  currentPreviewUrl = URL.createObjectURL(file);
  preview.src = currentPreviewUrl;
  preview.hidden = false;
  clearBtn.disabled = false;
  result.hidden = true;
  result.textContent = '';
});

clearBtn.addEventListener('click', () => {
  clearSelection();
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = input.files?.[0];
  if (!file) {
    result.hidden = false;
    result.textContent = 'Please select an image first.';
    return;
  }

  const fd = new FormData();
  fd.append('file', file);

  submitBtn.disabled = true;
  submitBtn.textContent = 'Analyzing...';
  result.hidden = false;
  result.textContent = 'Running inference...';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: fd,
    });

    const payload = await res.json();

    if (!res.ok) {
      result.innerHTML = `<strong>Request failed:</strong> ${payload.detail || 'Unknown error'}`;
      return;
    }

    const pred = payload.prediction;
    const conf = (payload.confidence * 100).toFixed(2);
    const normal = (payload.probabilities.NORMAL * 100).toFixed(2);
    const pna = (payload.probabilities.PNEUMONIA * 100).toFixed(2);
    const alpha = (payload.probabilities.adaptive_quantum_weight * 100).toFixed(2);

    result.innerHTML = `
      <strong>Prediction: ${pred}</strong><br>
      Confidence: ${conf}%<br>
      NORMAL: ${normal}%<br>
      PNEUMONIA: ${pna}%<br>
      Adaptive quantum contribution: ${alpha}%
    `;
  } catch (err) {
    result.innerHTML = `<strong>Error:</strong> ${err.message}`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Analyze Image';
  }
});

  const revealNodes = document.querySelectorAll('.reveal');
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('visible');
        io.unobserve(entry.target);
      });
    },
    {
      threshold: 0.2,
    }
  );

  revealNodes.forEach((node) => io.observe(node));
