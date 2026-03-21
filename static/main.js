// Photo preview for cat registration
function initPhotoPreview() {
  const input = document.getElementById('cat-photos');
  const preview = document.getElementById('photo-preview');
  const label = document.getElementById('photo-label');
  if (!input || !preview) return;

  input.addEventListener('change', function () {
    preview.innerHTML = '';
    const files = Array.from(this.files);
    if (label) label.textContent = `${files.length} תמונות נבחרו`;
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = e => {
        const img = document.createElement('img');
        img.src = e.target.result;
        preview.appendChild(img);
      };
      reader.readAsDataURL(file);
    });
  });
}

// Identify photo preview
function initIdentifyPreview() {
  const input = document.getElementById('identify-input');
  const preview = document.getElementById('identify-preview');
  if (!input || !preview) return;
  let featuresReady = true; // true = ok to submit

  input.addEventListener('change', function () {
    const originalFile = this.files[0];
    if (!originalFile) return;
    showCropModal(originalFile, function(croppedBlob) {
      featuresReady = false;
      // Show preview of original
      const reader = new FileReader();
      reader.onload = e => { preview.innerHTML = `<img src="${e.target.result}" alt="תמונה לזיהוי">`; };
      reader.readAsDataURL(originalFile);

      // Send cropped to feature extraction
      const featForm = new FormData();
      featForm.append('photo', croppedBlob, 'cropped.jpg');
      fetch('/api/extract-features', {method: 'POST', body: featForm})
        .then(r => r.json())
        .then(data => {
          let tokenInput = document.getElementById('identify-features-token');
          if (!tokenInput) {
            tokenInput = document.createElement('input');
            tokenInput.type = 'hidden';
            tokenInput.id = 'identify-features-token';
            tokenInput.name = 'features_token';
            document.getElementById('identify-form').appendChild(tokenInput);
          }
          tokenInput.value = data.token || '';
        })
        .catch(() => {})
        .finally(() => { featuresReady = true; });
    });
  });

  // Block form submit until features are ready
  document.getElementById('identify-form').addEventListener('submit', function(e) {
    if (!featuresReady) {
      e.preventDefault();
      const btn = document.getElementById('identify-btn');
      if (btn) { btn.textContent = '⏳ מכין תמונה...'; btn.disabled = true; }
      const interval = setInterval(() => {
        if (featuresReady) {
          clearInterval(interval);
          document.getElementById('identify-form').requestSubmit();
        }
      }, 200);
    }
  }, true); // capture phase — runs before initIdentifyForm's submit handler
}

// Animate progress steps during identification
function runProgressSteps() {
  const steps = [
    { id: 'step-1', duration: 600 },
    { id: 'step-2', duration: 1800 },
    { id: 'step-3', duration: 1800 },
    { id: 'step-4', duration: 800 },
  ];
  const bar = document.getElementById('progress-bar-fill');
  const total = steps.reduce((s, x) => s + x.duration, 0);
  let elapsed = 0;

  steps.forEach((step, i) => {
    // activate step
    setTimeout(() => {
      document.getElementById(step.id)?.classList.add('active');
    }, elapsed);

    elapsed += step.duration;

    // mark step done
    setTimeout(() => {
      document.getElementById(step.id)?.classList.add('done');
      if (bar) bar.style.width = `${Math.round(((i + 1) / steps.length) * 100)}%`;
    }, elapsed);
  });
}

// Show loading steps on identify form submit
function initIdentifyForm() {
  const form = document.getElementById('identify-form');
  const loading = document.getElementById('loading-indicator');
  const formContent = form?.parentElement;
  const btn = document.getElementById('identify-btn');
  if (!form || !loading) return;

  form.addEventListener('submit', function () {
    form.style.display = 'none';
    loading.classList.add('active');
    if (btn) btn.disabled = true;
    runProgressSteps();
  });
}

// Toggle add-photo section per cat card
function showAddPhoto(catId) {
  const section = document.getElementById(`add-photo-${catId}`);
  if (!section) return;
  section.style.display = section.style.display === 'none' ? 'block' : 'none';
}

// Upload photo — handled inline in cats.html (uses feature extraction + Cloudinary)

// ── Crop modal ────────────────────────────────────────────────
let _cropperInstance = null;
let _cropCallback = null;
let _cropOriginalFile = null;

function showCropModal(file, callback) {
  _cropCallback = callback;
  _cropOriginalFile = file;
  const modal = document.getElementById('crop-modal');
  const img = document.getElementById('crop-image');
  if (_cropperInstance) { _cropperInstance.destroy(); _cropperInstance = null; }
  img.src = URL.createObjectURL(file);
  modal.style.display = 'flex';
  img.onload = function() {
    _cropperInstance = new Cropper(img, {
      viewMode: 1, autoCropArea: 0.75, movable: true, zoomable: false
    });
  };
}

function confirmCrop() {
  if (!_cropperInstance || !_cropCallback) return;
  _cropperInstance.getCroppedCanvas({maxWidth: 900, maxHeight: 900})
    .toBlob(function(blob) {
      document.getElementById('crop-modal').style.display = 'none';
      _cropperInstance.destroy(); _cropperInstance = null;
      const cb = _cropCallback; _cropCallback = null; _cropOriginalFile = null;
      cb(blob);
    }, 'image/jpeg', 0.92);
}

function skipCrop() {
  document.getElementById('crop-modal').style.display = 'none';
  if (_cropperInstance) { _cropperInstance.destroy(); _cropperInstance = null; }
  const cb = _cropCallback; const orig = _cropOriginalFile;
  _cropCallback = null; _cropOriginalFile = null;
  if (cb) cb(orig);
}

// Show a dismissible notice when uploaded photo resembles a friend's cat
function showSimilarNotice(catName, ownerName) {
  const notice = document.createElement('div');
  notice.className = 'similar-notice';
  notice.innerHTML = `
    <strong>שים לב!</strong> התמונה שהעלית דומה לחתול <strong>${catName}</strong> של המשתמש <strong>${ownerName}</strong>.
    <button onclick="this.parentElement.remove()">✕</button>
  `;
  document.body.prepend(notice);
}

// Delete confirmation
function confirmDelete(catName) {
  return confirm(`האם למחוק את החתול "${catName}"?`);
}

// ── Real-time nav badge polling ───────────────────────────────
function updateNavBadge(selector, count, el) {
  const link = el || document.querySelector(selector);
  if (!link) return;
  let badge = link.querySelector('.nav-badge');
  if (count > 0) {
    if (!badge) { badge = document.createElement('span'); badge.className = 'nav-badge'; link.appendChild(badge); }
    badge.textContent = count;
  } else if (badge) {
    badge.remove();
  }
}

async function pollNavCounts() {
  try {
    const res = await fetch('/api/nav-counts');
    if (!res.ok) return;
    const d = await res.json();
    // nav has two /friends links: first=friends(😽), second=chat(💬)
    const friendsLinks = document.querySelectorAll('nav a[href*="/friends"]');
    if (friendsLinks[0]) updateNavBadge(null, d.pending_requests, friendsLinks[0]);
    if (friendsLinks[1]) updateNavBadge(null, d.unread_messages, friendsLinks[1]);
    updateNavBadge('nav a[href*="/notifications"]', d.unread_notifs);
  } catch(e) {}
}

document.addEventListener('DOMContentLoaded', () => {
  initPhotoPreview();
  initIdentifyPreview();
  initIdentifyForm();

  // Poll every 20 seconds if logged in (nav links exist)
  if (document.querySelector('nav .nav-links')) {
    setInterval(pollNavCounts, 20000);
  }

  const similar = sessionStorage.getItem('similar_notice');
  if (similar) {
    sessionStorage.removeItem('similar_notice');
    const { cat_name, owner } = JSON.parse(similar);
    showSimilarNotice(cat_name, owner);
  }
});
