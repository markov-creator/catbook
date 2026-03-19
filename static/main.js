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

  input.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;
    showCropModal(file, function(croppedBlob) {
      // Replace file input with cropped blob so the form submits the cropped image
      try {
        const dt = new DataTransfer();
        dt.items.add(new File([croppedBlob], 'cropped.jpg', {type: 'image/jpeg'}));
        input.files = dt.files;
      } catch(e) {
        // DataTransfer not supported — fall back to original file
      }
      const reader = new FileReader();
      reader.onload = e => {
        preview.innerHTML = `<img src="${e.target.result}" alt="תמונה לזיהוי">`;
      };
      reader.readAsDataURL(croppedBlob);
    });
  });
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

document.addEventListener('DOMContentLoaded', () => {
  initPhotoPreview();
  initIdentifyPreview();
  initIdentifyForm();

  const similar = sessionStorage.getItem('similar_notice');
  if (similar) {
    sessionStorage.removeItem('similar_notice');
    const { cat_name, owner } = JSON.parse(similar);
    showSimilarNotice(cat_name, owner);
  }
});
