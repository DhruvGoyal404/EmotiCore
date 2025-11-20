// API Base URL
const API_BASE = 'http://localhost:8000';

// DOM Elements
const textSection = document.getElementById('textSection');
const imageSection = document.getElementById('imageSection');
const textBtn = document.getElementById('textBtn');
const imageBtn = document.getElementById('imageBtn');
const results = document.getElementById('results');
const loading = document.getElementById('loading');
const errorSection = document.getElementById('error');

// Model display names
const TEXT_MODEL_NAMES = {
    'basic': 'Basic (Original TF-IDF + LogReg)',
    'logreg': 'Logistic Regression',
    'svm': 'SVM (Support Vector Machine)',
    'xgboost': 'XGBoost',
    'nb': 'Naive Bayes',
    'huggingface': 'HuggingFace (DistilRoBERTa) - Best'
};

const FACIAL_MODEL_NAMES = {
    'custom': 'Custom CNN (97.43%) ★ BEST MODEL',
    'opencv': 'OpenCV (Haar Cascade + CNN backend)',
    'rf': 'Random Forest (29.47%)',
    'xgboost': 'XGBoost (29.41%)',
    'logreg': 'Logistic Regression (24.05%)'
};

// Check API status on load
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    loadAvailableModels();
});

// Check if API is running
async function checkAPIStatus() {
    const statusEl = document.getElementById('apiStatus');
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        const textCount = data.text_models_loaded.length;
        const facialCount = data.facial_models_loaded.length;

        statusEl.textContent = `API Online | Text Models: ${textCount} | Facial Models: ${facialCount}`;
        statusEl.parentElement.className = 'api-status online';
    } catch (error) {
        statusEl.textContent = 'API Offline - Make sure server is running (uvicorn api.main:app --reload)';
        statusEl.parentElement.className = 'api-status offline';
    }
}

// Load available models from API
async function loadAvailableModels() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        const data = await response.json();

        // Update text model dropdown
        const textModelSelect = document.getElementById('textModel');
        textModelSelect.innerHTML = '';

        // Sort to put huggingface first (best), then others
        const textModelOrder = ['huggingface', 'logreg', 'svm', 'xgboost', 'nb', 'basic'];
        const sortedTextModels = data.text_models.sort((a, b) => {
            return textModelOrder.indexOf(a) - textModelOrder.indexOf(b);
        });

        sortedTextModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = TEXT_MODEL_NAMES[model] || model;
            textModelSelect.appendChild(option);
        });

        // Update facial model dropdown
        const imageModelSelect = document.getElementById('imageModel');
        imageModelSelect.innerHTML = '';

        // Sort facial models
        const facialModelOrder = ['deepface', 'opencv', 'custom', 'rf', 'xgboost', 'logreg'];
        const sortedFacialModels = data.facial_models.sort((a, b) => {
            return facialModelOrder.indexOf(a) - facialModelOrder.indexOf(b);
        });

        sortedFacialModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = FACIAL_MODEL_NAMES[model] || model;
            imageModelSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Switch between text and image input
function selectInputType(type) {
    if (type === 'text') {
        textSection.classList.remove('hidden');
        imageSection.classList.add('hidden');
        textBtn.classList.add('active');
        imageBtn.classList.remove('active');
    } else {
        textSection.classList.add('hidden');
        imageSection.classList.remove('hidden');
        textBtn.classList.remove('active');
        imageBtn.classList.add('active');
    }

    // Hide previous results
    results.classList.add('hidden');
    errorSection.classList.add('hidden');
}

// Preview uploaded image
function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('imagePreview');
            preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    }
}

// Show loading state
function showLoading() {
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    errorSection.classList.add('hidden');
}

// Hide loading state
function hideLoading() {
    loading.classList.add('hidden');
}

// Show error message
function showError(message) {
    errorSection.classList.remove('hidden');
    document.getElementById('errorMessage').textContent = message;
}

// Display results
function displayResults(data) {
    results.classList.remove('hidden');

    // Update emotion display
    document.getElementById('emotionEmoji').textContent = data.emoji;
    document.getElementById('emotionLabel').textContent = data.emotion;

    // Update confidence
    const confidence = Math.round(data.confidence * 100);
    document.getElementById('confidenceValue').textContent = `${confidence}%`;
    document.getElementById('confidenceBar').style.width = `${confidence}%`;

    // Update model used - show friendly name
    const modelUsed = data.model_used;
    const friendlyName = TEXT_MODEL_NAMES[modelUsed] || FACIAL_MODEL_NAMES[modelUsed] || modelUsed;
    document.getElementById('modelUsed').textContent = friendlyName;

    // Update probability bars
    const probContainer = document.getElementById('probabilityBars');
    probContainer.innerHTML = '';

    // Sort probabilities
    const sorted = Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1]);

    sorted.forEach(([emotion, prob]) => {
        const percentage = Math.round(prob * 100);
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <span class="prob-label">${emotion}</span>
            <div class="prob-bar-container">
                <div class="prob-bar emotion-${emotion.toLowerCase()}" style="width: ${percentage}%"></div>
            </div>
            <span class="prob-value">${percentage}%</span>
        `;
        probContainer.appendChild(item);
    });
}

// Analyze text
async function analyzeText() {
    const text = document.getElementById('textInput').value.trim();
    const model = document.getElementById('textModel').value;

    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }

    showLoading();

    try {
        const response = await fetch(`${API_BASE}/api/text-emotion?model=${model}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text, model })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to analyze text');
        }

        const data = await response.json();
        hideLoading();
        displayResults(data);

    } catch (error) {
        hideLoading();
        showError(error.message);
    }
}

// Analyze image
async function analyzeImage() {
    const fileInput = document.getElementById('imageInput');
    const model = document.getElementById('imageModel').value;

    if (!fileInput.files || !fileInput.files[0]) {
        showError('Please select an image to analyze');
        return;
    }

    showLoading();

    try {
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        const response = await fetch(`${API_BASE}/api/facial-emotion?model=${model}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to analyze image');
        }

        const data = await response.json();
        hideLoading();
        displayResults(data);

    } catch (error) {
        hideLoading();
        showError(error.message);
    }
}
