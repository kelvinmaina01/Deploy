/**
 * TuneKit Landing
 */

const form = document.getElementById('apiKeyForm');
const status = document.getElementById('formStatus');

// Load saved keys
function loadKeys() {
    const keys = {
        openaiKey: localStorage.getItem('tunekit_openai_key'),
        modalKey: localStorage.getItem('tunekit_modal_key'),
        modalSecret: localStorage.getItem('tunekit_modal_secret')
    };
    
    Object.entries(keys).forEach(([id, value]) => {
        const input = document.getElementById(id);
        if (input && value) input.value = value;
    });
}

loadKeys();

// Form submit
form.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const openaiKey = document.getElementById('openaiKey').value.trim();
    const modalKey = document.getElementById('modalKey').value.trim();
    const modalSecret = document.getElementById('modalSecret').value.trim();
    
    status.className = 'form-status';
    
    // Basic validation
    if (openaiKey && !openaiKey.startsWith('sk-')) {
        showStatus('OpenAI key should start with "sk-"', 'error');
        return;
    }
    
    if (modalKey && !modalKey.startsWith('ak-')) {
        showStatus('Modal Token ID should start with "ak-"', 'error');
        return;
    }
    
    if (modalSecret && !modalSecret.startsWith('as-')) {
        showStatus('Modal Secret should start with "as-"', 'error');
        return;
    }
    
    // Save
    if (openaiKey) localStorage.setItem('tunekit_openai_key', openaiKey);
    if (modalKey) localStorage.setItem('tunekit_modal_key', modalKey);
    if (modalSecret) localStorage.setItem('tunekit_modal_secret', modalSecret);
    
    showStatus('Saved! Redirecting...', 'success');
    
    setTimeout(() => {
        window.location.href = '/dashboard';
    }, 500);
});

function showStatus(msg, type) {
    status.textContent = msg;
    status.className = `form-status ${type}`;
}
