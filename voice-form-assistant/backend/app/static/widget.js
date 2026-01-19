/**
 * Voice Form Assistant Widget
 * Main widget for voice-based form filling
 */

(function() {
    'use strict';

    const CONFIG = {
        wsUrl: 'ws://localhost:8000/ws',
        staticUrl: 'http://localhost:8000/static',
        sampleRate: 16000,
    };

    /**
     * Form Analyzer - Scans DOM for forms and extracts schema
     */
    class FormAnalyzer {
        constructor() {
            this.forms = [];
        }

        analyzePage() {
            const forms = document.querySelectorAll('form');
            const schema = {
                name: document.title || 'Form',
                fields: []
            };

            forms.forEach(form => {
                const fields = this.analyzeForm(form);
                schema.fields.push(...fields);
            });

            // Also check for standalone inputs
            const standaloneInputs = document.querySelectorAll('input:not(form input), select:not(form select), textarea:not(form textarea)');
            standaloneInputs.forEach(input => {
                const field = this.analyzeField(input);
                if (field) {
                    schema.fields.push(field);
                }
            });

            console.log('[VoiceAssistant] Form schema:', schema);
            return schema;
        }

        analyzeForm(form) {
            const fields = [];
            const inputs = form.querySelectorAll('input, select, textarea');

            inputs.forEach(input => {
                const field = this.analyzeField(input);
                if (field) {
                    fields.push(field);
                }
            });

            return fields;
        }

        analyzeField(element) {
            const type = element.type || element.tagName.toLowerCase();

            if (['hidden', 'submit', 'button', 'reset', 'image'].includes(type)) {
                return null;
            }

            const name = element.name || element.id;
            if (!name) return null;

            let label = '';
            const labelEl = document.querySelector(`label[for="${element.id}"]`);
            if (labelEl) {
                label = labelEl.textContent.trim();
            } else {
                const parentLabel = element.closest('label');
                if (parentLabel) {
                    label = parentLabel.textContent.replace(element.value || '', '').trim();
                }
            }

            label = label.replace(/[*:]/g, '').trim() || name;

            const field = {
                id: element.id || name,
                name: name,
                type: type,
                label: label,
                required: element.required || element.hasAttribute('required'),
                pattern: element.pattern || null,
                maxLength: element.maxLength > 0 ? element.maxLength : null,
                placeholder: element.placeholder || null,
                field_type: this.detectFieldType({ name, label, type, pattern: element.pattern })
            };

            if (element.tagName === 'SELECT') {
                field.options = Array.from(element.options)
                    .filter(opt => opt.value)
                    .map(opt => ({
                        value: opt.value,
                        label: opt.textContent.trim()
                    }));
            }

            if (type === 'radio') {
                const radios = document.querySelectorAll(`input[name="${name}"]`);
                field.options = Array.from(radios).map(radio => ({
                    value: radio.value,
                    label: this.getRadioLabel(radio)
                }));
            }

            return field;
        }

        getRadioLabel(radio) {
            const label = document.querySelector(`label[for="${radio.id}"]`);
            if (label) return label.textContent.trim();

            const parent = radio.parentElement;
            if (parent) {
                return parent.textContent.replace(radio.value, '').trim() || radio.value;
            }
            return radio.value;
        }

        detectFieldType(fieldInfo) {
            const { name, label, type, pattern } = fieldInfo;
            const nameLower = (name || '').toLowerCase();
            const labelLower = (label || '').toLowerCase();

            const labelNoParens = labelLower.replace(/\([^)]*\)/g, '').trim();
            const combinedId = `${nameLower}`;
            const combinedLabel = `${labelNoParens}`;
            if (
                combinedId.includes('name') ||
                combinedLabel.includes('name') ||
                combinedLabel.includes("father") ||
                combinedLabel.includes("mother") ||
                combinedLabel.includes("spouse")
            ) {
                if (!combinedId.includes('username') && !combinedLabel.includes('user name')) {
                    return 'name';
                }
            }
            const looksLikeAadhaar =
                combinedId.includes('aadhaar') ||
                combinedId.includes('aadhar') ||
                combinedId.includes('uid') ||
                (pattern && (pattern.includes('\\d{12}') || pattern.includes('[0-9]{12}')));
            if (looksLikeAadhaar) return 'aadhaar';
            const looksLikePan =
                (combinedId.includes('pan') && !combinedId.includes('company')) ||
                (pattern && /[A-Z]\{5\}.*\\d\{4\}.*[A-Z]/i.test(pattern));
            if (looksLikePan) return 'pan';
            if (combinedId.includes('mobile') || combinedId.includes('phone') || type === 'tel') {
                return 'mobile';
            }
            if (combinedId.includes('email') || type === 'email') {
                return 'email';
            }
            if (
                combinedId.includes('pincode') ||
                combinedId.includes('pin') ||
                combinedLabel.includes('pin') ||
                combinedLabel.includes('zip') ||
                (pattern && (pattern.includes('\\d{6}') || pattern.includes('[0-9]{6}')))
            ) {
                return 'pincode';
            }
            if (type === 'date' || combinedId.includes('dob') || combinedLabel.includes('birth')) {
                return 'date';
            }
            if (combinedId.includes('address') || combinedLabel.includes('address') || type === 'textarea') {
                return 'address';
            }
            if ((combinedId.includes('state') || combinedLabel.includes('state')) && !combinedLabel.includes('status')) {
                return 'select';
            }

            return type;
        }

        fillField(fieldId, value) {
            const element = document.getElementById(fieldId) || document.querySelector(`[name="${fieldId}"]`);

            if (!element) {
                console.warn(`[VoiceAssistant] Field not found: ${fieldId}`);
                return false;
            }

            const type = element.type || element.tagName.toLowerCase();

            if (type === 'radio') {
                const radios = document.querySelectorAll(`input[name="${element.name}"]`);
                radios.forEach(radio => {
                    if (radio.value.toLowerCase() === value.toLowerCase()) {
                        radio.checked = true;
                        radio.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                });
            } else if (element.tagName === 'SELECT') {
                const options = Array.from(element.options);
                const match = options.find(opt =>
                    opt.value.toLowerCase() === value.toLowerCase() ||
                    opt.textContent.toLowerCase().includes(value.toLowerCase())
                );
                if (match) {
                    element.value = match.value;
                }
            } else {
                element.value = value;
            }

            element.dispatchEvent(new Event('input', { bubbles: true }));
            element.dispatchEvent(new Event('change', { bubbles: true }));

            console.log(`[VoiceAssistant] Filled ${fieldId} with: ${value}`);
            return true;
        }

        highlightField(fieldId) {
            document.querySelectorAll('.va-highlighted-field').forEach(el => {
                el.classList.remove('va-highlighted-field');
            });

            const element = document.getElementById(fieldId) || document.querySelector(`[name="${fieldId}"]`);
            if (element) {
                element.classList.add('va-highlighted-field');
                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                element.focus();
            }
        }
    }

    /**
     * Audio Handler - Manages microphone capture and playback
     */
    class AudioHandler {
        constructor() {
            this.mediaRecorder = null;
            this.audioContext = null;
            this.analyser = null;
            this.stream = null;
            this.isRecording = false;
            this.audioChunks = [];
            this.onAudioComplete = null;
            this.audioQueue = [];  // Queue for audio playback
            this.isPlayingAudio = false;  // Flag to prevent overlapping
        }

        async init() {
            try {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000  // Match backend sample rate for better compatibility
                });
                console.log('[VoiceAssistant] Audio context initialized');
                return true;
            } catch (error) {
                console.error('[VoiceAssistant] Failed to initialize audio:', error);
                return false;
            }
        }

        async startRecording(onComplete) {
            try {
                this.stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });

                this.onAudioComplete = onComplete;
                this.audioChunks = [];

                if (!this.audioContext) await this.init();
                if (this.audioContext.state === 'suspended') {
                    await this.audioContext.resume();
                }

                this.analyser = this.audioContext.createAnalyser();
                this.analyser.fftSize = 256;

                const source = this.audioContext.createMediaStreamSource(this.stream);
                source.connect(this.analyser);
                let mimeType = 'audio/webm';
                if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                    mimeType = 'audio/webm;codecs=opus';
                } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                    mimeType = 'audio/webm';
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    mimeType = 'audio/mp4';
                } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
                    mimeType = 'audio/ogg';
                }

                console.log('[VoiceAssistant] Using audio format:', mimeType);

                this.mediaRecorder = new MediaRecorder(this.stream, {
                    mimeType: mimeType,
                    audioBitsPerSecond: 128000  // 128kbps for better quality
                });
                this.mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        this.audioChunks.push(event.data);
                    }
                };
                this.mediaRecorder.onstop = async () => {
                    if (this.audioChunks.length > 0 && this.onAudioComplete) {
                        const audioBlob = new Blob(this.audioChunks, { type: mimeType });
                        const base64 = await this.blobToBase64(audioBlob);
                        console.log('[VoiceAssistant] Audio recorded:', audioBlob.size, 'bytes');
                        this.onAudioComplete(base64);
                    }
                };
                this.mediaRecorder.start();
                this.isRecording = true;

                console.log('[VoiceAssistant] Recording started');
                return true;

            } catch (error) {
                console.error('[VoiceAssistant] Failed to start recording:', error);
                return false;
            }
        }

        stopRecording() {
            if (this.mediaRecorder && this.isRecording) {
                this.mediaRecorder.stop();
                this.isRecording = false;
            }

            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }

            console.log('[VoiceAssistant] Recording stopped');
        }

        async blobToBase64(blob) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64 = reader.result.split(',')[1];
                    resolve(base64);
                };
                reader.onerror = reject;
                reader.readAsDataURL(blob);
            });
        }

        async playAudio(base64Audio, text = '') {
            return new Promise((resolve) => {
                this.audioQueue.push({ base64Audio, text, resolve });
                console.log(`[VoiceAssistant] Audio queued (queue length: ${this.audioQueue.length})`);
                if (!this.isPlayingAudio) {
                    this._processAudioQueue();
                }
            });
        }

        async _processAudioQueue() {
            if (this.audioQueue.length === 0) {
                this.isPlayingAudio = false;
                return;
            }

            this.isPlayingAudio = true;
            const { base64Audio, text, resolve } = this.audioQueue.shift();

            try {
                if (!base64Audio) {
                    resolve();
                    this._processAudioQueue();
                    return;
                }

                console.log(`[VoiceAssistant] Playing audio: "${text.substring(0, 50)}..."`);
                const audio = new Audio(`data:audio/wav;base64,${base64Audio}`);
                
                audio.onended = () => {
                    console.log('[VoiceAssistant] Audio completed');
                    resolve();
                    setTimeout(() => this._processAudioQueue(), 300);
                };
                
                audio.onerror = async (e) => {
                    console.warn('[VoiceAssistant] HTML5 Audio failed, trying Web Audio API');
                    try {
                        const audioData = atob(base64Audio);
                        const arrayBuffer = new ArrayBuffer(audioData.length);
                        const view = new Uint8Array(arrayBuffer);

                        for (let i = 0; i < audioData.length; i++) {
                            view[i] = audioData.charCodeAt(i);
                        }

                        if (!this.audioContext) await this.init();
                        
                        if (this.audioContext.state === 'suspended') {
                            await this.audioContext.resume();
                        }

                        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
                        const source = this.audioContext.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(this.audioContext.destination);
                        source.start(0);
                        
                        source.onended = () => {
                            console.log('[VoiceAssistant] Web Audio API playback completed');
                            resolve();
                            setTimeout(() => this._processAudioQueue(), 300);
                        };
                    } catch (webAudioError) {
                        console.error('[VoiceAssistant] Both audio methods failed:', webAudioError);
                        resolve();
                        this._processAudioQueue();
                    }
                };
                
                await audio.play();

            } catch (error) {
                console.error('[VoiceAssistant] Audio playback error:', error);
                resolve();
                this._processAudioQueue();
            }
        }

        getAnalyserData() {
            if (!this.analyser) return null;
            const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            this.analyser.getByteFrequencyData(dataArray);
            return dataArray;
        }
        
        getMicLevel() {
            if (!this.analyser) return 0;
            const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            this.analyser.getByteFrequencyData(dataArray);
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                sum += dataArray[i];
            }
            return (sum / dataArray.length) / 255;
        }
    }

    /**
     * Waveform Visualizer
     */
    class WaveformVisualizer {
        constructor(canvas) {
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');
            this.animationId = null;
            this.audioHandler = null;
        }

        setAudioHandler(handler) {
            this.audioHandler = handler;
        }

        start() {
            const draw = () => {
                this.animationId = requestAnimationFrame(draw);
                const data = this.audioHandler?.getAnalyserData();
                this.drawWaveform(data);
            };
            draw();
        }

        stop() {
            if (this.animationId) {
                cancelAnimationFrame(this.animationId);
                this.animationId = null;
            }
            this.clear();
        }

        drawWaveform(data) {
            const { width, height } = this.canvas;
            this.ctx.fillStyle = '#1a1a2e';
            this.ctx.fillRect(0, 0, width, height);

            if (!data || data.length === 0) {
                this.ctx.strokeStyle = '#4a9eff';
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.moveTo(0, height / 2);
                this.ctx.lineTo(width, height / 2);
                this.ctx.stroke();
                return;
            }

            const barWidth = width / data.length;
            const gradient = this.ctx.createLinearGradient(0, 0, 0, height);
            gradient.addColorStop(0, '#4a9eff');
            gradient.addColorStop(1, '#0f3460');
            this.ctx.fillStyle = gradient;

            for (let i = 0; i < data.length; i++) {
                const barHeight = (data[i] / 255) * height * 0.8;
                const x = i * barWidth;
                const y = (height - barHeight) / 2;
                this.ctx.fillRect(x, y, barWidth - 1, barHeight);
            }
        }

        clear() {
            const { width, height } = this.canvas;
            this.ctx.fillStyle = '#1a1a2e';
            this.ctx.fillRect(0, 0, width, height);
        }
    }

    /**
     * UI Overlay
     */
    class UIOverlay {
        constructor() {
            this.container = null;
            this.isOpen = false;
            this.elements = {};
            // Disable the record button for multiple reasons (processing, playback).
            // This prevents recording while the assistant is speaking (common cause of "rubbish" transcripts).
            this._disableReasons = { processing: false, playback: false };
        }

        create() {
            this.container = document.createElement('div');
            this.container.id = 'voice-assistant-widget';
            this.container.innerHTML = `
                <button class="va-toggle-btn" title="Voice Assistant">
                    <svg viewBox="0 0 24 24" width="24" height="24">
                        <path fill="currentColor" d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.91-3c-.49 0-.9.36-.98.85C16.52 14.2 14.47 16 12 16s-4.52-1.8-4.93-4.15c-.08-.49-.49-.85-.98-.85-.61 0-1.09.54-1 1.14.49 3 2.89 5.35 5.91 5.78V20c0 .55.45 1 1 1s1-.45 1-1v-2.08c3.02-.43 5.42-2.78 5.91-5.78.1-.6-.39-1.14-1-1.14z"/>
                    </svg>
                </button>
                <div class="va-panel">
                    <div class="va-header">
                        <span class="va-title">Voice Assistant</span>
                        <button class="va-close-btn">&times;</button>
                    </div>
                    <div class="va-content">
                        <div class="va-status">Click Start to begin</div>
                        <div class="va-current-field" style="display: none;"></div>
                        <canvas class="va-waveform" width="280" height="60"></canvas>
                        <div class="va-mic-level-container">
                            <div class="va-mic-level-bar"></div>
                        </div>
                        <div class="va-transcript"></div>
                        <div class="va-controls">
                            <button class="va-start-btn" title="Start session">
                                <svg viewBox="0 0 24 24" width="32" height="32">
                                    <path fill="currentColor" d="M8 5v14l11-7z"/>
                                </svg>
                            </button>
                            <button class="va-record-btn" style="display: none;" title="Click to record">
                                <svg viewBox="0 0 24 24" width="32" height="32">
                                    <circle cx="12" cy="12" r="10" fill="currentColor"/>
                                </svg>
                            </button>
                        </div>
                        <div class="va-speak-hint" style="display: none;">Speak now</div>
                        <div class="va-confirmation" style="display: none;">
                            <div class="va-confirm-text"></div>
                            <div class="va-confirm-buttons">
                                <button class="va-yes-btn">Yes ✓</button>
                                <button class="va-no-btn">No ✗</button>
                            </div>
                        </div>
                        <div class="va-progress">
                            <div class="va-progress-bar"></div>
                            <span class="va-progress-text">0/0 fields</span>
                        </div>
                    </div>
                </div>
            `;

            document.body.appendChild(this.container);

            this.elements = {
                toggleBtn: this.container.querySelector('.va-toggle-btn'),
                panel: this.container.querySelector('.va-panel'),
                closeBtn: this.container.querySelector('.va-close-btn'),
                status: this.container.querySelector('.va-status'),
                currentField: this.container.querySelector('.va-current-field'),
                speakHint: this.container.querySelector('.va-speak-hint'),
                waveform: this.container.querySelector('.va-waveform'),
                micLevelBar: this.container.querySelector('.va-mic-level-bar'),
                transcript: this.container.querySelector('.va-transcript'),
                startBtn: this.container.querySelector('.va-start-btn'),
                recordBtn: this.container.querySelector('.va-record-btn'),
                confirmation: this.container.querySelector('.va-confirmation'),
                confirmText: this.container.querySelector('.va-confirm-text'),
                yesBtn: this.container.querySelector('.va-yes-btn'),
                noBtn: this.container.querySelector('.va-no-btn'),
                progressBar: this.container.querySelector('.va-progress-bar'),
                progressText: this.container.querySelector('.va-progress-text')
            };

            this.elements.toggleBtn.addEventListener('click', () => this.toggle());
            this.elements.closeBtn.addEventListener('click', () => this.hide());

            console.log('[VoiceAssistant] UI created');
        }

        show() {
            this.elements.panel.classList.add('va-open');
            this.isOpen = true;
        }

        hide() {
            this.elements.panel.classList.remove('va-open');
            this.isOpen = false;
        }

        toggle() {
            if (this.isOpen) {
                this.hide();
            } else {
                this.show();
            }
        }

        setStatus(message, type = 'info') {
            this.elements.status.textContent = message;
            this.elements.status.className = `va-status va-status-${type}`;
        }

        setCurrentField(fieldLabel, fieldId) {
            if (fieldLabel) {
                // Clean up the label - remove content in parentheses and extra whitespace
                const cleanLabel = fieldLabel.replace(/\s*\([^)]*\)/g, '').trim();
                this.elements.currentField.textContent = `Current: ${cleanLabel}`;
                this.elements.currentField.style.display = 'block';
            } else {
                this.elements.currentField.style.display = 'none';
            }
        }

        setTranscript(text) {
            this.elements.transcript.textContent = text;
        }

        showRecordingState() {
            this.elements.recordBtn.classList.add('va-recording');
            this.setStatus('Listening... (click to stop)', 'recording');
        }

        hideRecordingState() {
            this.elements.recordBtn.classList.remove('va-recording');
        }

        showProcessingState() {
            this.setStatus('Processing...', 'processing');
            this._disableReasons.processing = true;
            this._updateRecordDisabled();
        }

        hideProcessingState() {
            this._disableReasons.processing = false;
            this._updateRecordDisabled();
        }

        showPlaybackState() {
            this._disableReasons.playback = true;
            this._updateRecordDisabled();
        }

        hidePlaybackState() {
            this._disableReasons.playback = false;
            this._updateRecordDisabled();
            this.showSpeakHint();
        }

        _updateRecordDisabled() {
            if (!this.elements.recordBtn) return;
            this.elements.recordBtn.disabled = Boolean(
                this._disableReasons.processing || this._disableReasons.playback
            );
        }

        showStartButton() {
            if (this.elements.startBtn) {
                this.elements.startBtn.style.display = 'block';
            }
            if (this.elements.recordBtn) {
                this.elements.recordBtn.style.display = 'none';
            }
        }

        showRecordButton() {
            if (this.elements.startBtn) {
                this.elements.startBtn.style.display = 'none';
            }
            if (this.elements.recordBtn) {
                this.elements.recordBtn.style.display = 'block';
            }
        }

        showSpeakHint() {
            if (this.elements.speakHint) {
                this.elements.speakHint.style.display = 'block';
            }
            if (this.elements.recordBtn) {
                this.elements.recordBtn.classList.add('va-ready-to-record');
            }
        }

        hideSpeakHint() {
            if (this.elements.speakHint) {
                this.elements.speakHint.style.display = 'none';
            }
            if (this.elements.recordBtn) {
                this.elements.recordBtn.classList.remove('va-ready-to-record');
            }
        }

        updateMicLevel(level) {
            if (this.elements.micLevelBar) {
                const percentage = Math.min(100, Math.max(0, level * 100));
                this.elements.micLevelBar.style.width = `${percentage}%`;
            }
        }

        showConfirmation(text) {
            this.elements.confirmText.textContent = text;
            this.elements.confirmation.style.display = 'block';
        }

        hideConfirmation() {
            this.elements.confirmation.style.display = 'none';
        }

        updateProgress(filled, total) {
            const percentage = total > 0 ? (filled / total) * 100 : 0;
            this.elements.progressBar.style.width = `${percentage}%`;
            this.elements.progressText.textContent = `${filled}/${total} fields`;
        }

        showError(message) {
            this.setStatus(message, 'error');
        }

        getCanvas() {
            return this.elements.waveform;
        }
    }

    /**
     * Main Voice Form Assistant Widget
     */
    class VoiceFormAssistant {
        constructor() {
            this.ui = new UIOverlay();
            this.formAnalyzer = new FormAnalyzer();
            this.audioHandler = new AudioHandler();
            this.visualizer = null;
            this.ws = null;
            this.sessionId = null;
            this.isConnected = false;
            this.isRecording = false;
        }

        async init() {
            console.log('[VoiceAssistant] Initializing...');

            this.ui.create();

            this.visualizer = new WaveformVisualizer(this.ui.getCanvas());
            this.visualizer.setAudioHandler(this.audioHandler);

            await this.audioHandler.init();

            this.setupEventListeners();

            console.log('[VoiceAssistant] Initialized');
        }

        setupEventListeners() {
            // Start button - initiates session without recording
            this.ui.elements.startBtn.addEventListener('click', async () => {
                if (this.audioHandler.audioContext && 
                    this.audioHandler.audioContext.state === 'suspended') {
                    await this.audioHandler.audioContext.resume();
                    console.log('[VoiceAssistant] AudioContext resumed after user interaction');
                }
                await this.startSession();
            });

            // Record button - only for recording responses
            this.ui.elements.recordBtn.addEventListener('click', async () => {
                if (this.audioHandler.audioContext && 
                    this.audioHandler.audioContext.state === 'suspended') {
                    await this.audioHandler.audioContext.resume();
                    console.log('[VoiceAssistant] AudioContext resumed after user interaction');
                }
                
                if (this.isRecording) {
                    this.stopRecording();
                } else {
                    this.startRecording();
                }
            });

            this.ui.elements.yesBtn.addEventListener('click', async () => {
                if (this.audioHandler.audioContext && 
                    this.audioHandler.audioContext.state === 'suspended') {
                    await this.audioHandler.audioContext.resume();
                }
                this.sendConfirmation(true);
            });

            this.ui.elements.noBtn.addEventListener('click', async () => {
                if (this.audioHandler.audioContext && 
                    this.audioHandler.audioContext.state === 'suspended') {
                    await this.audioHandler.audioContext.resume();
                }
                this.sendConfirmation(false);
            });
        }

        async connect() {
            return new Promise((resolve, reject) => {
                try {
                    this.ws = new WebSocket(CONFIG.wsUrl);

                    this.ws.onopen = () => {
                        console.log('[VoiceAssistant] WebSocket connected');
                        this.isConnected = true;
                        this.ui.setStatus('Connected', 'success');
                        resolve();
                    };

                    this.ws.onclose = () => {
                        console.log('[VoiceAssistant] WebSocket disconnected');
                        this.isConnected = false;
                        this.ui.setStatus('Disconnected', 'error');
                    };

                    this.ws.onerror = (error) => {
                        console.error('[VoiceAssistant] WebSocket error:', error);
                        this.ui.showError('Connection failed');
                        reject(error);
                    };

                    this.ws.onmessage = (event) => {
                        this.handleMessage(JSON.parse(event.data));
                    };

                } catch (error) {
                    reject(error);
                }
            });
        }

        async handleMessage(message) {
            console.log('[VoiceAssistant] Received:', message.type);

            switch (message.type) {
                case 'greeting':
                    this.sessionId = message.sessionId;
                    this.ui.setStatus('Assistant ready');
                    this.ui.setTranscript(message.text);
                    this.ui.showPlaybackState();
                    await this.audioHandler.playAudio(message.audio, message.text);
                    this.ui.hidePlaybackState();
                    // Now show record button for user to respond
                    this.ui.showRecordButton();
                    break;

                case 'ask_field':
                    this.ui.setStatus('Waiting for your response...');
                    this.ui.setCurrentField(message.fieldLabel || message.fieldId, message.fieldId);
                    this.ui.setTranscript(message.text);
                    this.ui.hideConfirmation();
                    this.formAnalyzer.highlightField(message.fieldId);
                    if (message.progress) {
                        this.ui.updateProgress(message.progress.filled_fields, message.progress.total_fields);
                    }
                    this.ui.showPlaybackState();
                    await this.audioHandler.playAudio(message.audio, message.text);
                    this.ui.hidePlaybackState();
                    this.ui.hideProcessingState();
                    // Show record button so user can respond
                    this.ui.showRecordButton();
                    break;

                case 'confirmation_request':
                    this.ui.setStatus('Confirm value');
                    if (message.fieldLabel || message.fieldId) {
                        this.ui.setCurrentField(message.fieldLabel || message.fieldId, message.fieldId);
                    }
                    this.ui.setTranscript(message.text);
                    this.ui.showConfirmation(message.text, message.value);
                    this.ui.showPlaybackState();
                    await this.audioHandler.playAudio(message.audio, message.text);
                    this.ui.hidePlaybackState();
                    this.ui.hideProcessingState();
                    // Show record button for voice confirmation
                    this.ui.showRecordButton();
                    break;

                case 'fill_field':
                    this.formAnalyzer.fillField(message.fieldId, message.value);
                    if (message.text) {
                        this.ui.setTranscript(message.text);
                        this.ui.showPlaybackState();
                        await this.audioHandler.playAudio(message.audio, message.text);
                        this.ui.hidePlaybackState();
                    }
                    if (message.nextField) {
                        // Update current field to next field
                        this.ui.setCurrentField(message.nextField.label, message.nextField.id);
                        this.formAnalyzer.highlightField(message.nextField.id);
                    }
                    this.ui.hideProcessingState();
                    // Show record button for next field
                    this.ui.showRecordButton();
                    break;

                case 'validation_error':
                    this.ui.showError(message.error);
                    this.ui.setTranscript(message.text);
                    if (message.fieldLabel || message.fieldId) {
                        this.ui.setCurrentField(message.fieldLabel || message.fieldId, message.fieldId);
                    }
                    this.ui.showPlaybackState();
                    await this.audioHandler.playAudio(message.audio, message.text);
                    this.ui.hidePlaybackState();
                    this.ui.hideProcessingState();
                    // Show record button to re-enter value
                    this.ui.showRecordButton();
                    break;

                case 'repeat':
                case 'clarify':
                    this.ui.setStatus('Please repeat');
                    this.ui.setTranscript(message.text);
                    this.ui.showPlaybackState();
                    await this.audioHandler.playAudio(message.audio, message.text);
                    this.ui.hidePlaybackState();
                    this.ui.hideProcessingState();
                    // Show record button to repeat
                    this.ui.showRecordButton();
                    break;

                case 'audio_quality_error':
                    this.ui.setStatus('Speak louder', 'error');
                    this.ui.setTranscript(message.text);
                    this.ui.showPlaybackState();
                    await this.audioHandler.playAudio(message.audio, message.text);
                    this.ui.hidePlaybackState();
                    this.ui.hideProcessingState();
                    // Show record button to try again
                    this.ui.showRecordButton();
                    break;

                case 'completion':
                    this.ui.setStatus('Form completed!', 'success');
                    this.ui.setCurrentField(null);  // Hide current field indicator
                    this.ui.setTranscript(message.text);
                    this.ui.hideConfirmation();
                    if (message.progress) {
                        this.ui.updateProgress(message.progress.filled_fields, message.progress.total_fields);
                    }
                    this.ui.showPlaybackState();
                    await this.audioHandler.playAudio(message.audio, message.text);
                    this.ui.hidePlaybackState();
                    this.ui.hideProcessingState();
                    break;

                case 'error':
                    this.ui.showError(message.error || 'An error occurred');
                    this.ui.setTranscript(message.text);
                    this.ui.showPlaybackState();
                    await this.audioHandler.playAudio(message.audio, message.text);
                    this.ui.hidePlaybackState();
                    this.ui.hideProcessingState();
                    break;

                default:
                    console.log('[VoiceAssistant] Unknown message type:', message.type);
                    this.ui.hideProcessingState();
            }
        }

        async startSession() {
            if (!this.isConnected) {
                await this.connect();
            }

            const formSchema = this.formAnalyzer.analyzePage();

            if (formSchema.fields.length === 0) {
                this.ui.showError('No form fields found on this page');
                return;
            }

            this.send({
                type: 'init',
                formSchema: formSchema
            });

            this.ui.setStatus('Starting session...');
            this.ui.updateProgress(0, formSchema.fields.length);
        }

        async startRecording() {
            // Check if session exists
            if (!this.sessionId) {
                this.ui.setStatus('Please start a session first', 'error');
                return;
            }

            // Prevent recording while assistant audio is playing; otherwise the mic often captures TTS.
            if (this.audioHandler.isPlayingAudio || (this.audioHandler.audioQueue && this.audioHandler.audioQueue.length > 0)) {
                this.ui.setStatus('Wait for the assistant to finish speaking...', 'info');
                return;
            }

            this.isRecording = true;
            this.ui.hideSpeakHint();
            this.ui.showRecordingState();
            this.visualizer.start();
            this._startMicLevelMonitor();
            await this.audioHandler.startRecording((base64Audio) => {
                this.ui.showProcessingState();
                this.send({
                    type: 'audio_complete',
                    sessionId: this.sessionId,
                    audio: base64Audio
                });
            });
        }

        stopRecording() {
            this.isRecording = false;
            this._stopMicLevelMonitor();
            this.audioHandler.stopRecording();
            this.ui.hideRecordingState();
            this.ui.updateMicLevel(0);
            this.visualizer.stop();
        }
        
        _startMicLevelMonitor() {
            this._micLevelInterval = setInterval(() => {
                const level = this.audioHandler.getMicLevel();
                this.ui.updateMicLevel(level);
            }, 50);
        }
        
        _stopMicLevelMonitor() {
            if (this._micLevelInterval) {
                clearInterval(this._micLevelInterval);
                this._micLevelInterval = null;
            }
        }

        sendConfirmation(confirmed) {
            this.send({
                type: 'user_confirmation',
                sessionId: this.sessionId,
                confirmed: confirmed
            });
            this.ui.hideConfirmation();
            this.ui.showProcessingState();
        }

        send(message) {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify(message));
                console.log('[VoiceAssistant] Sent:', message.type);
            } else {
                console.warn('[VoiceAssistant] WebSocket not connected');
            }
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.voiceAssistant = new VoiceFormAssistant();
            window.voiceAssistant.init();
        });
    } else {
        window.voiceAssistant = new VoiceFormAssistant();
        window.voiceAssistant.init();
    }

})();
