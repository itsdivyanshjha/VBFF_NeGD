/**
 * Voice Form Assistant - Embed Loader
 * Add this single script to any page to enable voice form filling.
 *
 * Usage: <script src="http://localhost:8000/static/embed.js"></script>
 */
(function() {
    // Prevent double loading
    if (window.VoiceAssistantLoaded) return;
    window.VoiceAssistantLoaded = true;

    // Configuration - change this to your server URL
    var baseUrl = 'http://localhost:8000';

    // Load CSS
    var style = document.createElement('link');
    style.rel = 'stylesheet';
    style.href = baseUrl + '/static/widget.css';
    document.head.appendChild(style);

    // Load main widget script
    var script = document.createElement('script');
    script.src = baseUrl + '/static/widget.js';
    script.async = true;
    document.head.appendChild(script);

    console.log('[VoiceAssistant] Loader initialized');
})();
