// Polling-based livereload for local development.
// Used by translation pages (built via mdbook watch, which lacks WebSocket livereload).
// English pages use mdbook serve's native WebSocket livereload instead.

(function() {
    'use strict';

    // Only run on localhost (no impact on deployed site)
    var host = window.location.hostname;
    if (host !== 'localhost' && host !== '127.0.0.1') return;

    // Skip if mdbook serve's native livereload is already active
    if (typeof defined_socket !== 'undefined') return;

    var lastModified = null;

    setInterval(function() {
        fetch(window.location.pathname, { method: 'HEAD', cache: 'no-cache' })
            .then(function(r) {
                if (!r.ok) return; // Skip during rebuild (file temporarily missing)
                var mod = r.headers.get('Last-Modified');
                if (lastModified !== null && mod !== lastModified) {
                    location.reload();
                }
                lastModified = mod;
            })
            .catch(function() {}); // Ignore network errors
    }, 1000);
})();
