(function () {
    // This function runs after DOM is fully loaded to ensure theme persistence
    function ensureThemePersistence() {
        let theme = localStorage.getItem('mdbook-theme');
        if (!theme) {
            theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches
                ? 'light' : 'navy';
            localStorage.setItem('mdbook-theme', theme);
        }

        // Use the path to mdBook's built-in themes
        const themeElement = document.getElementById('theme');
        if (themeElement) {
            // This path works with mdBook's internal themes
            themeElement.setAttribute('href', `${path_to_root}${default_theme_path}${theme}.css`);
        }

        // Force repaint to ensure theme is applied
        document.body.style.display = 'none';
        document.body.offsetHeight; // Force reflow
        document.body.style.display = '';
    }

    // Run on page load
    ensureThemePersistence();

    // Also handle theme changes
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', e => {
        if (!localStorage.getItem('mdbook-theme')) {
            const newTheme = e.matches ? 'light' : 'navy';
            localStorage.setItem('mdbook-theme', newTheme);
            const root = typeof path_to_root === 'undefined' ? './' : path_to_root;
            document.getElementById('theme').setAttribute('href', `${root}theme/${newTheme}.css`);
        }
    });

    window.addEventListener('storage', function (e) {
        if (e.key === 'mdbook-theme') {
            const root = typeof path_to_root === 'undefined' ? './' : path_to_root;
            document.getElementById('theme').setAttribute('href', `${root}theme/${e.newValue}.css`);
        }
    });
})();
