(function () {
    function initializeTheme() {
        let theme = localStorage.getItem('mdbook-theme');

        if (!theme) {
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                theme = 'navy';
            } else {
                theme = 'light';
            }
            localStorage.setItem('mdbook-theme', theme);
        }

        const root = typeof path_to_root === 'undefined' ? './' : path_to_root;

        const themeElement = document.getElementById('theme');
        if (themeElement) {
            themeElement.setAttribute('href', `${root}theme/${theme}.css`);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeTheme);
    } else {
        initializeTheme();
    }

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        if (!localStorage.getItem('mdbook-theme')) {
            const newTheme = e.matches ? 'navy' : 'light';
            localStorage.setItem('mdbook-theme', newTheme);
            const root = typeof path_to_root === 'undefined' ? './' : path_to_root;
            const themeElement = document.getElementById('theme');
            if (themeElement) {
                themeElement.setAttribute('href', `${root}theme/${newTheme}.css`);
            }
        }
    });

    window.addEventListener('storage', function (e) {
        if (e.key === 'mdbook-theme') {
            const root = typeof path_to_root === 'undefined' ? './' : path_to_root;
            const themeElement = document.getElementById('theme');
            if (themeElement) {
                themeElement.setAttribute('href', `${root}theme/${e.newValue}.css`);
            }
        }
    });
})();
