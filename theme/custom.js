window.addEventListener('load', function () {
    let theme = localStorage.getItem('mdbook-theme');
    if (!theme) {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            theme = 'navy';
        } else {
            theme = 'light';
        }
        localStorage.setItem('mdbook-theme', theme);
    }

    document.getElementById('theme').setAttribute('href', `${path_to_root}theme/${theme}.css`);

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        if (!localStorage.getItem('mdbook-theme')) {
            const newTheme = e.matches ? 'navy' : 'light';
            localStorage.setItem('mdbook-theme', newTheme);
            document.getElementById('theme').setAttribute('href', `${path_to_root}theme/${newTheme}.css`);
        }
    });
});

window.addEventListener('storage', function (e) {
    if (e.key === 'mdbook-theme') {
        document.getElementById('theme').setAttribute('href', `${path_to_root}theme/${e.newValue}.css`);
    }
});
