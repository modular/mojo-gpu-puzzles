// Language Switcher for mdBook
// Adds a language dropdown to the navigation bar

(function() {
    'use strict';

    // Configuration
    const LANGUAGES = {
        en: { label: 'EN', name: 'English' },
        ko: { label: 'KO', name: '한국어' }
    };

    // Detect current language from URL path
    function getCurrentLanguage() {
        const path = window.location.pathname;
        if (path.startsWith('/ko/') || path === '/ko') {
            return 'ko';
        }
        return 'en';
    }

    // Get the equivalent page path in another language
    function getLanguagePath(targetLang) {
        const path = window.location.pathname;
        const currentLang = getCurrentLanguage();

        if (currentLang === targetLang) {
            return path;
        }

        let pagePath = path;

        // Remove current language prefix if exists
        if (currentLang === 'ko') {
            pagePath = path.replace(/^\/ko/, '');
        }

        // Add target language prefix
        if (targetLang === 'ko') {
            return '/ko' + (pagePath || '/index.html');
        }

        return pagePath || '/index.html';
    }

    // Create the language switcher dropdown
    function createLanguageSwitcher() {
        const currentLang = getCurrentLanguage();

        const container = document.createElement('div');
        container.className = 'language-switcher';

        const button = document.createElement('button');
        button.className = 'lang-dropdown-btn';
        button.setAttribute('aria-label', 'Select language');
        button.setAttribute('aria-haspopup', 'true');
        button.setAttribute('aria-expanded', 'false');
        button.innerHTML = `<span class="material-symbols-outlined">language</span>`;

        const dropdown = document.createElement('ul');
        dropdown.className = 'lang-dropdown';
        dropdown.setAttribute('role', 'menu');

        for (const [lang, info] of Object.entries(LANGUAGES)) {
            const item = document.createElement('li');
            item.setAttribute('role', 'none');

            if (lang === currentLang) {
                const span = document.createElement('span');
                span.className = 'lang-item lang-item-current';
                span.setAttribute('role', 'menuitem');
                span.textContent = info.name;
                item.appendChild(span);
            } else {
                const link = document.createElement('a');
                link.className = 'lang-item';
                link.setAttribute('role', 'menuitem');
                link.href = getLanguagePath(lang);
                link.textContent = info.name;
                item.appendChild(link);
            }

            dropdown.appendChild(item);
        }

        container.appendChild(button);
        container.appendChild(dropdown);

        // Toggle dropdown
        button.addEventListener('click', function(e) {
            e.stopPropagation();
            const isExpanded = button.getAttribute('aria-expanded') === 'true';
            button.setAttribute('aria-expanded', !isExpanded);
            container.classList.toggle('open');
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', function() {
            button.setAttribute('aria-expanded', 'false');
            container.classList.remove('open');
        });

        return container;
    }

    // Insert the language switcher into the page
    function insertLanguageSwitcher() {
        // Skip if already exists
        if (document.querySelector('.language-switcher')) {
            return;
        }

        const placeholder = document.getElementById('language-switcher-placeholder');
        if (placeholder) {
            const switcher = createLanguageSwitcher();
            placeholder.replaceWith(switcher);
        }
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', insertLanguageSwitcher);
    } else {
        insertLanguageSwitcher();
    }
})();
