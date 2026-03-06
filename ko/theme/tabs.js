document.addEventListener('DOMContentLoaded', function () {
    // Initialize all tab groups on the page
    const tabContainers = document.querySelectorAll('.code-tabs');

    tabContainers.forEach(container => {
        const tabButtons = container.querySelectorAll('.tab-button');
        const tabContents = container.querySelectorAll('.tab-content');

        // Set first tab as active by default
        if (tabButtons.length > 0) {
            tabButtons[0].classList.add('active');
            tabContents[0].classList.add('active');
        }

        // Add click event listeners to tab buttons
        tabButtons.forEach((button, index) => {
            button.addEventListener('click', function () {
                // Remove active class from all buttons and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked button and corresponding content
                button.classList.add('active');
                tabContents[index].classList.add('active');

                // Store user preference in localStorage
                const tabGroup = container.getAttribute('data-tab-group');
                if (tabGroup) {
                    localStorage.setItem(`tab-preference-${tabGroup}`, index);
                }
            });
        });

        // Check for saved preference
        const tabGroup = container.getAttribute('data-tab-group');
        if (tabGroup) {
            const savedIndex = localStorage.getItem(`tab-preference-${tabGroup}`);
            if (savedIndex !== null && tabButtons[savedIndex]) {
                tabButtons[savedIndex].click();
            }
        }
    });
});
