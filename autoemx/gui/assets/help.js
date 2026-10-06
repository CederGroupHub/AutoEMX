// Clicking a "?" icon of a parameter shows (or hides) its help text below the parameter.
// Dash serves the scripts of the assets folder automatically.
(function () {
    function toggleHelp(icon) {
        const row = icon.closest('.param-row');
        if (!row) { return; }
        const shown = row.querySelector(':scope > .help-text');
        if (shown) { shown.remove(); icon.classList.remove('help-open'); return; }
        const text = document.createElement('div');
        text.className = 'help-text';
        text.textContent = icon.getAttribute('data-help') || icon.getAttribute('title') || '';
        row.appendChild(text);
        icon.classList.add('help-open');
    }
    document.addEventListener('click', function (event) {
        const icon = event.target.closest && event.target.closest('.help');
        if (!icon) { return; }
        // The icon can sit inside a <label>: do not toggle the checkbox or focus the field
        event.preventDefault();
        event.stopPropagation();
        toggleHelp(icon);
    }, true);
    document.addEventListener('keydown', function (event) {
        if ((event.key === 'Enter' || event.key === ' ') && event.target.classList
                && event.target.classList.contains('help')) {
            event.preventDefault();
            toggleHelp(event.target);
        }
    });
})();
