// Editable cells of the GUI tables (e.g. the elements of the samples in the Quantification tab) behave like text
// fields: a single click opens the cell for editing with the cursor at the end of its text, so the text can be
// changed character by character. Without this, a click only selects the cell (as in a spreadsheet): typing then
// replaces its whole text, and editing in place needs a double click.
// Dash serves the scripts of the assets folder automatically.
(function () {
    function placeCaretAtEnd(td) {
        const input = td.querySelector('input.dash-cell-value');
        if (!input || input.classList.contains('unfocused')) { return false; }
        input.focus();
        const end = input.value.length;
        input.setSelectionRange(end, end);
        return true;
    }
    document.addEventListener('click', function (event) {
        const td = event.target.closest && event.target.closest('.dash-table-container td.dash-cell');
        if (!td) { return; }
        const input = td.querySelector('input.dash-cell-value');
        // Only editable cells hold an input; a cell being edited holds a focused one (no 'unfocused' class)
        if (!input || !input.classList.contains('unfocused')) { return; }
        // After the table has handled the click (selecting the cell), open it as a double click does
        setTimeout(function () {
            td.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true, view: window }));
            let tries = 0;
            (function waitForEditor() {
                if (!placeCaretAtEnd(td) && ++tries < 10) { setTimeout(waitForEditor, 20); }
            })();
        }, 0);
    });
})();
