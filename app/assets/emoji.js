const editor = element.closest("#editor-column");
const targets = '#single-text textarea, #batch-input textarea, #queue-table td[data-col="1"] textarea';
const status = element.querySelector(".emoji-status");
let selection = null;

function remember(field) {
    if (!field.matches(targets)) return;
    selection = { field, start: field.selectionStart, end: field.selectionEnd };
    status.hidden = true;
}

for (const type of ["focusin", "focusout", "select", "keyup", "pointerup", "input"]) {
    editor.addEventListener(type, (event) => remember(event.target), true);
}

element.addEventListener("pointerdown", (event) => {
    if (event.target.closest("summary, button[data-emoji]")) event.preventDefault();
});

element.addEventListener("click", (event) => {
    event.stopPropagation();
    const button = event.target.closest("button[data-emoji]");
    if (!button) return;
    if (!selection || !selection.field.isConnected || !selection.field.getClientRects().length) {
        status.textContent = "Place the cursor in a dialogue field first.";
        status.hidden = false;
        return;
    }

    const { field, start, end } = selection;
    const emoji = button.dataset.emoji;
    field.focus({ preventScroll: true });
    field.setSelectionRange(start, end);
    field.setRangeText(emoji, start, end, "end");
    field.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: emoji }));
    if (field.closest("#queue-table")) {
        field.dispatchEvent(new FocusEvent("blur"));
    }
    remember(field);
});
