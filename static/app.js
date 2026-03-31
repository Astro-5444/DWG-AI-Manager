// static/app.js  –  Two-mode workflow: Make Excel | Count Symbols

// =============================================================================
// STATE & TIMING
// =============================================================================
let pollInterval = null;
let startTime = null;
let stageStartTimes = {};

// Current working mode: null | "excel" | "count"
let currentMode = null;

// ── Fix #1: One-shot guard so checkboxes are never rebuilt after user touches them ──
let countPanelPopulated = false;

// Symbol/file data for count mode
let symbolsData = [];
let filesData = [];

// Source picker state
let sourcesShown = false;   // true once the source modal has been displayed this run

// Stage definitions per mode
const STAGES_EXCEL = [
    { id: 1, name: 'Todo / Setup', icon: 'file-text', color: '#3b82f6' },
    { id: 2, name: 'Images / Extract', icon: 'zap', color: '#8b5cf6' },
    { id: 3, name: 'AI Processing', icon: 'database', color: '#ec4899' },
    { id: 4, name: 'Excel Report', icon: 'file-spreadsheet', color: '#10b981' },
];

const STAGES_COUNT = [
    { id: 1, name: 'Todo / Setup', icon: 'file-text', color: '#3b82f6' },
    { id: 2, name: 'Image Conversion', icon: 'image', color: '#8b5cf6' },
    { id: 5, name: 'Generating JSON', icon: 'scan-search', color: '#f59e0b' },
];

const STAGES_ALL = [
    { id: 1, name: 'Todo / Setup', icon: 'file-text', color: '#3b82f6' },
    { id: 2, name: 'Images / Extract', icon: 'zap', color: '#8b5cf6' },
    { id: 3, name: 'AI Processing', icon: 'database', color: '#ec4899' },
    { id: 4, name: 'Excel Report', icon: 'file-spreadsheet', color: '#10b981' },
    { id: 5, name: 'Symbol Counter', icon: 'scan-search', color: '#f59e0b' },
];

const stageNames = {
    1: 'Todo / Setup',
    2: 'Images / Extract',
    3: 'AI Processing',
    4: 'Excel Report',
    5: 'Generating JSON'
};

// =============================================================================
// DOM ELEMENTS
// =============================================================================
const selectBtn = document.getElementById('selectBtn');
const excelBtn = document.getElementById('excelBtn');
const countBtn = document.getElementById('countBtn');
const actionBtns = document.getElementById('actionBtns');
const fileCountEl = document.getElementById('fileCount');
const selectedPathEl = document.getElementById('selectedPath');
const statusText = document.getElementById('statusText');
const progressPercent = document.getElementById('progressPercent');
const progressFill = document.getElementById('progressFill');
const logContainer = document.getElementById('logContainer');
const successModal = document.getElementById('successModal');
const modalOutputPath = document.getElementById('modalOutputPath');
const openExcelBtn = document.getElementById('openExcelBtn');
const closeModal = document.getElementById('closeModal');
const fileCounter = document.getElementById('fileCounter');
const countPanel = document.getElementById('countPanel');

// =============================================================================
// PYWEBVIEW HELPER
// =============================================================================
function getPyWebView() {
    return new Promise((resolve) => {
        if (window.pywebview && window.pywebview.api) return resolve(window.pywebview.api);
        window.addEventListener('pywebviewready', () => resolve(window.pywebview.api));
        setTimeout(() => {
            if (window.pywebview && window.pywebview.api) resolve(window.pywebview.api);
        }, 1000);
    });
}

// =============================================================================
// SELECT FOLDER
// =============================================================================
selectBtn.addEventListener('click', async () => {
    if (typeof pywebview === 'undefined') {
        alert('This feature only works when launched as a desktop app.');
        return;
    }
    try {
        const api = await getPyWebView();
        const folderPath = await api.selectFolder();
        if (!folderPath) return;

        const res = await fetch('/set-folder', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: folderPath })
        });
        const data = await res.json();

        if (data.success) {
            fileCountEl.textContent = data.file_count;
            selectedPathEl.textContent = folderPath;
            actionBtns.classList.remove('hidden');
            excelBtn.disabled = false;
            countBtn.disabled = false;
            countPanel.classList.add('hidden');
            // Reset the panel guard so a fresh selection is possible
            countPanelPopulated = false;
            currentMode = null;
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        console.error('Select folder error:', e);
        alert('Failed to select folder:\n' + e.message);
    }
});

// =============================================================================
// MAKE EXCEL BUTTON
// =============================================================================
excelBtn.addEventListener('click', async () => {
    if (excelBtn.disabled) return;
    currentMode = 'excel';
    countPanel.classList.add('hidden');
    countPanelPopulated = false;

    try {
        const res = await fetch('/start-excel', { method: 'POST' });
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.error || 'Failed to start');
        lockButtons();
        startTime = Date.now();
        stageStartTimes = {};
        startPolling();
    } catch (err) {
        alert('Error: ' + err.message);
    }
});

// =============================================================================
// COUNT BUTTON  – starts prepare step (images + discovery)
// =============================================================================
countBtn.addEventListener('click', async () => {
    if (countBtn.disabled) return;
    currentMode = 'count';
    // Reset guards so panel/modals are rebuilt fresh
    countPanelPopulated = false;
    sourcesShown = false;

    logContainer.innerHTML = ''; // clear for new run
    renderLogs(['[..:..:..] ⏳ Preparing project images... Please wait for symbol selection.']);

    try {
        const res = await fetch('/start-count-prepare', { method: 'POST' });
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.error || 'Failed to start');
        lockButtons();
        startTime = Date.now();
        stageStartTimes = {};
        startPolling();
    } catch (err) {
        alert('Error: ' + err.message);
    }
});

// =============================================================================
// SUBMIT COUNT (after user selects symbols + files)
// =============================================================================
async function submitCount() {
    const selSymbols = [...document.querySelectorAll('#symbolsGrid  input[type=checkbox]:checked')]
        .map(cb => cb.value);
    const selFiles = [...document.querySelectorAll('#filesGrid    input[type=checkbox]:checked')]
        .map(cb => cb.value);

    if (selSymbols.length === 0) { alert('Please select at least one symbol.'); return; }
    if (selFiles.length === 0) { alert('Please select at least one file.'); return; }

    try {
        const res = await fetch('/start-count-execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbols: selSymbols, files: selFiles })
        });
        const data = await res.json();
        if (!res.ok || !data.success) throw new Error(data.error || 'Failed to start count');

        countPanel.classList.add('hidden');
        lockButtons();
        startPolling();
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

// =============================================================================
// TOGGLE ALL CHECKBOXES
// =============================================================================
function toggleAll(prefix, state) {
    const gridId = prefix === 'sym' ? 'symbolsGrid' : 'filesGrid';
    document.querySelectorAll(`#${gridId} input[type=checkbox]`).forEach(cb => { cb.checked = state; });
    updateSelCount(prefix);
}

function updateSelCount(prefix) {
    const gridId = prefix === 'sym' ? 'symbolsGrid' : 'filesGrid';
    const countEl = document.getElementById(prefix === 'sym' ? 'symSelCount' : 'fileSelCount');
    if (!countEl) return;
    const n = document.querySelectorAll(`#${gridId} input[type=checkbox]:checked`).length;
    const total = document.querySelectorAll(`#${gridId} input[type=checkbox]`).length;
    countEl.textContent = `(${n} / ${total} selected)`;
}

// =============================================================================
// BUTTON LOCKING
// =============================================================================
function lockButtons() {
    selectBtn.disabled = true;
    excelBtn.disabled = true;
    countBtn.disabled = true;
    const rb = document.getElementById('runCountBtn');
    if (rb) rb.disabled = true;
}
function unlockButtons(folderSet) {
    selectBtn.disabled = false;
    excelBtn.disabled = !folderSet;
    countBtn.disabled = !folderSet;
    const rb = document.getElementById('runCountBtn');
    if (rb) rb.disabled = false;
}

// =============================================================================
// MODAL
// =============================================================================
openExcelBtn.addEventListener('click', async () => {
    try {
        const res = await fetch('/open-excel');
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Could not open file');
    } catch (err) {
        alert('Error: ' + err.message);
    }
});

closeModal.addEventListener('click', () => { successModal.classList.add('hidden'); });

// =============================================================================
// POLLING
// =============================================================================
function startPolling() {
    updateUI();
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(updateUI, 1000);
}

async function updateUI() {
    try {
        const res = await fetch('/state');
        if (!res.ok) return;
        const state = await res.json();

        // Sync mode if set server-side
        if (state.mode && !currentMode) currentMode = state.mode;

        // Timing
        if (state.processing && !startTime) startTime = Date.now();
        if (!state.processing && !state.output_path) { startTime = null; stageStartTimes = {}; }
        if (state.processing && state.current_stage > 0 && !stageStartTimes[state.current_stage])
            stageStartTimes[state.current_stage] = Date.now();

        // Timer display
        const totalElapsed = startTime ? Math.floor((Date.now() - startTime) / 1000) : 0;
        let currentStageTime = 0;
        let stageName = '—';
        if (state.processing && state.current_stage > 0 && stageStartTimes[state.current_stage]) {
            currentStageTime = Math.floor((Date.now() - stageStartTimes[state.current_stage]) / 1000);
            stageName = stageNames[state.current_stage] || 'Processing';
        }
        const timerEl = document.getElementById('timerDisplay');
        if (timerEl) {
            timerEl.textContent = (state.output_path && !state.processing)
                ? `Total: ${totalElapsed}s | Done!`
                : `Total: ${totalElapsed}s | Stage: ${stageName} (${currentStageTime}s)`;
        }

        // Buttons
        if (state.processing) {
            lockButtons();
        } else {
            unlockButtons(!!state.source_folder);
        }

        // Info panel
        fileCountEl.textContent = state.file_count || 0;
        selectedPathEl.textContent = state.source_folder || 'No folder selected';
        if (state.source_folder) actionBtns.classList.remove('hidden');

        // Progress bar
        const percent = Math.round(state.progress * 100) + '%';
        progressPercent.textContent = percent;
        progressFill.style.width = percent;

        // File counter / stage detail
        if (state.processing && state.files_total > 0) {
            fileCounter.classList.remove('hidden');
            const left = state.files_total - state.files_done;
            const fileTag = state.current_file
                ? ` &nbsp;<span class="text-yellow-300 font-semibold">${state.current_file}</span>` : '';
            const detailTag = state.stage_detail
                ? `<div class="text-gray-400 text-xs mt-1 ml-6">${state.stage_detail}</div>` : '';
            fileCounter.innerHTML =
                `<i data-lucide="files" class="w-4 h-4 mr-2"></i>` +
                `Done <b>${state.files_done}</b> / <b>${state.files_total}</b>&nbsp;|&nbsp;` +
                `<b>${left}</b> left` + fileTag + detailTag;
        } else if (state.processing && state.stage_detail) {
            fileCounter.classList.remove('hidden');
            fileCounter.innerHTML =
                `<i data-lucide="loader-2" class="w-4 h-4 mr-2 animate-spin"></i>` + state.stage_detail;
        } else {
            fileCounter.classList.add('hidden');
        }

        // Status + stages
        updateStatusText(state);
        renderStages(state.current_stage);
        renderLogs(state.logs);

        // ── Count mode – decide what to show once prepare finishes ─────────
        if (currentMode === 'count' && !state.processing && !state.output_path) {

            // Case A: symbols already exist (or were just extracted) → show panel
            if (!countPanelPopulated && state.symbols_list && state.symbols_list.length > 0) {
                symbolsData = state.symbols_list;
                filesData = state.files_list || [];
                populateCountPanel(symbolsData, filesData);
                countPanelPopulated = true;

                // Case B: no symbols yet → show source picker (once), even if no auto-sources
            } else if (!sourcesShown && (!state.symbols_list || state.symbols_list.length === 0)) {
                sourcesShown = true;
                window._lastSymbolSources = state.symbol_sources || [];
                showSourcePicker(window._lastSymbolSources);
            }

            // Keep sources cached for Re-extract button
            if (state.symbol_sources) window._lastSymbolSources = state.symbol_sources;
        }

        // Success modal
        if (state.output_path && !state.processing && pollInterval) {
            modalOutputPath.textContent = state.output_path;
            successModal.classList.remove('hidden');
            clearInterval(pollInterval);
            pollInterval = null;
        }

        if (window.lucide) window.lucide.createIcons();

    } catch (err) {
        console.error('Poll error:', err);
    }
}

// =============================================================================
// SOURCE FILE PICKER  (choose which image to extract from)
// =============================================================================
function showSourcePicker(sources) {
    const modal = document.getElementById('sourcePickerModal');
    const list = document.getElementById('sourceList');
    if (!modal || !list) return;

    if (sources.length > 0) {
        list.innerHTML = sources.map((s, i) => {
            const autoPath = escHtml(s.path);
            const manualPath = escHtml(s.manual_path || s.path);
            // show a small badge when the two paths differ
            const badge = s.manual_path
                ? ` <span title="Auto: ${escHtml(s.path)}&#10;Manual: ${manualPath}"
                         style="font-size:.65rem;opacity:.55;margin-left:4px">(HR)</span>`
                : '';
            return `
            <label class="check-item" style="cursor:pointer">
                <input type="radio" name="srcPick"
                       value="${autoPath}"
                       data-manual-path="${manualPath}"
                       ${i === 0 ? 'checked' : ''}
                       style="accent-color:#f59e0b;width:16px;height:16px;cursor:pointer;flex-shrink:0" />
                <span>${escHtml(s.name)}${badge}</span>
            </label>`;
        }).join('');
    } else {
        list.innerHTML = '<div class="text-gray-500 text-sm py-2">No source images auto-detected. Use Browse to select manually.</div>';
    }

    modal.classList.remove('hidden');
}

// Open source picker from the Re-extract button (always usable from the count panel)
function showSourcePickerFromPanel() {
    const sources = window._lastSymbolSources || [];
    sourcesShown = false;  // allow re-opening
    showSourcePicker(sources);
}

// ── Browse for any image file ────────────────────────────────────────────────
async function browseForSymbolSource() {
    let filePath = null;

    // 1️⃣  Try server-side native dialog first (most reliable on Windows)
    try {
        const res = await fetch('/browse-file');
        const data = await res.json();
        filePath = data.file_path || null;
    } catch (_) { }

    // 2️⃣  Fallback: pywebview desktop API
    if (!filePath && window.pywebview && window.pywebview.api) {
        try { filePath = await window.pywebview.api.selectFile(); } catch (_) { }
    }

    if (!filePath) return;   // user cancelled

    // Inject a new radio option and select it
    const list = document.getElementById('sourceList');
    if (!list) return;
    const safePath = escHtml(filePath);
    const safeName = escHtml(filePath.split(/[\/\\]/).pop());
    const existingCustom = list.querySelector('input[data-custom]');
    if (existingCustom) existingCustom.closest('label').remove();

    const label = document.createElement('label');
    label.className = 'check-item';
    label.style.cursor = 'pointer';
    label.innerHTML = `
        <input type="radio" name="srcPick" value="${safePath}" data-custom="1" checked
               style="accent-color:#f59e0b;width:16px;height:16px;cursor:pointer;flex-shrink:0" />
        <span>📂 ${safeName}</span>`;
    list.prepend(label);
}

document.addEventListener('click', async (e) => {
    // Browse button inside source picker modal
    if (e.target.id === 'browseFileBtn' || e.target.closest('#browseFileBtn')) {
        await browseForSymbolSource();
        return;
    }

    if (e.target.id === 'extractBtn' || e.target.closest('#extractBtn') ||
        e.target.id === 'manualExtractBtn' || e.target.closest('#manualExtractBtn')) {

        const isManual = !!(e.target.id === 'manualExtractBtn' || e.target.closest('#manualExtractBtn'));
        const picked = document.querySelector('input[name="srcPick"]:checked');
        if (!picked) { alert('Please select a file or use Browse to pick one.'); return; }

        // ── Key change: Manual uses High_Resolution path, Auto uses Information_Box path ──
        const filePath = isManual
            ? (picked.dataset.manualPath || picked.value)   // High_Resolution.png
            : picked.value;                                  // Information_Box.png

        const modal = document.getElementById('sourcePickerModal');
        const spinner = document.getElementById(isManual ? 'manualSpinner' : 'extractSpinner');
        const btn = document.getElementById(isManual ? 'manualExtractBtn' : 'extractBtn');
        const autoBtn = document.getElementById('extractBtn');
        const manBtn = document.getElementById('manualExtractBtn');

        if (spinner) spinner.classList.remove('hidden');
        if (autoBtn) autoBtn.disabled = true;
        if (manBtn) manBtn.disabled = true;

        try {
            const res = await fetch('/extract-symbols', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_path: filePath, manual: isManual })
            });
            const data = await res.json();

            if (!res.ok || (!data.success && !data.fallback)) {
                throw new Error(data.error || 'Extraction failed');
            }

            modal.classList.add('hidden');

            if (data.fallback) {
                // Symbol Maker GUI opened – reset the guard so the poll loop
                // will show the count panel as soon as the window closes and
                // app_state["symbols_list"] is refreshed by the server thread.
                countPanelPopulated = false;
                sourcesShown = true;  // don't re-open source picker

                // Show a persistent non-blocking notice in the log area
                appendLog('✏️  Symbol Maker is open — draw your selections, then click 💾 Save All.  The panel will update automatically when you close the window.');

            } else {
                // Auto-extracted – refresh panel immediately
                symbolsData = data.symbols || [];
                // Update filesData if the server returned an updated list
                if (data.files_list) filesData = data.files_list;

                countPanelPopulated = false;
                populateCountPanel(symbolsData, filesData);
                countPanelPopulated = true;
            }
        } catch (err) {
            alert('Extraction error: ' + err.message);
        } finally {
            if (spinner) spinner.classList.add('hidden');
            if (autoBtn) autoBtn.disabled = false;
            if (manBtn) manBtn.disabled = false;
        }
    }

    if (e.target.id === 'cancelExtractBtn') {
        document.getElementById('sourcePickerModal').classList.add('hidden');
    }
});

// =============================================================================
// POPULATE COUNT PANEL  –  show symbol IMAGE only (no visible name label)
// =============================================================================
function populateCountPanel(symbols, files) {
    const symGrid = document.getElementById('symbolsGrid');
    const fileGrid = document.getElementById('filesGrid');
    if (!symGrid || !fileGrid) return;

    countPanel.classList.remove('hidden');

    // ── Symbols as image-only cards (name shown only as tooltip + on img-fail) ─
    if (symbols.length === 0) {
        symGrid.innerHTML = '<div class="text-gray-500 text-sm col-span-full py-4 text-center">No symbols found in symbols/ folder.</div>';
    } else {
        // Filter out legend_table from selectable symbols
        const selectable = symbols.filter(s => s.name !== 'legend_table');
        symGrid.innerHTML = selectable.map(s => `
            <label class="sym-card" title="${escHtml(s.name)}">
                <input type="checkbox" value="${escHtml(s.name)}"
                    onchange="updateSelCount('sym')" checked />
                <div class="sym-img-wrap">
                    <img src="/symbol-img/${encodeURIComponent(s.name)}"
                         alt="${escHtml(s.name)}"
                         onerror="this.style.display='none';this.nextElementSibling.style.display='block'"
                    />
                    <span class="sym-name-fallback" style="display:none">${escHtml(s.name)}</span>
                </div>
            </label>`).join('');
        updateSelCount('sym');
    }

    // ── Files as text checkboxes ────────────────────────────────────────────
    if (files.length === 0) {
        fileGrid.innerHTML = '<div class="text-gray-500 text-sm col-span-full py-4 text-center">No floor plan files found.</div>';
    } else {
        fileGrid.innerHTML = files.map(f => `
            <label class="check-item">
                <input type="checkbox" value="${escHtml(f.name)}"
                    onchange="updateSelCount('file')" checked />
                <span>${escHtml(f.name)}</span>
            </label>`).join('');
        updateSelCount('file');
    }

    if (window.lucide) window.lucide.createIcons();
}

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// =============================================================================
// APPEND A SINGLE LOG ENTRY  (non-blocking notice in the activity log)
// =============================================================================
function appendLog(msg) {
    if (!logContainer) return;
    const div = document.createElement('div');
    div.className = 'flex space-x-3';
    let colorClass = 'text-amber-300';
    if (msg.includes('✅') || msg.includes('✓')) colorClass = 'text-green-400';
    else if (msg.includes('❌') || msg.includes('✗')) colorClass = 'text-red-400';
    div.innerHTML = `<span class="flex-1 ${colorClass}">${msg}</span>`;
    logContainer.appendChild(div);
    logContainer.scrollTop = logContainer.scrollHeight;
}

// =============================================================================
// REFRESH SYMBOLS  – re-scan symbols/ folder without restarting
// =============================================================================
async function refreshSymbols() {
    try {
        const res = await fetch('/refresh-symbols');
        const data = await res.json();
        if (!res.ok || data.error) {
            appendLog('❌ Refresh failed: ' + (data.error || 'Unknown error'));
            return;
        }
        const syms = data.symbols || [];
        appendLog(`🔄 Refreshed – ${syms.length} symbol(s) found.`);
        if (syms.length > 0) {
            symbolsData = syms;
            countPanelPopulated = false;
            populateCountPanel(symbolsData, filesData);
            countPanelPopulated = true;
        }
    } catch (err) {
        appendLog('❌ Refresh error: ' + err.message);
    }
}

// =============================================================================
// STATUS TEXT
// =============================================================================
function updateStatusText(state) {
    const el = statusText;
    const map = {
        1: 'Todo / Setup',
        2: 'Images / Extract',
        3: 'AI Processing',
        4: 'Creating Excel Report',
        5: 'Symbol Counter'
    };
    if (state.processing) {
        const task = map[state.current_stage] || 'Processing…';
        el.className = 'flex items-center space-x-2';
        el.innerHTML = `<i data-lucide="loader-2" class="text-blue-400 w-5 h-5 animate-spin"></i><span class="text-white">Status: ${task}</span>`;
    } else if (state.output_path) {
        el.className = 'flex items-center space-x-2 text-green-400';
        el.innerHTML = `<i data-lucide="check-circle-2" class="w-5 h-5"></i><span>Status: Complete</span>`;
    } else if (currentMode === 'count' && state.symbols_list && state.symbols_list.length > 0 && !state.processing) {
        el.className = 'flex items-center space-x-2 text-amber-400';
        el.innerHTML = `<i data-lucide="scan-search" class="w-5 h-5"></i><span>Status: Select symbols &amp; files to count</span>`;
    } else {
        el.className = 'text-gray-400';
        el.textContent = 'Status: Idle';
    }
}

// =============================================================================
// RENDER STAGES  –  Fix #2: show only relevant stages per mode
// =============================================================================
function renderStages(currentStage) {
    // Pick which stage list to show
    let stages;
    if (currentMode === 'excel') stages = STAGES_EXCEL;
    else if (currentMode === 'count') stages = STAGES_COUNT;
    else stages = STAGES_ALL;     // idle / unknown

    const container = document.getElementById('stagesContainer');
    if (!container) return;
    container.innerHTML = '';

    const total = stages.length;

    stages.forEach((stage, idx) => {
        const isCompleted = currentStage > stage.id;
        const isActive = currentStage === stage.id;

        let wrapperClass = 'flex items-center space-x-3 p-3 rounded-lg transition-all duration-300 ';
        if (isActive) wrapperClass += 'bg-slate-700/50 shadow-lg scale-105';
        else if (isCompleted) wrapperClass += 'bg-slate-700/30';
        else wrapperClass += 'bg-slate-700/10';

        let iconBgStyle;
        if (isCompleted) iconBgStyle = 'background-color:#10b981;';
        else if (isActive) iconBgStyle = `background:linear-gradient(135deg,${stage.color},#8b5cf6);`;
        else iconBgStyle = 'background-color:#334155;';

        const div = document.createElement('div');
        div.className = wrapperClass;

        const iconEl = document.createElement('div');
        iconEl.className = 'w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0';
        iconEl.style.cssText = iconBgStyle;

        const iconTag = document.createElement('i');
        iconTag.setAttribute('data-lucide', isCompleted ? 'check-circle-2' : stage.icon);
        iconTag.className = 'text-white w-5 h-5';
        iconEl.appendChild(iconTag);

        let timeText = '';
        if (isCompleted && stageStartTimes[stage.id] && stageStartTimes[stages[idx + 1]?.id]) {
            const elapsed = Math.floor((stageStartTimes[stages[idx + 1].id] - stageStartTimes[stage.id]) / 1000);
            timeText = ` (${elapsed}s)`;
        } else if (isActive && stageStartTimes[stage.id]) {
            const elapsed = Math.floor((Date.now() - stageStartTimes[stage.id]) / 1000);
            timeText = ` (${elapsed}s)`;
        }

        const textEl = document.createElement('div');
        textEl.className = 'flex-1 min-w-0';
        textEl.innerHTML = `
            <div class="text-sm font-medium ${isActive || isCompleted ? 'text-white' : 'text-gray-400'} truncate">
                ${stage.name}${timeText}
            </div>
            <div class="text-xs text-gray-500">Step ${idx + 1}/${total}</div>`;

        div.appendChild(iconEl);
        div.appendChild(textEl);
        container.appendChild(div);
    });
}

// =============================================================================
// RENDER LOGS
// =============================================================================
function renderLogs(logs) {
    if (!logContainer) return;
    if (!logs || logs.length === 0) {
        logContainer.innerHTML = `
            <div class="flex flex-col items-center justify-center text-gray-500 py-8">
                <i data-lucide="alert-circle" class="w-12 h-12 opacity-50 mb-3"></i>
                <p>No activity yet. Select a folder to begin.</p>
            </div>`;
        return;
    }

    logContainer.innerHTML = '';
    logs.forEach(log => {
        const div = document.createElement('div');
        div.className = 'flex space-x-3';
        let colorClass = 'text-gray-300';
        if (log.includes('✅') || log.includes('✓') || log.includes('Success')) colorClass = 'text-green-400';
        else if (log.includes('❌') || log.includes('✗') || log.includes('Error')) colorClass = 'text-red-400';
        else if (log.includes('⚠') || log.includes('SKIP')) colorClass = 'text-yellow-400';
        else if (log.includes('📂') || log.includes('Stage')) colorClass = 'text-blue-300';
        else if (log.includes('🔍') || log.includes('🔢')) colorClass = 'text-purple-300';
        else if (log.includes('💾')) colorClass = 'text-cyan-400';

        const match = log.match(/^\[(.*?)\] (.*)/);
        if (match) {
            div.innerHTML = `
                <span class="text-gray-500 text-xs mt-1 shrink-0">[${match[1]}]</span>
                <span class="flex-1 ${colorClass}">${match[2]}</span>`;
        } else {
            div.innerHTML = `<span class="flex-1 ${colorClass}">${log}</span>`;
        }
        logContainer.appendChild(div);
    });
    logContainer.scrollTop = logContainer.scrollHeight;
}

// =============================================================================
// INITIAL LOAD
// =============================================================================
updateUI();