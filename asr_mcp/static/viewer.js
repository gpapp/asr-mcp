const SPEAKER_COLORS = [
    '#007bff','#28a745','#dc3545','#fd7e14','#6f42c1',
    '#20c997','#e83e8c','#17a2b8','#ffc658','#6610f2',
    '#2d6a4f','#d63384','#0dcaf0','#198754','#cf222e',
];

function getSpeakerColor(name, allSpeakers) {
    const idx = allSpeakers.indexOf(name);
    return SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
}

function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 100);
    return m + ':' + String(s).padStart(2, '0') + '.' + String(ms).padStart(2, '0');
}

function fillSpeakerBars(container, segments, audioDur, allSpeakers) {
    container.innerHTML = '';
    if (!audioDur || !segments || !segments.length) return;
    const sorted = [...segments].sort((a, b) => a.start - b.start);
    for (const spk of allSpeakers) {
        const row = document.createElement('div');
        row.className = 'bar-row';
        for (const seg of sorted) {
            if (seg.speaker !== spk) continue;
            const act = document.createElement('div');
            act.className = 'bar-act';
            act.style.left = (seg.start / audioDur * 100).toFixed(4) + '%';
            act.style.width = ((seg.end - seg.start) / audioDur * 100).toFixed(4) + '%';
            act.style.background = getSpeakerColor(spk, allSpeakers);
            act.title = spk + '  ' + formatTime(seg.start) + ' - ' + formatTime(seg.end);
            row.appendChild(act);
        }
        const name = document.createElement('div');
        name.className = 'bar-name';
        name.textContent = spk;
        row.appendChild(name);
        container.appendChild(row);
    }
}

function buildTimeAxis(audioDur) {
    const timeAxis = document.createElement('div');
    timeAxis.className = 'time-axis';
    const numTicks = Math.min(10, Math.max(2, Math.floor(audioDur / 10)));
    for (let i = 0; i <= numTicks; i++) {
        const tick = document.createElement('span');
        tick.textContent = formatTime(audioDur * i / numTicks);
        timeAxis.appendChild(tick);
    }
    return timeAxis;
}

function renderBlocks(pane, results) {
    for (const r of results) {
        const b = document.createElement('div');
        b.className = 'tp-block';
        b.dataset.start = r.start || 0;
        b.dataset.end = r.end || 0;
        const head = document.createElement('div');
        head.className = 'tp-head';
        head.textContent = '[' + (r.speaker || '?') + '] ' +
            formatTime(r.start || 0) + ' – ' + formatTime(r.end || 0);
        const txt = document.createElement('div');
        txt.className = 'tp-text';
        txt.textContent = r.text || '';
        b.appendChild(head);
        b.appendChild(txt);
        pane.appendChild(b);
    }
}

function seekPane(pane, caret, audioDur, clientX, layer) {
    const rect = layer.getBoundingClientRect();
    const frac = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    const t = frac * audioDur;
    caret.style.display = 'block';
    caret.style.left = (frac * 100).toFixed(2) + '%';
    const blocks = pane.querySelectorAll('.tp-block');
    let target = null, prev = null;
    for (const b of blocks) {
        const s = parseFloat(b.dataset.start) || 0;
        const en = parseFloat(b.dataset.end) || 0;
        if (en > s && t >= s && t < en) { target = b; break; }
        if (s > t) { target = prev; break; }
        prev = b;
    }
    if (!target) target = prev || blocks[0];
    if (!target) return;
    const bs = parseFloat(target.dataset.start) || 0;
    const be = parseFloat(target.dataset.end) || 0;
    const inner = be > bs ? Math.min(Math.max((t - bs) / (be - bs), 0), 1) : 0;
    const paneRect = pane.getBoundingClientRect();
    const blockRect = target.getBoundingClientRect();
    const pointY = blockRect.top + inner * blockRect.height;
    const delta = pointY - (paneRect.top + pane.clientHeight / 2);
    pane.scrollTop = Math.min(
        Math.max(pane.scrollTop + delta, 0),
        pane.scrollHeight - pane.clientHeight
    );
}

function renderTranscriptViewer(result, container) {
    const results = (result.results || []).filter(r => r && (r.text || '').trim());
    const segments = result.segments || [];
    const audioDur = result.audio_duration_sec ||
        (segments.length ? Math.max(...segments.map(s => s.end)) :
            (results.length ? Math.max(...results.map(r => r.end || 0)) : 0));
    if (!audioDur) return false;
    const source = segments.length
        ? segments
        : results.map(r => ({ start: r.start || 0, end: r.end || 0, speaker: r.speaker || 'Speaker 1' }));
    if (!source.length) return false;
    const allSpeakers = [...new Set(source.map(s => s.speaker))].sort();

    if (result._fileName) {
        const fileLabel = document.createElement('div');
        fileLabel.className = 'file-label';
        fileLabel.textContent = result._fileName + ' - ' + allSpeakers.length +
            ' speaker(s), ' + formatTime(audioDur);
        container.appendChild(fileLabel);
    }

    const viewer = document.createElement('div');
    viewer.className = 'tv-viewer';

    const bars = document.createElement('div');
    bars.className = 'tv-bars';
    const stack = document.createElement('div');
    stack.style.position = 'relative';
    fillSpeakerBars(stack, source, audioDur, allSpeakers);

    const layer = document.createElement('div');
    layer.style.cssText = 'position:absolute;top:0;bottom:0;left:0;right:0;cursor:pointer;';
    const caret = document.createElement('div');
    caret.style.cssText = 'display:none;position:absolute;top:0;bottom:0;left:0;width:2px;background:#ff4444;z-index:10;';
    layer.appendChild(caret);
    stack.appendChild(layer);
    bars.appendChild(stack);
    bars.appendChild(buildTimeAxis(audioDur));
    viewer.appendChild(bars);

    const pane = document.createElement('div');
    pane.className = 'tv-text';
    if (results.length) {
        renderBlocks(pane, results);
        layer.addEventListener('click', (e) => seekPane(pane, caret, audioDur, e.clientX, layer));
    } else {
        const note = document.createElement('div');
        note.style.cssText = 'color:#999;font-size:0.85rem;';
        note.textContent = 'No transcription text stored for this entry.';
        pane.appendChild(note);
    }
    viewer.appendChild(pane);
    container.appendChild(viewer);
    return true;
}
