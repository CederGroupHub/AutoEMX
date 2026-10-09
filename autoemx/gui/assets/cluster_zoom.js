// Zoom of the clustering plot, as in matplotlib: in 3D, the mouse wheel scales the axis ranges around the
// data point nearest to the mouse pointer (which stays under it) and right-drag shifts them; in ternary, the mouse wheel scales the shown
// triangle around its centre.
// Double-click goes back to the view chosen in the "Zoom on" menu. The view set with Plotly's own tools (camera
// rotated, zoomed or panned) is kept when Dash redraws the plot, e.g. when the toolbar mode is changed, but
// not when another plot or "Zoom on" choice is shown (another uirevision). (Plotly only moves the 3D camera, which
// cannot zoom onto a small cluster, and has no wheel zoom in ternary plots. 2D plots use Plotly's zoom.)
// Dash serves the scripts of the assets folder automatically.
(function () {
    const LO = -3, HI = 103;  // limits of the 3D axes, percent (as _WALL_GAP_3D in plots.py)
    const MIN_SPAN = 0.05;    // percent
    const WHEEL = 0.0015;     // zoom speed
    const AXES = ['x', 'y', 'z'];
    const TERNARY_AXES = ['a', 'b', 'c'];

    function ticks(lo, hi, n) {  // as _ticks in plots.py
        lo = Math.max(lo, 0); hi = Math.min(hi, 100); n = n || 6;
        const steps = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 25];
        const step = steps.find(s => (hi - lo) / s <= n) || 25;
        const out = [];
        for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) { out.push(Math.round(v * 100) / 100); }
        return out;
    }

    const is3d = gd => !!(gd.layout && gd.layout.scene && gd.layout.scene.xaxis && gd.layout.scene.xaxis.range);
    const isTernary = gd => !!(gd.layout && gd.layout.ternary);

    function current(gd) {
        if (is3d(gd)) { return {kind: '3d', v: AXES.map(a => gd.layout.scene[a + 'axis'].range.slice())}; }
        if (isTernary(gd)) { return {kind: 'ternary', v: TERNARY_AXES.map(a => gd.layout.ternary[a + 'axis'].min || 0)}; }
        return null;
    }

    // Zoom state of a graph, reset when the plot is redrawn with a new uirevision (new zoom choice, axes...)
    function state(gd) {
        const rev = gd.layout.uirevision;
        if (!gd._autoemxZoom || gd._autoemxZoom.rev !== rev) {
            gd._autoemxZoom = {rev: rev, base: current(gd), user: null, camera: null, busy: false};
        }
        return gd._autoemxZoom;
    }

    function sameCamera(a, b) {
        const vals = c => ['eye', 'center', 'up'].flatMap(k => ['x', 'y', 'z'].map(x => (c[k] || {})[x] || 0));
        const va = vals(a), vb = vals(b);
        return va.every((v, i) => Math.abs(v - vb[i]) < 1e-6);
    }

    function update(view) {
        const u = {};
        if (view.kind === '3d') {
            AXES.forEach((a, i) => {
                const r = view.v[i];
                u[`scene.${a}axis.range`] = r;
                u[`scene.${a}axis.tickvals`] = ticks(Math.min(r[0], r[1]), Math.max(r[0], r[1]));
            });
        } else {
            TERNARY_AXES.forEach((a, i) => { u[`ternary.${a}axis.min`] = view.v[i]; });
        }
        return u;
    }

    function apply(gd, view, isUser) {
        const st = state(gd);
        st.user = isUser ? view : null;
        st.busy = true;
        Promise.resolve(Plotly.relayout(gd, update(view))).finally(() => { st.busy = false; });
    }

    // 3D axis range kept within the axis limits, with the same orientation (x and y are reversed)
    function clampRange(r, lo, hi) {
        const rev = r[0] > r[1];
        let a = Math.min(r[0], r[1]), b = Math.max(r[0], r[1]);
        const span = Math.min(Math.max(b - a, MIN_SPAN), hi - lo);
        const c = Math.min(Math.max((a + b) / 2, lo + span / 2), hi - span / 2);
        a = c - span / 2; b = c + span / 2;
        return rev ? [b, a] : [a, b];
    }

    // 3D camera in scene coordinates, where the axis box spans ±aspect ratio / 2 (as Plotly draws it)
    function camera(gd) {
        const scene = gd._fullLayout.scene, cam = scene.camera, asp = scene.aspectratio;
        const sub = (p, q) => p.map((c, i) => c - q[i]);
        const cross = (p, q) => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
        const norm = p => { const n = Math.hypot(...p) || 1; return p.map(c => c / n); };
        const eye = [cam.eye.x, cam.eye.y, cam.eye.z], center = [cam.center.x, cam.center.y, cam.center.z];
        const forward = norm(sub(center, eye));
        const right = norm(cross(forward, [cam.up.x, cam.up.y, cam.up.z]));
        return {eye: eye, center: center, forward: forward, right: right, up: cross(right, forward),
                dist: Math.hypot(...sub(center, eye)), aspect: [asp.x, asp.y, asp.z],
                tan: Math.tan(Math.PI / 8)};  // field of view 45°
    }

    const NEAR_PX = 60;  // a data point nearer than this to the pointer is the centre of the wheel zoom

    // Screen position (client pixels) of a point in data coordinates
    function project(c, view, box, v) {
        const half = box.height / 2;
        const d = view.v.map((r, i) => ((v[i] - r[0]) / (r[1] - r[0]) - 0.5) * c.aspect[i] - c.eye[i]);
        const dot = q => d.reduce((s, x, i) => s + x * q[i], 0);
        const depth = dot(c.forward);
        return [box.x + box.width / 2 + dot(c.right) / (depth * c.tan) * half,
                box.y + half - dot(c.up) / (depth * c.tan) * half];
    }

    // Centre of the wheel zoom: the shown data point nearest to the pointer, else the point under the pointer
    // in the plane through the camera centre facing the camera
    function pointAt(gd, view, clientX, clientY, sceneEl) {
        const c = camera(gd), box = sceneEl.getBoundingClientRect(), half = box.height / 2;
        let best = null, bestDist = NEAR_PX;
        for (const tr of gd._fullData || []) {
            if (tr.type !== 'scatter3d' || tr.visible !== true || !String(tr.mode).includes('markers')) { continue; }
            for (let k = 0; k < (tr.x || []).length; k++) {
                const v = [tr.x[k], tr.y[k], tr.z[k]];
                const inside = v.every((x, i) => x >= Math.min(...view.v[i]) && x <= Math.max(...view.v[i]));
                if (!inside) { continue; }
                const [px, py] = project(c, view, box, v);
                const dist = Math.hypot(px - clientX, py - clientY);
                if (dist < bestDist) { best = v; bestDist = dist; }
            }
        }
        if (best) { return best; }
        const sx = (clientX - box.x - box.width / 2) / half, sy = -(clientY - box.y - half) / half;
        const ray = c.forward.map((f, i) => f + (sx * c.right[i] + sy * c.up[i]) * c.tan);
        const along = ray.reduce((s, r, i) => s + r * c.forward[i], 0);
        const t = c.dist / along;
        return view.v.map((r, i) => {
            const u = (c.eye[i] + t * ray[i]) / c.aspect[i] + 0.5;  // 0-1 along the axis box
            return r[0] + u * (r[1] - r[0]);
        });
    }

    // Zoomed by *factor* around *anchor* (3D: data coordinates, which stay under the pointer)
    function zoomed(view, factor, anchor) {
        if (view.kind === '3d') {
            return {kind: '3d', v: view.v.map((r, i) => {
                const a = anchor ? anchor[i] : (r[0] + r[1]) / 2;
                return clampRange([a + (r[0] - a) * factor, a + (r[1] - a) * factor], LO, HI);
            })};
        }
        // Ternary: the shown triangle has the side 100 - sum(minima); scale it around its centre
        const side = 100 - view.v.reduce((s, m) => s + m, 0);
        const newSide = Math.min(Math.max(side * factor, 0.5), 100);
        let mins = view.v.map(m => Math.max(0, m + side / 3 - newSide / 3));
        const sum = mins.reduce((s, m) => s + m, 0);
        if (sum > 99.5) { mins = mins.map(m => m * 99.5 / sum); }
        return {kind: 'ternary', v: mins};
    }

    // 3D: shift of the axis ranges moving the plot content by (dx, dy) pixels on screen
    function panned(gd, start, dx, dy, heightPx) {
        const c = camera(gd);
        const k = 2 * c.dist * c.tan / heightPx;  // scene units per pixel, at the camera centre
        const move = c.right.map((r, i) => (dx * r - dy * c.up[i]) * k);
        return {kind: '3d', v: start.v.map((r, i) => {
            const shift = move[i] / c.aspect[i] * (r[1] - r[0]);
            return clampRange([r[0] - shift, r[1] - shift], LO, HI);
        })};
    }

    function inPlotArea(gd, target) {
        if (!target.closest || target.closest('.legend, .modebar')) { return false; }
        return is3d(gd) ? !!target.closest('.gl-container') : isTernary(gd);
    }

    function attach(gd) {
        gd.addEventListener('wheel', ev => {
            if (!inPlotArea(gd, ev.target)) { return; }
            const view = current(gd);
            if (!view) { return; }
            ev.preventDefault();
            ev.stopPropagation();  // capture phase: Plotly's own 3D (camera) zoom does not run
            state(gd);
            const delta = ev.deltaMode === 1 ? ev.deltaY * 30 : ev.deltaY;
            const sceneEl = view.kind === '3d' ? ev.target.closest('.gl-container > div') : null;
            const anchor = sceneEl ? pointAt(gd, view, ev.clientX, ev.clientY, sceneEl) : null;
            apply(gd, zoomed(view, Math.exp(delta * WHEEL), anchor), true);
        }, {capture: true, passive: false});

        gd.addEventListener('mousedown', ev => {
            if (ev.button !== 2 || !is3d(gd) || !inPlotArea(gd, ev.target)) { return; }
            ev.preventDefault();
            ev.stopPropagation();  // Plotly's right-drag moves the camera instead
            state(gd);
            const start = current(gd), x0 = ev.clientX, y0 = ev.clientY;
            const height = (ev.target.closest('.gl-container > div') || gd).clientHeight;
            let pending = null;
            const move = e => {
                pending = [e.clientX - x0, e.clientY - y0];
                requestAnimationFrame(() => {
                    if (!pending) { return; }
                    const [dx, dy] = pending; pending = null;
                    apply(gd, panned(gd, start, dx, dy, height), true);
                });
            };
            const up = () => { window.removeEventListener('mousemove', move, true); window.removeEventListener('mouseup', up, true); };
            window.addEventListener('mousemove', move, true);
            window.addEventListener('mouseup', up, true);
        }, true);

        gd.addEventListener('contextmenu', ev => { if (is3d(gd) && inPlotArea(gd, ev.target)) { ev.preventDefault(); } }, true);

        // Double-click: back to the "Zoom on" view (3D here; in ternary, Plotly reports it as plotly_doubleclick)
        gd.addEventListener('dblclick', ev => {
            if (!is3d(gd) || !inPlotArea(gd, ev.target)) { return; }
            const st = state(gd);
            if (!st.base) { return; }
            ev.stopPropagation();
            apply(gd, st.base, false);
        }, true);

        // Plotly's own zoom (e.g. box zoom in ternary) becomes the user's zoom; the 3D camera moved with
        // Plotly's tools is remembered for this uirevision
        gd.on('plotly_relayout', ev => {
            const st = state(gd);
            if (st.busy) { return; }
            if (st.user) { st.user = current(gd); }
            // 'scene.camera' when moved, 'scene.camera.eye' etc. from the toolbar's reset buttons
            if (ev && Object.keys(ev).some(k => k.startsWith('scene.camera')) && is3d(gd)) {
                const cam = gd._fullLayout.scene.camera;
                st.camera = JSON.parse(JSON.stringify({eye: cam.eye, center: cam.center, up: cam.up}));
            }
        });
        gd.on('plotly_doubleclick', () => {
            const st = state(gd);
            st.user = null;
            // After Plotly's own reset (to the whole triangle in ternary)
            if (st.base) { setTimeout(() => apply(gd, st.base, false), 50); }
        });
        // A redraw by Dash (e.g. a point clicked, or the toolbar mode changed) resets the axes and the 3D
        // camera: set them again
        gd.on('plotly_react', () => {
            const st = state(gd);
            if (st.busy) { return; }
            const u = {};
            const now = current(gd);
            if (st.user && now && now.kind === st.user.kind && JSON.stringify(now.v) !== JSON.stringify(st.user.v)) {
                Object.assign(u, update(st.user));
            }
            if (is3d(gd) && st.camera && !sameCamera(gd._fullLayout.scene.camera, st.camera)) {
                u['scene.camera'] = st.camera;
            }
            if (!Object.keys(u).length) { return; }
            st.busy = true;
            Promise.resolve(Plotly.relayout(gd, u)).finally(() => { st.busy = false; });
        });
        gd._autoemxZoomAttached = true;
    }

    // The graph is created by Dash after this script runs (and again if the page is redrawn)
    setInterval(() => {
        const gd = document.querySelector('#cluster-graph .js-plotly-plot');
        if (gd && gd.on && !gd._autoemxZoomAttached) { attach(gd); }
    }, 500);
})();
