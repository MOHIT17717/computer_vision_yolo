/**
 * app.js — Shared Application Logic
 * ===================================
 * Handles navigation, scroll effects, reveal animations,
 * and system status checks used across all pages.
 */

// ──────────────────────────────────────────────
// Navbar scroll effect
// ──────────────────────────────────────────────
(function initNavbar() {
    const navbar = document.getElementById('navbar');
    if (!navbar) return;

    window.addEventListener('scroll', () => {
        if (window.scrollY > 30) {
            navbar.classList.add('scrolled');
        } else {
            // Only remove on pages that don't start scrolled (home page)
            if (document.querySelector('.hero')) {
                navbar.classList.remove('scrolled');
            }
        }
    });

    // Trigger on load
    if (window.scrollY > 30) navbar.classList.add('scrolled');
})();


// ──────────────────────────────────────────────
// Mobile nav toggle
// ──────────────────────────────────────────────
(function initNavToggle() {
    const toggle = document.getElementById('nav-toggle');
    const links  = document.getElementById('nav-links');
    if (!toggle || !links) return;

    toggle.addEventListener('click', () => {
        links.classList.toggle('open');
    });

    // Close on link click
    links.querySelectorAll('a').forEach(a => {
        a.addEventListener('click', () => links.classList.remove('open'));
    });
})();


// ──────────────────────────────────────────────
// Intersection Observer — reveal animations
// ──────────────────────────────────────────────
(function initRevealAnimations() {
    const reveals = document.querySelectorAll('.reveal');
    if (!reveals.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    reveals.forEach(el => observer.observe(el));
})();


// ──────────────────────────────────────────────
// System status check
// ──────────────────────────────────────────────
async function checkSystemStatus() {
    const dot  = document.getElementById('system-status-dot');
    const text = document.getElementById('system-status-text');
    if (!dot || !text) return;

    try {
        const res = await fetch('/api/stats');
        if (res.ok) {
            dot.classList.remove('offline');
            dot.style.background = 'var(--accent-green)';
            text.textContent = 'System Online';
        } else {
            throw new Error('Bad response');
        }
    } catch {
        dot.classList.add('offline');
        dot.style.background = 'var(--text-tertiary)';
        text.textContent = 'Offline';
    }
}

// Check status every 10 seconds
checkSystemStatus();
setInterval(checkSystemStatus, 10000);


// ──────────────────────────────────────────────
// Animated counter (used by multiple pages)
// ──────────────────────────────────────────────
function animateCounter(element, targetValue, duration = 800) {
    const start = parseInt(element.textContent) || 0;
    const range = targetValue - start;
    if (range === 0) return;

    const startTime = performance.now();

    function tick(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const ease = 1 - Math.pow(1 - progress, 3);
        element.textContent = Math.round(start + range * ease);

        if (progress < 1) {
            requestAnimationFrame(tick);
        }
    }

    requestAnimationFrame(tick);
}


// ──────────────────────────────────────────────
// Chart.js default theme (dark mode)
// ──────────────────────────────────────────────
if (typeof Chart !== 'undefined') {
    Chart.defaults.color = '#8b949e';
    Chart.defaults.borderColor = 'rgba(48, 54, 61, 0.4)';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
    Chart.defaults.plugins.legend.labels.padding = 16;
    Chart.defaults.plugins.tooltip.backgroundColor = '#161b22';
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(48, 54, 61, 0.6)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
}


// ──────────────────────────────────────────────
// Utility: create status pill HTML
// ──────────────────────────────────────────────
function statusPillHTML(status) {
    const map = {
        'OVERFLOW':  { cls: 'overflow',  label: 'Overflow' },
        'NEAR_FULL': { cls: 'near-full', label: 'Near Full' },
        'OK':        { cls: 'ok',        label: 'OK' },
    };
    const info = map[status] || map['OK'];
    return `<span class="status-pill ${info.cls}">${info.label}</span>`;
}


// ──────────────────────────────────────────────
// Utility: populate events table
// ──────────────────────────────────────────────
function populateEventsTable(tbodyId, events, maxRows = 50) {
    const tbody = document.getElementById(tbodyId);
    if (!tbody || !events) return;

    if (events.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:2rem; color:var(--text-tertiary);">No events recorded yet</td></tr>`;
        return;
    }

    const rows = events.slice(0, maxRows).map(e => `
        <tr>
            <td class="timestamp">${e.timestamp || ''}</td>
            <td>#${e.bin_id || ''}</td>
            <td>${e.location || ''}</td>
            <td>${e.fill_percent || ''}%</td>
            <td>${statusPillHTML(e.status)}</td>
            <td class="text-mono">${e.confidence || ''}</td>
        </tr>
    `).join('');

    tbody.innerHTML = rows;
}
