/**
 * dashboard.js — Live Dashboard Logic
 * =====================================
 * Handles auto-refreshing stats, gauge ring animation,
 * distribution donut chart, and timeline chart.
 * Fetches data from /api/stats every 5 seconds.
 */

// ──────────────────────────────────────────────
// State
// ──────────────────────────────────────────────
let distributionChart = null;
let timelineChart = null;
const timelineData = { labels: [], ok: [], nearFull: [], overflow: [] };
const MAX_TIMELINE_POINTS = 20;


// ──────────────────────────────────────────────
// Gauge Ring
// ──────────────────────────────────────────────
function updateGauge(fillPercent) {
    const el    = document.getElementById('gauge-fill');
    const label = document.getElementById('gauge-value');
    if (!el || !label) return;

    const circumference = 2 * Math.PI * 58; // r=58
    const pct = Math.max(0, Math.min(100, fillPercent));
    const offset = circumference - (circumference * pct / 100);

    el.style.strokeDasharray  = circumference;
    el.style.strokeDashoffset = offset;

    // Color based on fill level
    if (pct >= 80)      el.style.stroke = 'var(--accent-red)';
    else if (pct >= 65) el.style.stroke = 'var(--accent-orange)';
    else                el.style.stroke = 'var(--accent-green)';

    label.textContent = `${Math.round(pct)}%`;
}


// ──────────────────────────────────────────────
// Distribution Donut Chart
// ──────────────────────────────────────────────
function initDistributionChart() {
    const ctx = document.getElementById('chart-distribution');
    if (!ctx) return;

    distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['OK', 'Near Full', 'Overflow'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(63, 185, 80, 0.8)',
                    'rgba(210, 153, 34, 0.8)',
                    'rgba(248, 81, 73, 0.8)',
                ],
                borderColor: [
                    'rgba(63, 185, 80, 1)',
                    'rgba(210, 153, 34, 1)',
                    'rgba(248, 81, 73, 1)',
                ],
                borderWidth: 2,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                },
            },
            animation: {
                animateRotate: true,
                duration: 800,
            }
        }
    });
}

function updateDistributionChart(ok, nearFull, overflow) {
    if (!distributionChart) return;
    distributionChart.data.datasets[0].data = [ok, nearFull, overflow];
    distributionChart.update('none');
}


// ──────────────────────────────────────────────
// Timeline Chart
// ──────────────────────────────────────────────
function initTimelineChart() {
    const ctx = document.getElementById('chart-timeline');
    if (!ctx) return;

    timelineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'OK',
                    data: [],
                    borderColor: 'rgba(63, 185, 80, 0.8)',
                    backgroundColor: 'rgba(63, 185, 80, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2,
                },
                {
                    label: 'Near Full',
                    data: [],
                    borderColor: 'rgba(210, 153, 34, 0.8)',
                    backgroundColor: 'rgba(210, 153, 34, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2,
                },
                {
                    label: 'Overflow',
                    data: [],
                    borderColor: 'rgba(248, 81, 73, 0.8)',
                    backgroundColor: 'rgba(248, 81, 73, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2,
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            scales: {
                x: {
                    display: true,
                    grid: { display: false },
                    ticks: { maxTicksLimit: 6 },
                },
                y: {
                    display: true,
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(48, 54, 61, 0.3)',
                    },
                    ticks: { stepSize: 1, maxTicksLimit: 5 },
                },
            },
            plugins: {
                legend: {
                    position: 'bottom',
                },
            },
            animation: {
                duration: 400,
            }
        }
    });
}

function updateTimelineChart(ok, nearFull, overflow) {
    if (!timelineChart) return;

    const now = new Date().toLocaleTimeString('en-US', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });

    timelineData.labels.push(now);
    timelineData.ok.push(ok);
    timelineData.nearFull.push(nearFull);
    timelineData.overflow.push(overflow);

    // Trim to max points
    if (timelineData.labels.length > MAX_TIMELINE_POINTS) {
        timelineData.labels.shift();
        timelineData.ok.shift();
        timelineData.nearFull.shift();
        timelineData.overflow.shift();
    }

    timelineChart.data.labels = [...timelineData.labels];
    timelineChart.data.datasets[0].data = [...timelineData.ok];
    timelineChart.data.datasets[1].data = [...timelineData.nearFull];
    timelineChart.data.datasets[2].data = [...timelineData.overflow];
    timelineChart.update('none');
}


// ──────────────────────────────────────────────
// Overflow card flash animation
// ──────────────────────────────────────────────
function flashOverflowCard(overflowCount) {
    const card = document.getElementById('card-overflow');
    if (!card) return;

    if (overflowCount > 0) {
        card.classList.add('alert-flash');
    } else {
        card.classList.remove('alert-flash');
    }
}


// ──────────────────────────────────────────────
// Fetch and update dashboard
// ──────────────────────────────────────────────
async function refreshDashboard() {
    try {
        const res = await fetch('/api/stats');
        if (!res.ok) throw new Error('Failed to fetch stats');
        const data = await res.json();

        // Update stat cards with animated counters
        const totalEl    = document.getElementById('stat-total');
        const overflowEl = document.getElementById('stat-overflow');
        const nearFullEl = document.getElementById('stat-near-full');
        const okEl       = document.getElementById('stat-ok');

        if (totalEl)    animateCounter(totalEl, data.total_bins);
        if (overflowEl) animateCounter(overflowEl, data.overflow_count);
        if (nearFullEl) animateCounter(nearFullEl, data.near_full_count);
        if (okEl)       animateCounter(okEl, data.ok_count);

        // FPS
        const fpsEl = document.getElementById('fps-display');
        if (fpsEl) fpsEl.textContent = `FPS: ${data.fps || 0}`;

        // Flash effect on overflow
        flashOverflowCard(data.overflow_count);

        // Calculate avg fill from events
        let avgFill = 0;
        if (data.events && data.events.length > 0) {
            const fills = data.events.map(e => parseFloat(e.fill_percent) || 0);
            avgFill = fills.reduce((a, b) => a + b, 0) / fills.length;
        }
        updateGauge(avgFill);

        // Update charts
        updateDistributionChart(data.ok_count, data.near_full_count, data.overflow_count);
        updateTimelineChart(data.ok_count, data.near_full_count, data.overflow_count);

        // Update events table
        if (data.events) {
            populateEventsTable('events-tbody', data.events.reverse());
            const countEl = document.getElementById('events-count');
            if (countEl) countEl.textContent = data.events.length;
        }

    } catch (err) {
        console.error('Dashboard refresh failed:', err);
    }
}


// ──────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initDistributionChart();
    initTimelineChart();

    // Initial refresh
    refreshDashboard();

    // Auto-refresh every 5 seconds
    setInterval(refreshDashboard, 5000);
});
