/**
 * analytics.js — Analytics Page Logic
 * =====================================
 * Fetches events data from /api/stats and renders
 * four analytical charts plus summary metrics.
 */

// ──────────────────────────────────────────────
// Fetch data and initialize
// ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('/api/stats');
        if (!res.ok) throw new Error('Failed to fetch');
        const data = await res.json();

        const events = data.events || [];

        // Summary metrics
        updateSummaryMetrics(events);

        // Charts
        createStatusBreakdownChart(events);
        createFillDistributionChart(events);
        createDetectionsTimeChart(events);
        createConfidenceChart(events);

        // Events table
        populateEventsTable('analytics-events-tbody', events, 100);
        const countEl = document.getElementById('analytics-events-count');
        if (countEl) countEl.textContent = events.length;

    } catch (err) {
        console.error('Analytics data load failed:', err);
    }
});


// ──────────────────────────────────────────────
// Summary Metrics
// ──────────────────────────────────────────────
function updateSummaryMetrics(events) {
    const totalEl    = document.getElementById('analytics-total');
    const overflowEl = document.getElementById('analytics-overflow');
    const avgConfEl  = document.getElementById('analytics-avg-conf');
    const avgFillEl  = document.getElementById('analytics-avg-fill');

    if (totalEl) totalEl.textContent = events.length;

    const overflowCount = events.filter(e => e.status === 'OVERFLOW').length;
    if (overflowEl) overflowEl.textContent = overflowCount;

    if (events.length > 0) {
        const avgConf = events.reduce((s, e) => s + (parseFloat(e.confidence) || 0), 0) / events.length;
        const avgFill = events.reduce((s, e) => s + (parseFloat(e.fill_percent) || 0), 0) / events.length;

        if (avgConfEl) avgConfEl.textContent = `${(avgConf * 100).toFixed(0)}%`;
        if (avgFillEl) avgFillEl.textContent = `${avgFill.toFixed(0)}%`;
    }
}


// ──────────────────────────────────────────────
// Chart 1: Status Breakdown (Pie)
// ──────────────────────────────────────────────
function createStatusBreakdownChart(events) {
    const ctx = document.getElementById('chart-status-breakdown');
    if (!ctx) return;

    const ok       = events.filter(e => e.status === 'OK').length;
    const nearFull = events.filter(e => e.status === 'NEAR_FULL').length;
    const overflow = events.filter(e => e.status === 'OVERFLOW').length;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['OK', 'Near Full', 'Overflow'],
            datasets: [{
                data: [ok, nearFull, overflow],
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
            cutout: '60%',
            plugins: { legend: { position: 'bottom' } },
        }
    });
}


// ──────────────────────────────────────────────
// Chart 2: Fill Level Distribution (Bar)
// ──────────────────────────────────────────────
function createFillDistributionChart(events) {
    const ctx = document.getElementById('chart-fill-distribution');
    if (!ctx) return;

    // Bucket fill percentages into ranges
    const buckets = { '0-20': 0, '20-40': 0, '40-60': 0, '60-80': 0, '80-100': 0 };

    events.forEach(e => {
        const fill = parseFloat(e.fill_percent) || 0;
        if (fill < 20)      buckets['0-20']++;
        else if (fill < 40) buckets['20-40']++;
        else if (fill < 60) buckets['40-60']++;
        else if (fill < 80) buckets['60-80']++;
        else                buckets['80-100']++;
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(buckets).map(k => k + '%'),
            datasets: [{
                label: 'Detections',
                data: Object.values(buckets),
                backgroundColor: [
                    'rgba(63, 185, 80, 0.6)',
                    'rgba(63, 185, 80, 0.6)',
                    'rgba(88, 166, 255, 0.6)',
                    'rgba(210, 153, 34, 0.6)',
                    'rgba(248, 81, 73, 0.6)',
                ],
                borderColor: [
                    'rgba(63, 185, 80, 1)',
                    'rgba(63, 185, 80, 1)',
                    'rgba(88, 166, 255, 1)',
                    'rgba(210, 153, 34, 1)',
                    'rgba(248, 81, 73, 1)',
                ],
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(48, 54, 61, 0.3)' },
                    ticks: { stepSize: 1 },
                },
            },
        }
    });
}


// ──────────────────────────────────────────────
// Chart 3: Detections Over Time (Line)
// ──────────────────────────────────────────────
function createDetectionsTimeChart(events) {
    const ctx = document.getElementById('chart-detections-time');
    if (!ctx) return;

    // Group events by timestamp (truncated to seconds)
    const timeCounts = {};
    events.forEach(e => {
        const ts = (e.timestamp || '').substring(0, 19); // trim to seconds
        timeCounts[ts] = (timeCounts[ts] || 0) + 1;
    });

    const labels = Object.keys(timeCounts).slice(-30); // Last 30 time points
    const data   = labels.map(l => timeCounts[l]);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels.map(l => l.substring(11)), // just time
            datasets: [{
                label: 'Detections',
                data: data,
                borderColor: 'rgba(88, 166, 255, 0.8)',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 2,
                pointBackgroundColor: 'rgba(88, 166, 255, 1)',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8 },
                },
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(48, 54, 61, 0.3)' },
                },
            },
        }
    });
}


// ──────────────────────────────────────────────
// Chart 4: Confidence Distribution (Bar)
// ──────────────────────────────────────────────
function createConfidenceChart(events) {
    const ctx = document.getElementById('chart-confidence');
    if (!ctx) return;

    const buckets = { '0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0 };

    events.forEach(e => {
        const conf = parseFloat(e.confidence) || 0;
        if (conf < 0.2)      buckets['0-0.2']++;
        else if (conf < 0.4) buckets['0.2-0.4']++;
        else if (conf < 0.6) buckets['0.4-0.6']++;
        else if (conf < 0.8) buckets['0.6-0.8']++;
        else                 buckets['0.8-1.0']++;
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(buckets),
            datasets: [{
                label: 'Detections',
                data: Object.values(buckets),
                backgroundColor: 'rgba(188, 140, 255, 0.5)',
                borderColor: 'rgba(188, 140, 255, 1)',
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(48, 54, 61, 0.3)' },
                    ticks: { stepSize: 1 },
                },
            },
        }
    });
}
