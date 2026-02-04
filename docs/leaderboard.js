/**
 * HeteroShot Interactive Leaderboard
 * Client-side logic for filtering, sorting, and displaying leaderboard data
 */

// Configuration
const CSV_PATH = '../leaderboard/leaderboard.csv';

// State
let allData = [];
let filteredData = [];
let sortColumn = 'score';
let sortDirection = 'desc';

/**
 * Parse CSV string into array of objects
 */
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return [];
    
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length === headers.length) {
            const row = {};
            headers.forEach((header, index) => {
                row[header] = values[index].trim();
            });
            // Parse numeric score
            row.score = parseFloat(row.score) || 0;
            data.push(row);
        }
    }
    
    return data;
}

/**
 * Parse a single CSV line, handling quoted values
 */
function parseCSVLine(line) {
    const values = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            values.push(current);
            current = '';
        } else {
            current += char;
        }
    }
    values.push(current);
    
    return values;
}

/**
 * Fetch and load leaderboard data
 */
async function loadLeaderboard() {
    const tbody = document.getElementById('leaderboard-body');
    tbody.innerHTML = '<tr><td colspan="6" class="loading">Loading leaderboard</td></tr>';
    
    try {
        const response = await fetch(CSV_PATH);
        if (!response.ok) {
            throw new Error(`Failed to load leaderboard data: ${response.status}`);
        }
        
        const csvText = await response.text();
        allData = parseCSV(csvText);
        
        if (allData.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No submissions yet</td></tr>';
            return;
        }
        
        // Initial sort and render
        applyFiltersAndSort();
        updateStats();
        
    } catch (error) {
        console.error('Error loading leaderboard:', error);
        tbody.innerHTML = `<tr><td colspan="6" class="empty-state">Error loading data: ${error.message}</td></tr>`;
    }
}

/**
 * Apply current filters and sort, then render
 */
function applyFiltersAndSort() {
    const searchTerm = document.getElementById('search').value.toLowerCase();
    const modelFilter = document.getElementById('model-filter').value;
    const dateFilter = document.getElementById('date-filter').value;
    
    // Filter data
    filteredData = allData.filter(row => {
        // Search filter
        if (searchTerm) {
            const searchFields = [row.team, row.model_type, row.notes, row.date].join(' ').toLowerCase();
            if (!searchFields.includes(searchTerm)) {
                return false;
            }
        }
        
        // Model type filter
        if (modelFilter && row.model_type !== modelFilter) {
            return false;
        }
        
        // Date filter
        if (dateFilter) {
            const days = parseInt(dateFilter);
            const rowDate = new Date(row.date);
            const cutoffDate = new Date();
            cutoffDate.setDate(cutoffDate.getDate() - days);
            
            if (rowDate < cutoffDate) {
                return false;
            }
        }
        
        return true;
    });
    
    // Sort data
    filteredData.sort((a, b) => {
        let aVal = a[sortColumn];
        let bVal = b[sortColumn];
        
        // Handle numeric comparison for score
        if (sortColumn === 'score') {
            aVal = parseFloat(aVal) || 0;
            bVal = parseFloat(bVal) || 0;
        } else {
            aVal = String(aVal || '').toLowerCase();
            bVal = String(bVal || '').toLowerCase();
        }
        
        let comparison = 0;
        if (aVal < bVal) comparison = -1;
        if (aVal > bVal) comparison = 1;
        
        return sortDirection === 'asc' ? comparison : -comparison;
    });
    
    renderTable();
    updateStats();
}

/**
 * Render the leaderboard table
 */
function renderTable() {
    const tbody = document.getElementById('leaderboard-body');
    
    if (filteredData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No matching entries found</td></tr>';
        return;
    }
    
    let html = '';
    
    filteredData.forEach((row, index) => {
        const rank = index + 1;
        const rankClass = rank <= 3 ? `rank-${rank}` : '';
        const modelClass = `model-${row.model_type || 'baseline'}`.replace(/\+/g, '\\+');
        
        html += `
            <tr>
                <td class="rank-cell ${rankClass}">${rank}</td>
                <td>${escapeHtml(row.team)}</td>
                <td class="toggle-model_type"><span class="model-badge model-${row.model_type || 'baseline'}">${escapeHtml(row.model_type || 'N/A')}</span></td>
                <td class="score-cell">${row.score.toFixed(6)}</td>
                <td class="date-cell toggle-date">${escapeHtml(row.date || 'N/A')}</td>
                <td class="notes-cell toggle-notes" title="${escapeHtml(row.notes || '')}">${escapeHtml(row.notes || '')}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
    
    // Apply column visibility
    applyColumnVisibility();
}

/**
 * Update statistics display
 */
function updateStats() {
    document.getElementById('total-entries').textContent = `${allData.length} total entries`;
    
    const filteredEl = document.getElementById('filtered-entries');
    if (filteredData.length !== allData.length) {
        filteredEl.textContent = `(${filteredData.length} shown)`;
    } else {
        filteredEl.textContent = '';
    }
}

/**
 * Handle column sorting
 */
function handleSort(column) {
    // Update sort state
    if (sortColumn === column) {
        sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        sortColumn = column;
        sortDirection = column === 'score' ? 'desc' : 'asc';
    }
    
    // Update header classes
    document.querySelectorAll('th.sortable').forEach(th => {
        th.classList.remove('sorted-asc', 'sorted-desc');
        if (th.dataset.sort === sortColumn) {
            th.classList.add(sortDirection === 'asc' ? 'sorted-asc' : 'sorted-desc');
        }
    });
    
    applyFiltersAndSort();
}

/**
 * Toggle column visibility
 */
function toggleColumn(columnName, visible) {
    const cells = document.querySelectorAll(`.toggle-${columnName}`);
    cells.forEach(cell => {
        cell.classList.toggle('hidden', !visible);
    });
}

/**
 * Apply current column visibility settings
 */
function applyColumnVisibility() {
    document.querySelectorAll('.column-toggles input[type="checkbox"]').forEach(checkbox => {
        toggleColumn(checkbox.dataset.column, checkbox.checked);
    });
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Debounce function for search input
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Load data
    loadLeaderboard();
    
    // Search input
    document.getElementById('search').addEventListener('input', debounce(applyFiltersAndSort, 300));
    
    // Filter dropdowns
    document.getElementById('model-filter').addEventListener('change', applyFiltersAndSort);
    document.getElementById('date-filter').addEventListener('change', applyFiltersAndSort);
    
    // Column toggles
    document.querySelectorAll('.column-toggles input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            toggleColumn(checkbox.dataset.column, checkbox.checked);
        });
    });
    
    // Sortable headers
    document.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', () => handleSort(th.dataset.sort));
    });
});
