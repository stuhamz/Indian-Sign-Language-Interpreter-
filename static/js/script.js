// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const downloadBtn = document.getElementById('downloadBtn');
const clearBtn = document.getElementById('clearBtn');
const historyList = document.getElementById('historyList');
const recordingStatus = document.getElementById('recordingStatus');
const videoFeed = document.getElementById('videoFeed');

// State
let isInterpreting = false;
let historyUpdateInterval = null;

// Helper Functions
const formatTimestamp = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString();
};

const formatConfidence = (confidence) => {
    return (confidence * 100).toFixed(1) + '%';
};

const updateHistory = async () => {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.status === 'success') {
            // Clear current history
            historyList.innerHTML = '';
            
            // Add new history items
            data.history.reverse().forEach(item => {
                const historyItem = document.createElement('div');
                historyItem.className = 'bg-gray-50 p-3 rounded-lg shadow-sm';
                historyItem.innerHTML = `
                    <div class="flex justify-between items-center">
                        <span class="text-lg font-semibold text-gray-800">${item.sign}</span>
                        <span class="text-sm text-gray-500">${formatTimestamp(item.timestamp)}</span>
                    </div>
                    <div class="mt-1">
                        <div class="flex items-center">
                            <div class="flex-1 bg-gray-200 rounded-full h-2">
                                <div class="bg-blue-500 h-2 rounded-full" style="width: ${formatConfidence(item.confidence)}"></div>
                            </div>
                            <span class="ml-2 text-sm text-gray-600">${formatConfidence(item.confidence)}</span>
                        </div>
                    </div>
                `;
                historyList.appendChild(historyItem);
            });
        }
    } catch (error) {
        console.error('Error updating history:', error);
    }
};

// Event Handlers
const startInterpreter = async () => {
    try {
        const response = await fetch('/api/start', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            isInterpreting = true;
            startBtn.classList.add('hidden');
            stopBtn.classList.remove('hidden');
            recordingStatus.classList.remove('hidden');
            recordingStatus.classList.add('flex');
            
            // Start history updates
            historyUpdateInterval = setInterval(updateHistory, 1000);
            
            // Refresh video feed
            videoFeed.src = '/video_feed?' + new Date().getTime();
        } else {
            alert('Failed to start interpreter: ' + data.message);
        }
    } catch (error) {
        console.error('Error starting interpreter:', error);
        alert('Error starting interpreter');
    }
};

const stopInterpreter = async () => {
    try {
        const response = await fetch('/api/stop', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            isInterpreting = false;
            stopBtn.classList.add('hidden');
            startBtn.classList.remove('hidden');
            recordingStatus.classList.add('hidden');
            recordingStatus.classList.remove('flex');
            
            // Stop history updates
            if (historyUpdateInterval) {
                clearInterval(historyUpdateInterval);
            }
        } else {
            alert('Failed to stop interpreter: ' + data.message);
        }
    } catch (error) {
        console.error('Error stopping interpreter:', error);
        alert('Error stopping interpreter');
    }
};

const downloadHistory = () => {
    window.location.href = '/api/download_history';
};

const clearHistory = async () => {
    if (!confirm('Are you sure you want to clear the history?')) {
        return;
    }
    
    try {
        const response = await fetch('/api/clear_history', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            historyList.innerHTML = '';
        } else {
            alert('Failed to clear history: ' + data.message);
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        alert('Error clearing history');
    }
};

// Event Listeners
startBtn.addEventListener('click', startInterpreter);
stopBtn.addEventListener('click', stopInterpreter);
downloadBtn.addEventListener('click', downloadHistory);
clearBtn.addEventListener('click', clearHistory);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Initial history load
    updateHistory();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (isInterpreting) {
        stopInterpreter();
    }
});

// Error handling for video feed
videoFeed.addEventListener('error', () => {
    if (isInterpreting) {
        console.error('Video feed error');
        stopInterpreter();
        alert('Video feed error. Interpreter stopped.');
    }
});