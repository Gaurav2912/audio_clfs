const resultLabel = document.getElementById('resultLabel');
const gifContainer = document.getElementById('gifContainer');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');

startButton.addEventListener('click', startPrediction);
stopButton.addEventListener('click', stopPrediction);

let pollInterval;

function startPrediction() {
    fetch('/start_microphone_prediction', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started') {
                gifContainer.style.display = 'block';
                startPollingResults();
            }
        });
}

function stopPrediction() {
    fetch('/stop_prediction', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'stopped') {
                gifContainer.style.display = 'none';
                stopPollingResults();
            }
        });
}

function startPollingResults() {
    pollInterval = setInterval(() => {
        fetch('/get_latest_result')
            .then(response => response.json())
            .then(data => {
                updateResultLabel(data.result);
            });
    }, 1000);
}

function stopPollingResults() {
    clearInterval(pollInterval);
}

function updateResultLabel(result) {
    resultLabel.textContent = result;
    if (result === "BACKGROUND NOISE") {
        resultLabel.style.color = '#013220';
    } else if (result === "GUN SHOT DETECTED") {
        resultLabel.style.color = '#8B0000';
    } else if (result === "DISTRESSED AUDIO DETECTED") {
        resultLabel.style.color = '#00008B';
    }
}
