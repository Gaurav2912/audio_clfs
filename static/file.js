const fileInput = document.getElementById('fileInput');
const fileLabel = document.getElementById('fileLabel');
const resultLabel = document.getElementById('resultLabel');
const gifContainer = document.getElementById('gifContainer');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');

let pollInterval;

fileInput.addEventListener('change', handleFileUpload);
startButton.addEventListener('click', startPrediction);
stopButton.addEventListener('click', stopPrediction);

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        fileLabel.textContent = `Selected: ${file.name}`;
        uploadFile(file);
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload_file', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        }
    });
}

function startPrediction() {
    fetch('/start_file_prediction', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started') {
                gifContainer.style.display = 'block';
                startPollingResults();
            } else if (data.error) {
                alert(data.error);
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
    } else if (result === "Audio processing completed") {
        resultLabel.style.color = '#4B0082';
        stopPollingResults();
        gifContainer.style.display = 'none';
    }
}
