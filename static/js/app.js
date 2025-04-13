document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const analyzeButton = document.getElementById('analyzeButton');
    const recordingStatus = document.getElementById('recordingStatus');
    const transcriptContainer = document.getElementById('transcriptContainer');
    const toneSentimentEl = document.getElementById('toneSentiment');
    const textSentimentEl = document.getElementById('textSentiment');
    const finalSentimentEl = document.getElementById('finalSentiment');
    const positiveCountEl = document.getElementById('positiveCount');
    const neutralCountEl = document.getElementById('neutralCount');
    const negativeCountEl = document.getElementById('negativeCount');
    const suggestionsEl = document.getElementById('suggestions');

    // State variables
    let isRecording = false;
    let mediaRecorder;
    let audioChunks = [];
    let sentimentHistory = [];
    let sentimentCounts = { POSITIVE: 0, NEUTRAL: 0, NEGATIVE: 0 };
    let audioBlob;
    let analysisInProgress = false;
    
    // Initialize the waveform visualizer
    const wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#4a6fa5',
        progressColor: '#166088',
        cursorWidth: 0,
        height: 100,
        barWidth: 2,
        barGap: 1,
        responsive: true
    });

    // Initialize sentiment history chart
    const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
    const sentimentChart = new Chart(sentimentCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Sentiment Score',
                data: [],
                fill: false,
                borderColor: '#4a6fa5',
                tension: 0.1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 3,
                    ticks: {
                        callback: function(value) {
                            if (value === 0) return '';
                            if (value === 1) return 'Negative';
                            if (value === 2) return 'Neutral';
                            if (value === 3) return 'Positive';
                            return '';
                        }
                    }
                }
            },
            maintainAspectRatio: false,
            responsive: true
        }
    });

    // Initialize sentiment pie chart
    const pieCtx = document.getElementById('sentimentPieChart').getContext('2d');
    const sentimentPieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    '#4caf50',
                    '#ffc107',
                    '#f44336'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });

    // Process analysis results 
    function processAnalysisResults(data) {
        if (data.error) {
            console.error('Server error:', data.error);
            displayError(`Analysis error: ${data.error}`);
            return;
        }

        // Update transcript
        if (data.transcribed_text && data.transcribed_text.trim() !== '') {
            const transcriptBubble = document.createElement('div');
            transcriptBubble.className = 'transcript-bubble';
            transcriptBubble.textContent = data.transcribed_text;
            transcriptContainer.appendChild(transcriptBubble);
            transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
        } else {
            const emptyTranscriptMsg = document.createElement('div');
            emptyTranscriptMsg.className = 'transcript-bubble empty';
            emptyTranscriptMsg.textContent = 'No speech detected. Please try again with clearer audio.';
            transcriptContainer.appendChild(emptyTranscriptMsg);
            transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
        }

        // Update sentiment indicators
        updateSentimentIndicator(toneSentimentEl, 'Tone', data.tone_sentiment);
        updateSentimentIndicator(textSentimentEl, 'Text', data.text_sentiment);
        updateSentimentIndicator(finalSentimentEl, 'Overall', data.final_sentiment);

        // Update sentiment history
        updateSentimentHistory(data.final_sentiment);
        
        // Update sentiment counts
        updateSentimentCounts(data.final_sentiment);

        // Update suggestions
        updateSuggestions(data.suggestions || []);
    }

    // Update suggestions display
    function updateSuggestions(suggestions) {
        suggestionsEl.innerHTML = '';
        
        if (suggestions.length === 0) {
            const noSuggestions = document.createElement('p');
            noSuggestions.textContent = 'No specific suggestions available.';
            suggestionsEl.appendChild(noSuggestions);
            return;
        }
        
        const suggestionsList = document.createElement('ul');
        suggestions.forEach(suggestion => {
            const item = document.createElement('li');
            item.textContent = suggestion;
            suggestionsList.appendChild(item);
        });
        
        suggestionsEl.appendChild(suggestionsList);
    }

    // Display error message
    function displayError(message) {
        const errorEl = document.createElement('div');
        errorEl.className = 'error-message';
        errorEl.textContent = message;
        
        // Add to transcript container for visibility
        transcriptContainer.appendChild(errorEl);
        transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
    }

    // Update sentiment indicator styling
    function updateSentimentIndicator(element, label, sentiment) {
        element.textContent = `${label}: ${sentiment.toLowerCase()}`;
        
        // Remove existing classes
        element.classList.remove('positive', 'neutral', 'negative');
        
        // Add appropriate class
        if (sentiment === 'POSITIVE') {
            element.classList.add('positive');
        } else if (sentiment === 'NEUTRAL') {
            element.classList.add('neutral');
        } else if (sentiment === 'NEGATIVE') {
            element.classList.add('negative');
        }
    }

    // Update sentiment history chart
    function updateSentimentHistory(sentiment) {
        let sentimentValue = 2; // Default to neutral
        
        if (sentiment === 'POSITIVE') {
            sentimentValue = 3;
        } else if (sentiment === 'NEGATIVE') {
            sentimentValue = 1;
        }
        
        sentimentHistory.push(sentimentValue);
        
        // Keep only the last 20 values
        if (sentimentHistory.length > 20) {
            sentimentHistory.shift();
        }
        
        // Update chart
        sentimentChart.data.labels = Array.from({ length: sentimentHistory.length }, (_, i) => i + 1);
        sentimentChart.data.datasets[0].data = sentimentHistory;
        sentimentChart.update();
    }

    // Update sentiment counts and pie chart
    function updateSentimentCounts(sentiment) {
        sentimentCounts[sentiment]++;
        
        positiveCountEl.textContent = sentimentCounts.POSITIVE;
        neutralCountEl.textContent = sentimentCounts.NEUTRAL;
        negativeCountEl.textContent = sentimentCounts.NEGATIVE;
        
        // Update pie chart
        sentimentPieChart.data.datasets[0].data = [
            sentimentCounts.POSITIVE,
            sentimentCounts.NEUTRAL,
            sentimentCounts.NEGATIVE
        ];
        sentimentPieChart.update();
    }

    // Setup audio visualization during recording
    function setupAudioVisualization(stream) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        
        source.connect(analyser);
        analyser.fftSize = 2048;
        
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        function updateWaveform() {
            if (!isRecording) return;
            
            analyser.getByteTimeDomainData(dataArray);
            
            const waveformData = Array.from(dataArray).map(val => {
                return (val / 128.0) - 1;
            });
            
            wavesurfer.loadDecodedBuffer({
                numberOfChannels: 1,
                length: waveformData.length,
                sampleRate: audioContext.sampleRate,
                getChannelData: function(channel) {
                    return new Float32Array(waveformData);
                }
            });
            
            requestAnimationFrame(updateWaveform);
        }
        
        updateWaveform();
    }

    // Start recording audio
    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            setupAudioVisualization(stream);
            
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm; codecs=opus'
            });
            
            mediaRecorder.addEventListener('dataavailable', event => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            });
            
            mediaRecorder.addEventListener('start', () => {
                audioChunks = [];
                isRecording = true;
                recordingStatus.textContent = 'Recording...';
                recordingStatus.classList.add('pulse');
                startButton.disabled = true;
                stopButton.disabled = false;
                analyzeButton.disabled = true;
                
                // Clear suggestions when starting new recording
                suggestionsEl.innerHTML = '<p>Recording in progress. Analysis results will appear here.</p>';
            });
            
            mediaRecorder.addEventListener('stop', () => {
                isRecording = false;
                recordingStatus.textContent = 'Recording stopped';
                recordingStatus.classList.remove('pulse');
                startButton.disabled = false;
                stopButton.disabled = true;
                analyzeButton.disabled = false;
                
                // Create audio blob when recording stops
                audioBlob = new Blob(audioChunks, { type: 'audio/webm; codecs=opus' });
                
                // Display recording duration
                const durationSeconds = audioChunks.length * 0.1; // Rough estimate
                recordingStatus.textContent = `Recording stopped (approx. ${durationSeconds.toFixed(1)}s)`;
            });
            
            // Start recording
            mediaRecorder.start(100); // Capture in 100ms chunks
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Error accessing microphone. Please ensure you have given permission to use the microphone.');
            recordingStatus.textContent = 'Microphone access denied';
        }
    }

    // Stop recording
    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }

    // Analyze the recorded audio
    async function analyzeAudio() {
        if (!audioBlob) {
            alert('No audio recorded');
            return;
        }
        
        if (analysisInProgress) {
            alert('Analysis already in progress');
            return;
        }
        
        try {
            analysisInProgress = true;
            
            // Create FormData and append the audio blob
            const formData = new FormData();
            formData.append('file', audioBlob, 'audio.webm');
            
            // Show loading state
            analyzeButton.disabled = true;
            analyzeButton.textContent = 'Analyzing...';
            suggestionsEl.innerHTML = '<p>Processing audio... Please wait.</p>';
            
            // Send to server for analysis
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            processAnalysisResults(data);
            
        } catch (error) {
            console.error('Error analyzing audio:', error);
            alert('Error analyzing audio. Please try again.');
            suggestionsEl.innerHTML = '<p>Analysis error. Please try recording again.</p>';
        } finally {
            // Reset button state
            analyzeButton.disabled = false;
            analyzeButton.textContent = 'Analyze';
            analysisInProgress = false;
        }
    }

    // Auto-analyze after stopping recording (optional feature)
    function autoAnalyzeAfterStop() {
        if (audioBlob && !analysisInProgress) {
            setTimeout(() => {
                analyzeAudio();
            }, 500); // Small delay after recording stops
        }
    }

    // Check browser compatibility
    function checkBrowserCompatibility() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Your browser does not support audio recording. Please use a modern browser like Chrome, Firefox, or Edge.');
            startButton.disabled = true;
            recordingStatus.textContent = 'Recording not supported in this browser';
        }
    }

    // Initialize application
    function init() {
        checkBrowserCompatibility();
        
        // Event listeners
        startButton.addEventListener('click', startRecording);
        stopButton.addEventListener('click', () => {
            stopRecording();
            // Uncomment below line to enable auto-analysis
            // autoAnalyzeAfterStop(); 
        });
        analyzeButton.addEventListener('click', analyzeAudio);
        
        // Clean up on page unload
        window.addEventListener('beforeunload', () => {
            if (isRecording) {
                stopRecording();
            }
        });
    }

    // Start the application
    init();
});