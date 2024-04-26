// File: client.js
const WebSocket = require('ws');

const SAMPLE_RATE = 16000; // samples per second
const BIT_DEPTH = 16; // bits per sample
const CHANNELS = 1; // number of audio channels
const BYTES_PER_SAMPLE = BIT_DEPTH / 8; // bytes per sample per channel
const BUFFER_SIZE = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE; // bytes per second

const lang = process.env.ASR_LANG ? process.env.ASR_LANG  : 'en'

const ws = new WebSocket(`ws://localhost:8000?lang=${lang}`);


ws.on('open', function open() {
  console.log('Connected to the server');
});

startTime = process.hrtime();


// Function to handle stream data
function handleStream() {
  // Read from the stream for 1 second
  let data = process.stdin.read(16000 * 2); // Assuming 44100 Hz, 16-bit audio, stereo (2 bytes per sample per channel)
  if (data) {
    // console.log('Reading data...');
    ws.send(data);
  }

  // Set a timeout to pause reading after 1 second
  setTimeout(handleStream, 1000); // Continue handling after 1 second
}

// Start processing the stream
handleStream();


process.stdin.on('end', () => {
  console.log('Finished reading stdin');
});

ws.on('close', function close() {
  console.log('Disconnected from server');
});

ws.on('error', function error(err) {
  console.error('WebSocket error:', err);
});

ws.on('message', (message) => {
  if (message != "None") {
    jsonData = JSON.parse(message);
    endTime = process.hrtime(startTime);

    console.log(`${endTime[0]}s Received: ${jsonData.StartTime.toFixed(2)} ${jsonData.EndTime.toFixed(2)} ${jsonData.Transcript}`);
  }
});





const { Readable } = require('stream');

// Create a new readable stream for stdin
const stdinStream = new Readable({
  read(size) {
    process.stdin.on('data', (chunk) => {
      this.push(chunk);
    });

    process.stdin.on('end', () => {
      this.push(null);
    });
  }
});

let reading = true;  // State flag to determine if we should read or pause
