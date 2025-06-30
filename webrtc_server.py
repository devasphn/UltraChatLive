import asyncio
import json
import logging
import os

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaBlackhole

import config
from audio_processing import AudioStreamTrack # Assuming AudioStreamTrack is in audio_processing.py

# The HTML client interface string
# TODO: Consider moving this to a separate HTML file and loading it.
HTML_CLIENT = """
<!DOCTYPE html>
<html>
<head>
    <title>UltraChat S2S - Enhanced</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; }
        .container { background: rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 30px; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); }
        h1 { text-align: center; margin-bottom: 30px; font-size: 2.5em; text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); }
        .controls { display: flex; justify-content: center; gap: 20px; margin: 30px 0; flex-wrap: wrap; }
        button { padding: 15px 30px; font-size: 16px; border: none; border-radius: 50px; cursor: pointer; transition: all 0.3s ease; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
        .start-btn { background: linear-gradient(45deg, #4CAF50, #45a049); color: white; }
        .stop-btn { background: linear-gradient(45deg, #f44336, #da190b); color: white; }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); }
        button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .status { text-align: center; font-size: 18px; margin: 20px 0; padding: 15px; border-radius: 10px; background: rgba(255, 255, 255, 0.2); }
        .status.connected { background: rgba(76, 175, 80, 0.3); }
        .status.error { background: rgba(244, 67, 54, 0.3); }
        .audio-visualizer { height: 100px; background: rgba(255, 255, 255, 0.1); border-radius: 10px; margin: 20px 0; display: flex; align-items: center; justify-content: center; font-size: 14px; }
        .instructions { background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 20px; margin-top: 20px; }
        .instructions h3 { margin-top: 0; } .instructions ul { padding-left: 20px; } .instructions li { margin: 10px 0; }
        .fix-note { background: rgba(76, 175, 80, 0.2); border-left: 4px solid #4CAF50; padding: 15px; margin: 20px 0; border-radius: 5px; }
    </style>
</head>
<body> <div class="container"> <h1>🎤 UltraChat S2S - Enhanced</h1> <div class="fix-note"> <strong>✅ Status:</strong> Enhanced logging, error handling, and configuration. </div> <div class="controls"> <button id="startBtn" class="start-btn">Start Conversation</button> <button id="stopBtn" class="stop-btn" disabled>Stop Conversation</button> </div> <div id="status" class="status">Ready to connect</div> <div class="audio-visualizer" id="visualizer"> Audio visualization will appear here </div> <div class="instructions"> <h3>Instructions:</h3> <ul> <li>Click "Start Conversation" to begin.</li> <li>Allow microphone access when prompted.</li> <li>Speak clearly. The system uses VAD to detect speech.</li> <li>Wait for the AI to respond.</li> </ul> </div> </div>
    <script>
        let websocket = null; let peerConnection = null; let localStream = null; let audioContext = null; let analyser = null;
        const startBtn = document.getElementById('startBtn'); const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status'); const visualizer = document.getElementById('visualizer');
        startBtn.addEventListener('click', startConversation); stopBtn.addEventListener('click', stopConversation);

        // Placeholder for STUN_SERVERS, to be replaced by Python's handle_http
        const clientConfig = { STUN_SERVERS: [] };

        function getWebSocketUrl() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${protocol}//${window.location.host}/ws`;
        }
        async function startConversation() {
            try {
                updateStatus('Connecting...', '');
                localStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, sampleRate: 16000 }});
                setupAudioVisualization();
                const wsUrl = getWebSocketUrl(); console.log('Connecting to WebSocket:', wsUrl);
                websocket = new WebSocket(wsUrl);
                websocket.onopen = () => { updateStatus('WebSocket Connected - Setting up WebRTC...', 'connected'); setupWebRTC(); };
                websocket.onmessage = handleSignalingMessage;
                websocket.onerror = (error) => { updateStatus('WebSocket connection error. See console.', 'error'); console.error('WebSocket error:', error); };
                websocket.onclose = (event) => {
                    const reason = event.reason ? `Reason: ${event.reason}` : 'No reason given.';
                    updateStatus(`WebSocket closed. Code: ${event.code}. ${reason}`, event.wasClean ? '' : 'error');
                    console.log('WebSocket closed:', event); stopConversationCleanup();
                };
                startBtn.disabled = true; stopBtn.disabled = false;
            } catch (error) { updateStatus('Error starting: ' + error.message, 'error'); console.error('Error starting conversation:', error); stopConversationCleanup(); }
        }
        async function setupWebRTC() {
            try {
                console.log("Using STUN servers for WebRTC:", clientConfig.STUN_SERVERS);
                peerConnection = new RTCPeerConnection({ iceServers: clientConfig.STUN_SERVERS.map(url => ({ urls: url })) });
                localStream.getTracks().forEach(track => peerConnection.addTrack(track, localStream));
                peerConnection.ontrack = (event) => { console.log('Remote track received:', event.track); playAudio(event.streams[0]); };
                peerConnection.onicecandidate = (event) => {
                    if (event.candidate && websocket && websocket.readyState === WebSocket.OPEN) {
                        if (event.candidate.candidate && event.candidate.candidate.trim() !== "") {
                            console.log('Sending ICE candidate:', event.candidate);
                            websocket.send(JSON.stringify({ type: 'ice-candidate', candidate: event.candidate.toJSON() }));
                        } else { console.log('Received empty/null ICE candidate, not sending.'); }
                    }
                };
                peerConnection.oniceconnectionstatechange = () => {
                    if (peerConnection) {
                        console.log('ICE connection state change:', peerConnection.iceConnectionState);
                        let state = peerConnection.iceConnectionState;
                        if (state === 'failed' || state === 'disconnected' || state === 'closed') { updateStatus('WebRTC issue: ' + state, 'error');
                        } else if (state === 'connected' || state === 'completed') { updateStatus('Ready - Start speaking!', 'connected'); }
                    }
                };
                const offer = await peerConnection.createOffer(); await peerConnection.setLocalDescription(offer);
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    console.log('Sending WebRTC offer.'); websocket.send(JSON.stringify({ type: 'offer', sdp: offer.sdp }));
                } else { updateStatus('WebSocket not open. Cannot send offer.', 'error'); }
                updateStatus('WebRTC Offer sent, waiting for answer...', 'connected');
            } catch (error) { updateStatus('WebRTC setup error: ' + error.message, 'error'); console.error('WebRTC error:', error); stopConversationCleanup(); }
        }
        async function handleSignalingMessage(event) {
            try {
                const message = JSON.parse(event.data); console.log('Received signaling message:', message.type);
                if (message.type === 'answer' && message.sdp) {
                    await peerConnection.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp: message.sdp }));
                    console.log('Remote description (answer) set.');
                } else if (message.type === 'ice-candidate' && message.candidate) {
                    if (message.candidate.candidate && message.candidate.candidate.trim() !== "") {
                        await peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
                        console.log('Added remote ICE candidate.');
                    } else { console.log('Received end-of-candidates signal or empty candidate.'); }
                }
            } catch (error) { console.error('Error handling signaling message:', error, 'Raw data:', event.data); updateStatus('Signaling error. See console.', 'error'); }
        }
        function setupAudioVisualization() {
            if (audioContext && audioContext.state !== 'closed') audioContext.close();
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(localStream);
            source.connect(analyser); analyser.fftSize = 256;
            const bufferLength = analyser.frequencyBinCount; const dataArray = new Uint8Array(bufferLength);
            function updateVisualization() {
                if (!analyser || !audioContext || audioContext.state === 'closed') return;
                analyser.getByteFrequencyData(dataArray); const average = dataArray.reduce((a, b) => a + b, 0) / bufferLength;
                visualizer.style.background = `linear-gradient(90deg, rgba(76, 175, 80, ${Math.min(1, average / 128)}) 0%, rgba(33, 150, 243, ${Math.min(1, average / 128)}) 100%)`;
                visualizer.textContent = average > 20 ? 'Speaking...' : 'Listening...'; // Lowered threshold
                requestAnimationFrame(updateVisualization);
            }
            updateVisualization();
        }
        function playAudio(remoteStream) {
            document.querySelectorAll("audio").forEach(el => el.remove()); // Clear old audio elements
            const audio = document.createElement('audio'); audio.srcObject = remoteStream; audio.autoplay = true;
            document.body.appendChild(audio);
            audio.play().catch(error => { console.error('Error playing remote audio:', error); updateStatus('Error playing audio. Check console.', 'error'); });
        }
        function stopConversation() { stopConversationCleanup(); updateStatus('Disconnected by user.', ''); }
        function stopConversationCleanup() {
            if (websocket) { websocket.onclose = null; if (websocket.readyState === WebSocket.OPEN || websocket.readyState === WebSocket.CONNECTING) websocket.close(1000, "User initiated disconnect"); websocket = null; }
            if (peerConnection) { peerConnection.onicecandidate = null; peerConnection.ontrack = null; peerConnection.oniceconnectionstatechange = null; peerConnection.close(); peerConnection = null; }
            if (localStream) { localStream.getTracks().forEach(track => track.stop()); localStream = null; }
            if (audioContext && audioContext.state !== 'closed') { audioContext.close().catch(console.error); audioContext = null; analyser = null; }
            startBtn.disabled = false; stopBtn.disabled = true;
            visualizer.style.background = 'rgba(255, 255, 255, 0.1)'; visualizer.textContent = 'Audio visualization will appear here';
        }
        function updateStatus(message, className) { statusDiv.textContent = message; statusDiv.className = 'status ' + (className || ''); }
        window.addEventListener('beforeunload', stopConversationCleanup);
    </script>
</body></html>
"""

class WebRTCConnection:
    def __init__(self):
        ice_servers = [{'urls': url} for url in config.STUN_SERVERS]
        self.pc = RTCPeerConnection(configuration={'iceServers': ice_servers})
        self.audio_track = AudioStreamTrack() # This will be our outgoing track with TTS
        self.recorder = None # To consume the outgoing track

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            logging.info(f"Data channel created by remote: {channel.label}")
            # TODO: Handle data channel messages if needed in the future

        @self.pc.on("track")
        def on_track(track: MediaStreamTrack):
            logging.info(f"Track {track.kind} received from remote client")
            if track.kind == "audio":
                # This is the audio from the client's microphone
                asyncio.create_task(self._handle_incoming_audio_track(track))

            @track.on("ended")
            async def on_ended():
                logging.info(f"Track {track.kind} from client ended")

    async def _handle_incoming_audio_track(self, track: MediaStreamTrack):
        logging.info("🎧 Starting to handle incoming audio track from client...")
        try:
            while True:
                try:
                    frame = await track.recv() # Receive audio frame from client
                    if frame:
                        audio_data_np = frame.to_ndarray(format='fltp').squeeze()
                        if audio_data_np.ndim > 1: # Ensure mono
                            audio_data_np = audio_data_np[0]
                        # Pass client audio to our AudioStreamTrack instance for processing
                        self.audio_track.receive_client_audio(audio_data_np.astype(np.float32))
                except Exception as frame_error:
                    logging.error(f"Error processing individual audio frame from client: {frame_error}", exc_info=True)
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logging.info("Incoming client audio track handling cancelled.")
        except Exception as e:
            logging.error(f"Major error handling incoming client audio track: {e}", exc_info=True)

def is_valid_ice_candidate(candidate_data: dict) -> bool:
    """Basic filter for ICE candidates."""
    if not candidate_data: return False
    candidate_str = candidate_data.get("candidate", "")
    if not candidate_str or candidate_str.strip() == "" or candidate_str == "null": return False
    try:
        parts = candidate_str.split()
        if len(parts) < 8: # Basic check for common structure
            logging.warning(f"ICE candidate seems malformed (too short): {candidate_str}")
            return False
        ip_address = parts[4]
        # Filter out common non-routable addresses if desired, though STUN should handle this
        if not ip_address or ip_address == "0.0.0.0":
            return False
        return True
    except Exception as e:
        logging.warning(f"Error parsing ICE candidate string '{candidate_str}': {e}", exc_info=True)
        return False

async def websocket_handler(request: web.Request):
    """Handles WebSocket connections for WebRTC signaling."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connection_id = hex(id(ws)) # For logging
    logging.info(f"🔌 New WebSocket connection established: {connection_id}")

    webrtc_conn = WebRTCConnection()
    pc = webrtc_conn.pc

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    message_type = data.get("type")

                    if message_type == "offer" and "sdp" in data:
                        logging.info(f"📨 [{connection_id}] Received WebRTC offer from client.")
                        offer_sdp = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                        await pc.setRemoteDescription(offer_sdp)

                        # Add our AudioStreamTrack to send TTS audio to the client
                        pc.addTrack(webrtc_conn.audio_track)

                        # Start consuming our outgoing audio track to make it work
                        if webrtc_conn.recorder is None:
                            webrtc_conn.recorder = MediaBlackhole()
                        webrtc_conn.recorder.addTrack(webrtc_conn.audio_track)
                        await webrtc_conn.recorder.start()

                        # Start the consumer task in our AudioStreamTrack that actually generates frames
                        webrtc_conn.audio_track.start_consumer_if_needed()

                        answer_sdp = await pc.createAnswer()
                        await pc.setLocalDescription(answer_sdp)
                        await ws.send_str(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))
                        logging.info(f"📤 [{connection_id}] Sent WebRTC answer to client.")

                    elif message_type == "ice-candidate" and "candidate" in data:
                        candidate_info = data["candidate"]
                        if is_valid_ice_candidate(candidate_info):
                            try:
                                ice_candidate = RTCIceCandidate(
                                    foundation=candidate_info.get("foundation"),
                                    component=candidate_info.get("component", 1),
                                    protocol=candidate_info.get("protocol", "udp"),
                                    priority=candidate_info.get("priority"),
                                    address=candidate_info.get("address"),
                                    port=candidate_info.get("port"),
                                    type=candidate_info.get("type"),
                                    sdpMid=candidate_info.get("sdpMid"),
                                    sdpMLineIndex=candidate_info.get("sdpMLineIndex"),
                                    usernameFragment=candidate_info.get("usernameFragment")
                                )
                                await pc.addIceCandidate(ice_candidate)
                            except Exception as e:
                                logging.error(f"❌ [{connection_id}] Failed to add ICE candidate: {candidate_info}. Error: {e}", exc_info=True)
                        else:
                            logging.debug(f"[{connection_id}] Filtered out invalid/empty ICE candidate: {candidate_info}")
                    else:
                        logging.warning(f"[{connection_id}] Received unknown WebSocket message: {data}")

                except json.JSONDecodeError as e:
                    logging.error(f"[{connection_id}] JSON decode error: {msg.data}. Error: {e}", exc_info=True)
                except Exception as e:
                    logging.error(f"[{connection_id}] Error handling WebSocket message: {e}", exc_info=True)

            elif msg.type == WSMsgType.ERROR:
                logging.error(f'[{connection_id}] WebSocket connection error: {ws.exception()}', exc_info=True)
                break
            elif msg.type == WSMsgType.CLOSED:
                logging.info(f"[{connection_id}] WebSocket connection closed by client (message type).")
                break

    except asyncio.CancelledError:
        logging.info(f"WebSocket handler for {connection_id} task cancelled.")
    except Exception as e:
        logging.error(f"[{connection_id}] Unhandled error in WebSocket handler main loop: {e}", exc_info=True)
    finally:
        logging.info(f"🔌 [{connection_id}] Cleaning up WebSocket and WebRTC connection.")
        if webrtc_conn.recorder:
            try: await webrtc_conn.recorder.stop()
            except Exception as e: logging.error(f"Error stopping MediaBlackhole for {connection_id}: {e}", exc_info=True)
        try: await pc.close()
        except Exception as e: logging.error(f"Error closing RTCPeerConnection for {connection_id}: {e}", exc_info=True)
        if not ws.closed:
            await ws.close()
        logging.info(f"🔌 [{connection_id}] WebSocket connection fully closed.")
    return ws

async def handle_http(request: web.Request):
    """Serves the HTML client, injecting STUN server configuration."""
    # Convert Python list of STUN servers to a JSON string for JavaScript
    stun_servers_json = json.dumps(config.STUN_SERVERS)

    # Replace a placeholder in the HTML_CLIENT string with this JSON string.
    # This is a simple templating approach. For more complex scenarios, a proper templating engine would be better.
    # Ensure the placeholder in HTML_CLIENT is unique and unlikely to clash, e.g., '{{STUN_SERVERS_JSON_PLACEHOLDER}}'
    # For this specific HTML, the JS part is: const clientConfig = { STUN_SERVERS: [] };
    # We will replace the `[]` part.

    # A slightly more robust placeholder might be <script id="config-data" type="application/json"> ... </script>
    # and then JS reads it. But for now, direct replacement in the script block:

    # Find: const clientConfig = { STUN_SERVERS: [] };
    # Replace with: const clientConfig = { STUN_SERVERS: [...] }; // where ... is stun_servers_json

    placeholder_string = "const clientConfig = { STUN_SERVERS: [] };"
    replacement_string = f"const clientConfig = {{ STUN_SERVERS: {stun_servers_json} }};"

    customized_html_client = HTML_CLIENT.replace(placeholder_string, replacement_string, 1)

    if placeholder_string not in HTML_CLIENT: # check if placeholder was even there
        logging.warning("STUN server placeholder not found in HTML_CLIENT. Client might use default STUN servers.")

    return web.Response(text=customized_html_client, content_type='text/html')

async def handle_favicon(request: web.Request):
    """Handles favicon requests with a simple transparent PNG."""
    favicon_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    return web.Response(body=favicon_data, content_type='image/png')
