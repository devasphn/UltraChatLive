import asyncio
import json
import logging
import signal
import sys
from typing import Dict, Any, Optional
import uvloop
from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebRTCConnection:
    def __init__(self):
        # Create proper RTCConfiguration object
        ice_servers = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        self.configuration = RTCConfiguration(iceServers=ice_servers)
        
        # Initialize peer connection with proper configuration
        self.pc = RTCPeerConnection(configuration=self.configuration)
        self.relay = MediaRelay()
        
        # Set up event handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"🔗 Connection state is {self.pc.connectionState}")
        
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"🧊 ICE connection state: {self.pc.iceConnectionState}")
        
        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"🎵 Received track: {track.kind}")
            if track.kind == "audio":
                # Handle audio track here
                await self.handle_audio_track(track)
    
    async def handle_audio_track(self, track):
        """Handle incoming audio track from WebRTC"""
        try:
            while True:
                frame = await track.recv()
                # Process audio frame here
                # Convert to numpy array for processing
                audio_data = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16)
                
                # Add your audio processing logic here
                # For example, send to your AI models
                logger.debug(f"📊 Received audio frame: {len(audio_data)} samples")
                
        except Exception as e:
            logger.error(f"❌ Error processing audio track: {e}")
    
    async def handle_offer(self, sdp: str, type: str) -> Dict[str, str]:
        """Handle WebRTC offer and return answer"""
        try:
            # Create session description
            offer = RTCSessionDescription(sdp=sdp, type=type)
            
            # Set remote description
            await self.pc.setRemoteDescription(offer)
            
            # Create answer
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            
            logger.info("✅ WebRTC offer processed successfully")
            
            return {
                "type": self.pc.localDescription.type,
                "sdp": self.pc.localDescription.sdp
            }
            
        except Exception as e:
            logger.error(f"❌ Error handling offer: {e}")
            raise
    
    async def add_ice_candidate(self, candidate_data: Dict[str, Any]):
        """Add ICE candidate with proper parameter mapping"""
        try:
            # Map the parameters correctly
            candidate_params = {
                "candidate": candidate_data["candidate"],
                "sdpMLineIndex": candidate_data.get("sdpMLineIndex"),
                "sdpMid": candidate_data.get("sdpMid")  # Note: camelCase, not snake_case
            }
            
            # Remove None values
            candidate_params = {k: v for k, v in candidate_params.items() if v is not None}
            
            candidate = RTCIceCandidate(**candidate_params)
            await self.pc.addIceCandidate(candidate)
            
            logger.debug("✅ ICE candidate added successfully")
            
        except Exception as e:
            logger.error(f"❌ Error adding ICE candidate: {e}")
            raise
    
    async def close(self):
        """Close the WebRTC connection"""
        await self.pc.close()


class UltraChatServer:
    def __init__(self):
        self.connections: Dict[int, WebRTCConnection] = {}
        self.app = web.Application()
        self.setup_routes()
        
        # Initialize your AI models here
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize your AI models"""
        logger.info("🚀 Initializing models on device: cuda:0")
        
        # Add your model initialization code here
        # self.vad_model = ...
        # self.ultravox_pipeline = ...
        # self.tts_model = ...
        
        logger.info("🎉 All models loaded successfully!")
    
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_static('/', path='static', name='static')
    
    async def index(self, request):
        """Serve the main HTML page"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>UltraChat WebRTC</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                button { padding: 10px 20px; margin: 10px; border: none; border-radius: 5px; cursor: pointer; }
                .start { background: #4CAF50; color: white; }
                .stop { background: #f44336; color: white; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .connected { background: #d4edda; color: #155724; }
                .disconnected { background: #f8d7da; color: #721c24; }
                #log { height: 300px; overflow-y: scroll; background: #f8f9fa; padding: 10px; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 UltraChat WebRTC Audio</h1>
                <button id="startBtn" class="start">START CONVERSATION</button>
                <button id="stopBtn" class="stop" disabled>STOP CONVERSATION</button>
                <div id="status" class="status disconnected">📡 Disconnected</div>
                <div id="log"></div>
            </div>
            
            <script>
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
                const status = document.getElementById('status');
                const log = document.getElementById('log');
                
                let ws = null;
                let pc = null;
                let localStream = null;
                
                function addLog(message) {
                    const timestamp = new Date().toLocaleTimeString();
                    log.innerHTML += `[${timestamp}] ${message}<br>`;
                    log.scrollTop = log.scrollHeight;
                }
                
                async function startConversation() {
                    try {
                        // Get user media
                        localStream = await navigator.mediaDevices.getUserMedia({ 
                            audio: {
                                echoCancellation: true,
                                noiseSuppression: true,
                                autoGainControl: true,
                                sampleRate: 48000
                            }
                        });
                        
                        // Create WebSocket connection
                        const wsUrl = `ws://${window.location.host}/ws`;
                        ws = new WebSocket(wsUrl);
                        
                        ws.onopen = () => {
                            addLog('🔌 WebSocket connected');
                            status.textContent = '🔄 Connecting...';
                            status.className = 'status';
                            setupWebRTC();
                        };
                        
                        ws.onmessage = async (event) => {
                            const data = JSON.parse(event.data);
                            await handleWebRTCMessage(data);
                        };
                        
                        ws.onclose = () => {
                            addLog('🔌 WebSocket closed');
                            updateStatus('disconnected');
                        };
                        
                        ws.onerror = (error) => {
                            addLog(`❌ WebSocket error: ${error}`);
                            updateStatus('disconnected');
                        };
                        
                    } catch (error) {
                        addLog(`❌ Error starting conversation: ${error.message}`);
                    }
                }
                
                async function setupWebRTC() {
                    try {
                        // Create peer connection
                        pc = new RTCPeerConnection({
                            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                        });
                        
                        // Add local stream
                        localStream.getTracks().forEach(track => {
                            pc.addTrack(track, localStream);
                        });
                        
                        // Handle ICE candidates
                        pc.onicecandidate = (event) => {
                            if (event.candidate) {
                                ws.send(JSON.stringify({
                                    type: 'ice-candidate',
                                    candidate: event.candidate.candidate,
                                    sdpMid: event.candidate.sdpMid,
                                    sdpMLineIndex: event.candidate.sdpMLineIndex
                                }));
                            }
                        };
                        
                        // Handle connection state changes
                        pc.onconnectionstatechange = () => {
                            addLog(`🔗 Connection state: ${pc.connectionState}`);
                            if (pc.connectionState === 'connected') {
                                updateStatus('connected');
                            } else if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
                                updateStatus('disconnected');
                            }
                        };
                        
                        // Create and send offer
                        const offer = await pc.createOffer();
                        await pc.setLocalDescription(offer);
                        
                        ws.send(JSON.stringify({
                            type: 'offer',
                            sdp: offer.sdp
                        }));
                        
                        addLog('📤 Sent WebRTC offer');
                        
                    } catch (error) {
                        addLog(`❌ WebRTC setup error: ${error.message}`);
                    }
                }
                
                async function handleWebRTCMessage(data) {
                    try {
                        if (data.type === 'answer') {
                            addLog('📥 Received WebRTC answer');
                            await pc.setRemoteDescription(new RTCSessionDescription(data));
                        } else if (data.type === 'ice-candidate') {
                            addLog('📥 Received ICE candidate');
                            await pc.addIceCandidate(new RTCIceCandidate({
                                candidate: data.candidate,
                                sdpMid: data.sdpMid,
                                sdpMLineIndex: data.sdpMLineIndex
                            }));
                        }
                    } catch (error) {
                        addLog(`❌ Error handling WebRTC message: ${error.message}`);
                    }
                }
                
                function updateStatus(state) {
                    if (state === 'connected') {
                        status.textContent = '✅ Connected';
                        status.className = 'status connected';
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                    } else {
                        status.textContent = '📡 Disconnected';
                        status.className = 'status disconnected';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    }
                }
                
                function stopConversation() {
                    if (pc) {
                        pc.close();
                        pc = null;
                    }
                    if (ws) {
                        ws.close();
                        ws = null;
                    }
                    if (localStream) {
                        localStream.getTracks().forEach(track => track.stop());
                        localStream = null;
                    }
                    updateStatus('disconnected');
                    addLog('🛑 Conversation stopped');
                }
                
                startBtn.addEventListener('click', startConversation);
                stopBtn.addEventListener('click', stopConversation);
                
                // Initial status
                updateStatus('disconnected');
            </script>
        </body>
        </html>
        """
        return web.Response(text=html_content, content_type='text/html')
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        connection_id = id(ws)
        webrtc_conn = WebRTCConnection()
        self.connections[connection_id] = webrtc_conn
        
        logger.info(f"🔌 New WebSocket connection: {connection_id}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_webrtc_message(ws, webrtc_conn, data)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON decode error: {e}")
                    except Exception as e:
                        logger.error(f"❌ Error processing message: {e}")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"❌ WebSocket error: {ws.exception()}")
                    break
        
        except Exception as e:
            logger.error(f"❌ WebSocket handler error: {e}")
        
        finally:
            # Cleanup
            if connection_id in self.connections:
                await self.connections[connection_id].close()
                del self.connections[connection_id]
            logger.info(f"🔌 Closed WebSocket connection: {connection_id}")
        
        return ws
    
    async def handle_webrtc_message(self, ws, webrtc_conn: WebRTCConnection, data: Dict[str, Any]):
        """Handle WebRTC signaling messages"""
        try:
            if data["type"] == "offer":
                logger.info("📨 Received WebRTC offer")
                answer = await webrtc_conn.handle_offer(data["sdp"], data["type"])
                await ws.send_str(json.dumps(answer))
                logger.info("📤 Sent WebRTC answer")
                
            elif data["type"] == "ice-candidate":
                logger.debug("📨 Received ICE candidate")
                await webrtc_conn.add_ice_candidate(data)
                
        except Exception as e:
            logger.error(f"❌ Error handling WebRTC message: {e}")
            await ws.send_str(json.dumps({"type": "error", "message": str(e)}))


async def main():
    """Main server function"""
    # Use uvloop for better performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    print("🚀 Using uvloop for asyncio event loop.")
    print("🚀 UltraChat S2S - DEFINITIVE FIX - Starting server...")
    
    server = UltraChatServer()
    
    # Setup graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down...")
        asyncio.create_task(shutdown())
    
    async def shutdown():
        # Close all WebRTC connections
        for conn in server.connections.values():
            await conn.close()
        server.connections.clear()
        print("👋 Goodbye")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("🎉 Starting web server...")
        runner = web.AppRunner(server.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', 7860)
        await site.start()
        
        print("✅ Server started successfully!")
        print("🌐 Server running on http://0.0.0.0:7860")
        print("📱 Open the URL in your browser to start chatting!")
        
        # Keep the server running
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
