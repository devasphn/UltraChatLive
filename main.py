import asyncio
import logging
import os
import config # Application configuration

# Import setup functions and server components from other modules
from logger_setup import setup_logging
from ai_models import initialize_all_models
from webrtc_server import handle_http, websocket_handler, handle_favicon

# aiohttp components for server setup
from aiohttp import web, web_runner

async def main_app():
    """
    Main application function:
    Initializes logging, AI models, sets up the web application routes,
    and starts the server.
    """
    # 1. Setup logging as the very first thing
    setup_logging() # Uses settings from config.py

    logging.info("🚀 Initializing Real-time S2S Agent...")

    # 2. Initialize AI models
    # Models (vad_model, uv_pipe, tts_model) are stored as globals in ai_models.py
    initialize_all_models()
    
    logging.info("🎉 Starting aiohttp server...")
    
    # 3. Create and configure aiohttp application
    app = web.Application()
    app.router.add_get('/', handle_http)             # Serve HTML client
    app.router.add_get('/ws', websocket_handler)     # WebSocket endpoint for WebRTC signaling
    app.router.add_get('/favicon.ico', handle_favicon) # Serve favicon
    
    # 4. Setup and start the server runner
    runner = web_runner.AppRunner(app)
    await runner.setup()
    
    server_host = config.SERVER_HOST
    # Prioritize PORT env var, then config, then a default fallback for safety
    server_port = int(os.environ.get("PORT", getattr(config, 'SERVER_PORT', 7860)))

    site = web.TCPSite(runner, server_host, server_port)
    try:
        await site.start()
        logging.info(f"✅ Server started successfully!")
        logging.info(f"🌐 HTTP server running on http://{server_host}:{server_port}")
        logging.info(f"🔌 WebSocket endpoint available at ws://{server_host}:{server_port}/ws")
        logging.info("Press Ctrl+C to stop the server...")

        # Keep the server running indefinitely until interrupted
        # This Future will only complete upon cancellation or an exception.
        await asyncio.Future()
    except Exception as e:
        # This might catch errors during site.start() or if the Future is cancelled with an error
        logging.critical(f"💥 Server failed to start or encountered a critical error: {e}", exc_info=True)
    finally:
        logging.info("🧹 Cleaning up server resources...")
        await runner.cleanup()
        logging.info("👋 Server shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main_app())
    except KeyboardInterrupt:
        # This handles Ctrl+C if it happens before or during asyncio.run()
        logging.info("\n👋 Application terminated by user (Ctrl+C at top level).")
    except Exception as e:
        # Catch any other unexpected errors during startup that asyncio.run might raise
        logging.critical(f"💥 Unhandled exception during application startup: {e}", exc_info=True)
