"""
Main FastAPI app for ChunScraper.
Handles the web server and API endpoints.
"""

import os
from dotenv import load_dotenv
import shutil

# Load .env file.
load_dotenv()

# Clean up old stuff from previous runs before starting.
def cleanup_startup():
    # Cleans out temp folders and logs.
    print("Performing startup cleanup...")
    folders_to_clean = ["downloads", "logs", "temp_scrapers"]
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"Successfully removed directory: {folder}")
            except OSError as e:
                print(f"Error removing directory {folder}: {e}")

cleanup_startup()


# Setup my logger.
from logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import asyncio

from agent import run_scraping_job

# Init FastAPI.
app = FastAPI(
    title="ChunScraper",
    description="An AI agent that writes and tests web scrapers.",
    version="1.0.0"
)

logger.info("ChunScraper starting up")

# Serve static files from the 'static' directory.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Make sure these folders exist.
os.makedirs("downloads", exist_ok=True)
os.makedirs("temp_scrapers", exist_ok=True)


class ScrapeRequest(BaseModel):
    # What the /api/scrape endpoint expects.
    url: str
    prompt: str
    settings: Dict[str, Any]


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # Serves the main index.html.
    logger.info("Serving frontend page")
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        logger.debug("Successfully loaded frontend HTML")
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error("Frontend HTML file not found at static/index.html")
        raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/static/settings.html", response_class=HTMLResponse)
async def serve_settings_page():
    # Serves the settings.html page.
    logger.info("Serving settings page")
    try:
        with open("static/settings.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error("Settings HTML file not found at static/settings.html")
        raise HTTPException(status_code=404, detail="Settings page not found")


@app.get("/api/scrape")
async def scrape_endpoint_sse(request: Request):
    # SSE endpoint to stream job progress back to the client.
    url = request.query_params.get('url')
    prompt = request.query_params.get('prompt')
    settings_str = request.query_params.get('settings')

    if not all([url, prompt, settings_str]):
        raise HTTPException(status_code=400, detail="Missing required query parameters")

    try:
        settings = json.loads(settings_str)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid settings format")

    logger.info(f"Received SSE scrape request for URL: {url}")

    async def event_generator():
        queue = asyncio.Queue()
        job_done = asyncio.Event()
        loop = asyncio.get_running_loop()

        def log_callback(message):
            try:
                # This runs in a sync function, so need to be thread-safe.
                loop.call_soon_threadsafe(queue.put_nowait, message)
            except Exception as e:
                logger.error(f"Error in log_callback: {e}")

        async def job_task():
            try:
                # Run the main scraping logic in a thread to not block asyncio.
                result = await asyncio.to_thread(run_scraping_job, prompt, url, settings, log_callback)
                
                # Let the logs know the download is ready.
                if result.get("status") in ["success", "action_required"] and "download_url" in result:
                    run_id = result["download_url"].split("/")[-1]
                    logger.info(f"SSE Job completed. Download will be available for run_id: {run_id}")

                await queue.put(json.dumps(result))
            except Exception as e:
                logger.error(f"Error in job_task: {e}", exc_info=True)
                error_result = {"status": "error", "message": f"Internal server error: {str(e)}"}
                await queue.put(json.dumps(error_result))
            finally:
                job_done.set()

        # Kick off the job in the background.
        task = asyncio.create_task(job_task())

        try:
            while not job_done.is_set():
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if isinstance(message, str) and message.startswith('{'):
                        # This is the final JSON result.
                        yield f"data: {message}\n\n"
                    else:
                        # This is just a log message.
                        log_data = {"log": message}
                        yield f"data: {json.dumps(log_data)}\n\n"
                    queue.task_done()
                except asyncio.TimeoutError:
                    # Check if the user closed the browser.
                    if await request.is_disconnected():
                        logger.warning("Client disconnected, cancelling job.")
                        task.cancel()
                        break
                    continue
        finally:
            # Make sure the task is cleaned up if the client disconnects.
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/scrape")
async def scrape_endpoint(request: ScrapeRequest) -> Dict[str, Any]:
    # The main endpoint that starts a scraping job.
    
    logger.info(f"Received scrape request for URL: {request.url}")
    logger.debug(f"User prompt: {request.prompt}")
    
    # Basic input validation.
    if not request.url or not request.prompt:
        logger.warning("Invalid request: missing URL or prompt")
        raise HTTPException(
            status_code=400, 
            detail="Both 'url' and 'prompt' are required"
        )
    
    if not request.url.startswith(("http://", "https://")):
        logger.warning(f"Invalid URL format: {request.url}")
        raise HTTPException(
            status_code=400, 
            detail="URL must start with http:// or https://"
        )
    
    try:
        logger.info("Starting scraping job")
        # This is where the magic happens.
        result = run_scraping_job(request.prompt, request.url, request.settings)
        
        # Log so I can find the download link later.
        if "download_url" in result:
            run_id = result["download_url"].split("/")[-1]
            logger.info(f"Job completed successfully. Download will be available for run_id: {run_id}")
        
        logger.info(f"Job status: {result.get('status', 'unknown')}")
        return result
    
    except Exception as e:
        logger.error(f"Error in scrape endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/api/download/{run_id}")
async def download_endpoint(run_id: str):
    # Lets the user download the generated scraper as a ZIP file.
    
    logger.info(f"Download request for run_id: {run_id}")
    
    # Figure out the path to the zip file.
    zip_path = os.path.join("downloads", f"{run_id}.zip")
    logger.debug(f"Serving zip file from path: {zip_path}")
    
    # Make sure the file is actually there.
    if not os.path.exists(zip_path):
        logger.error(f"Zip file not found at: {zip_path}")
        raise HTTPException(status_code=404, detail="Download file not found")
    
    logger.info(f"Successfully serving download for run_id: {run_id}")
    # Send the file.
    return FileResponse(
        path=zip_path,
        filename=f"scraper_{run_id}.zip",
        media_type="application/zip"
    )


@app.get("/api/health")
async def health_check():
    # A simple endpoint to check if the server is up.
    return {
        "status": "healthy",
        "service": "ChunScraper",
        "gemini_api_configured": "Client-side configured"
    }


@app.get("/api/downloads")
async def list_downloads():
    # Lists all the ZIP files available for download.
    try:
        downloads_dir = "downloads"
        if not os.path.exists(downloads_dir):
            return {"downloads": []}
        
        downloads = []
        for filename in os.listdir(downloads_dir):
            if filename.endswith('.zip'):
                file_path = os.path.join(downloads_dir, filename)
                stat = os.stat(file_path)
                downloads.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "created": stat.st_ctime,
                    "run_id": filename.replace('.zip', '')
                })
        
        downloads.sort(key=lambda x: x['created'], reverse=True)
        return {"downloads": downloads}
    
    except Exception as e:
        logger.error(f"Error listing downloads: {e}")
        return {"downloads": [], "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting ChunScraper application")
    
    # Run it.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["temp_scrapers/**", "downloads/**", "logs/**"]
    )