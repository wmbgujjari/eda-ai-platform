# routers/segmentation.py
from fastapi import APIRouter, HTTPException
from uuid import uuid4
from celery import signature
from FG.core.utils.common_db_service import get_division_section_map  # same helper you’re using
from FG.core.utils.log_and_progress import tracker_log_and_progress         # your progress logger
from fastapi.responses import StreamingResponse
from FG.core.utils.redis_client import redis_client
import asyncio  
import json
from fastapi import Request
import logging 
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/segmentation", tags=["Segmentation"])

@router.get("/stream-progress/{task_id}")
async def stream_progress(task_id: str, request: Request):
    async def event_stream():
        last_message = None

        logger.info(f"📡 Started streaming for task {task_id}")

        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"❌ Client disconnected for task {task_id}")
                    break

                data = redis_client.get(f"progress:{task_id}")
                if data:
                    if isinstance(data, bytes):
                        decoded_data = data.decode("utf-8")
                    else:
                        decoded_data = data

                    if decoded_data != last_message:
                        last_message = decoded_data
                        logger.info(f"📡 New progress message: {decoded_data}")
                        yield f"data: {decoded_data}\n\n"

                        try:
                            progress = json.loads(decoded_data)
                            if progress.get("status") in ("completed", "failed", "no_data", "error"):
                                logger.info(f"✅ Task {task_id} finished with status: {progress.get('status')}")
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"❌ JSON decode failed: {decoded_data}")
                            continue
                else:
                    yield ": keep-alive\n\n"

                await asyncio.sleep(2)  # Polling interval

        finally:
            yield f"data: {{\"status\": \"error\", \"message\": \"Streaming closed\"}}\n\n"
            logger.info(f"🔚 Streaming ended for task {task_id}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/train/{start_date}/{end_date}/{officeId}/{model_name}")
async def train_segmentation(start_date: str, end_date: str, officeId: str, model_name: str):
    """
    Trigger consumer segmentation training for all divisions/sections of an office.
    This queues Celery tasks (one per division-section) for background processing.
    """
    try:
        task_id = str(uuid4())

        # 1. Get mapping of office -> divisions -> sections
        division_section_map = await get_division_section_map(officeId)
        if not division_section_map:
            raise HTTPException(status_code=404, detail="No divisions or sections found.")

        tracker_log_and_progress(task_id, "🚀 Segmentation training initiated...")

        # 2. Iterate over divisions
        for division_id, section_ids in division_section_map.items():
            if not section_ids:
                continue  # Skip divisions with no sections

            print(f"🎯 Queuing segmentation task for division {division_id} with {len(section_ids)} section(s)")

            # 3. Queue Celery task (mirrors your newconnection pattern)
            signature(
                "FG.tasks.segmentation_tasks.fetch_and_train_segmentation",
                args=[section_ids, task_id, start_date, end_date, model_name],
                queue="segmentation_queue"
            ).apply_async()

            print("📨 Segmentation task sent to Celery...")

        tracker_log_and_progress(task_id, "✅ All segmentation tasks queued. Background training started.")
        return {"message": "Segmentation training started in the background.", "task_id": task_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation training failed: {str(e)}")
