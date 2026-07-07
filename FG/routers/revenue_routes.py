from fastapi import APIRouter, BackgroundTasks, HTTPException,Depends
from pydantic import BaseModel
from FG.core.utils.jwt_auth import verify_token,create_jwt_token,verify_test_token
from fastapi import Request
from FG.core.config import EXPECTED_USER_ID,EXPECTED_ROLE
from FG.core.utils.common_db_service import get_division_section_map
from uuid import uuid4
from FG.core.utils.log_and_progress import tracker_log_and_progress
from fastapi.responses import StreamingResponse
from FG.core.utils.redis_client import redis_client
import redis
import time
import json
from celery import signature
#EXPECTED_USER_ID='69d8bb2e-a0a06073-294c684c-2a7a5dcf'
#EXPECTED_ROLE='ROLE_BIHAR'
router = APIRouter(prefix="/revenue", tags=["Revenue Prediction"])



@router.get("/stream-progress/{task_id}")
async def stream_progress(task_id: str):
    def event_stream():
       last_message = None
       while True:
            data = redis_client.get(f"progress:{task_id}")
            if data:
                if isinstance(data, bytes):
                    decoded_data = data.decode("utf-8")
                else:
                    decoded_data = data
                if decoded_data != last_message:
                    last_message = decoded_data
                    yield f"data: {decoded_data}\n\n"
                    progress = json.loads(decoded_data)
                    if progress.get("status") in ("completed", "failed"):
                        break  # Stop streaming when done or failed    
            else:
            # Send a comment to keep connection alive
                yield ": keep-alive\n\n"    
            time.sleep(10)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/train/{train_year}/{train_month}/{officeId}/{model_name}")
async def train_demand(train_year: str,train_month: str,officeId: str,model_name: str):
    print("intrain")
    try:
        task_id = str(uuid4())  
        #user_id = token_payload.get("userId")
        #role = token_payload.get("role")
        #print(f"Expected USER: {EXPECTED_USER_ID}, ROLE: {EXPECTED_ROLE}")
        #print(EXPECTED_USER_ID,EXPECTED_ROLE)
        #if user_id != EXPECTED_USER_ID or role != EXPECTED_ROLE:
        #    raise HTTPException(status_code=403, detail="Unauthorized user or role")
        division_section_map = await get_division_section_map(officeId)
        print(division_section_map)
        if not division_section_map:
            raise HTTPException(status_code=404, detail="No divisions or sections found.")
        tracker_log_and_progress(task_id, "🚀 Training process initiated...")
        # Iterate over the divisions
        for division_id, section_id in division_section_map.items():
            if not section_id:
                continue  # Skip if no sections are found for this division

            print(f"🎯 Queuing task for division {division_id} with {len(section_id)} section(s)")
            # Send to consumption_queue explicitly
            signature("FG.tasks.revenue_tasks.fetch_data_and_train", args=[section_id, task_id, train_year, train_month,model_name], queue="revenue_queue").apply_async()
            
            print("Sending task to celery...")
            # Load the data for the sections associated with this division
            #df, file_size = await load_data(section_id)
            
            #if df.empty:
            #    continue  # Skip if data is empty

            # Convert DataFrame to dictionary format for background task
            #df_dict = df.to_dict(orient="records")       
        #df, file_size = await load_data()
        #Convert df to serializable dict
        #df_dict = df.to_dict(orient="records")
        #Queue background training
            #bg_tasks.add_task(select_or_increment_model.delay, df_dict, file_size)

        #select_or_increment_model.delay(df_dict, file_size)

        tracker_log_and_progress(task_id, "All training tasks queued. Background training started.")
        return {"message": "Revenue model training started in the background.", "task_id": task_id}


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")