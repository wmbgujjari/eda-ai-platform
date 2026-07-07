from fastapi import APIRouter, BackgroundTasks, HTTPException,Depends
from FG.models.dynamic_schema import QueryPreviewRequest, DynamicTrainRequest
from FG.services.dynamic_data_service import execute_dynamic_query
from FG.tasks.dynamic_tasks import fetch_data_and_train
from fastapi import Request
from uuid import uuid4
from celery import signature
from FG.core.utils.common_db_service import get_division_section_map
from FG.core.utils.log_and_progress import tracker_log_and_progress
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/dynamiccontroller", tags=["Dynamic Training Controller"])

@router.post("/preview")
async def preview_query_schema(request: QueryPreviewRequest):
    try:
        df = execute_dynamic_query(request.query + " FETCH FIRST 5 ROWS ONLY")
        return {"columns": list(df.columns)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.post("/dynamic-train")
async def train_dynamic_consumption(request: DynamicTrainRequest):
    try:
        task_id = str(uuid4())
        # Extract from request model
        user_query = request.query
        feature_columns = request.feature_columns
        output_column = request.target_column
        officeId =  request.officeId
        if not user_query or not feature_columns or not output_column:
            raise HTTPException(status_code=400, detail="Query, feature columns, and output column are required.")

        division_section_map = await get_division_section_map(officeId)
        if not division_section_map:
            raise HTTPException(status_code=404, detail="No divisions or sections found.")

        tracker_log_and_progress(task_id, "🚀 Dynamic training initiated...")

        for division_id, section_ids in division_section_map.items():
            if not section_ids:
                continue

            print(f"🎯 Queuing task for division {division_id} with {len(section_ids)} sections")

            # Convert section IDs into comma-separated string for SQL
            section_id_list = ",".join([f"'{sid}'" for sid in section_ids])

            # Inject section filter into the query
            if "where" in user_query.lower():
                modified_query = f"{user_query} AND om2.ID IN ({section_id_list})"
            else:
                modified_query = f"{user_query} WHERE om2.ID IN ({section_id_list})"

            # Queue training with the modified query
            signature(
                "FG.tasks.dynamic_tasks.fetch_data_and_train",
                args=[section_ids, task_id, modified_query, feature_columns, output_column],
                queue="dynamic_queue"
            ).apply_async()
        tracker_log_and_progress(task_id, "✅ All dynamic training tasks queued.")
        return {"message": "Dynamic consumption model training started in background.", "task_id": task_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")