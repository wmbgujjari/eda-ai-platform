from celery import shared_task
import requests
import logging
from celery import Celery
from FG.tasks.newconnection_daywise_tasks import schedule_train_newconnection_daywise
from FG.tasks.demand_tasks import schedule_train_demand
from FG.services.schedule_db_service import load_dynamic_schedules
from FG.tasks.segmentation_tasks import schedule_train_segmentation
from FG.tasks.revenue_tasks import schedule_train_revenue
from FG.tasks.consumption_tasks import schedule_train_consumption  # Import other use case tasks
from redbeat import RedBeatSchedulerEntry


celery = Celery(
    "FG.tasks.schedule_tasks",
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json']
)

@shared_task(name="FG.tasks.schedule_tasks.dynamic_schedule_trigger")
def dynamic_schedule_trigger(use_case, office_id, arg1, arg2, subtask_name, model_name, filter_type):
    """
    Dynamic dispatcher for scheduled tasks.
    - If filter_type == 'day'   → arg1=start_date, arg2=end_date
    - If filter_type == 'month' → arg1=month, arg2=year
    """

    logging.info(
        f"🚀 Dispatching dynamic schedule: {use_case}, office: {office_id}, "
        f"subtask: {subtask_name}, model_name: {model_name}, filter_type: {filter_type}"
    )

    # Build task args depending on filter_type
    if filter_type == "day":
        start_date, end_date = arg1, arg2
        task_args = [start_date, end_date, office_id, model_name]
    elif filter_type == "month":
        month, year = arg1, arg2
        task_args = [ year,month,office_id, model_name]
    else:
        logging.error(f"❌ Unknown filter_type: {filter_type}")
        return

    try:
        if use_case == "newconnection_daywise":
            schedule_train_newconnection_daywise.apply_async(args=task_args, queue="newconnection_daywise_queue")

        elif use_case == "consumption":
            schedule_train_consumption.apply_async(args=task_args, queue="consumption_queue")

        elif use_case == "segmentation":
            schedule_train_segmentation.apply_async(args=task_args, queue="segmentation_queue")

        elif use_case == "demand":
            schedule_train_demand.apply_async(args=task_args, queue="demand_queue")

        elif use_case == "revenue":
            schedule_train_revenue.apply_async(args=task_args, queue="revenue_queue")            

        else:
            logging.error(f"❌ Unknown use_case: {use_case}")
            return

        logging.info(f"✅ Task for {use_case} submitted successfully with args {task_args}")

    except Exception as e:
        logging.error(f"❌ Failed to dispatch training for {use_case}: {e}")


#this method is used to schedule usecase one after another 
#@shared_task(name="tasks.schedule_tasks.dynamic_schedule_trigger")
#def dynamic_schedule_trigger(use_case, start_date, end_date,office_id, subtask_name):
#    logging.info(f"🚀 Triggering use case: {use_case}, office: {office_id}, subtask: {subtask_name}")
#    url = f"http://127.0.0.1:8000/{use_case}/train/{start_date}/{end_date}/{office_id}"
#
#    try:
#        response = requests.post(url)
#        logging.info(f"✅ Triggered {use_case} training: {response.status_code} - {response.text}")
#    except Exception as e:
#        logging.error(f"❌ Failed to trigger training for {use_case}: {e}")
        
        
        
