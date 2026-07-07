from celery import Celery
from celery import shared_task

# Define Celery app
app = Celery('celery_task', broker='redis://localhost:6379/0',backend='redis://localhost:6379/0')

app.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json']
)

# Define a Celery task
@shared_task
def add(x,y):
    print(x,y)
    return x + y