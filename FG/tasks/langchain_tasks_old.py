import logging
from datetime import datetime
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from core.config import OPENAI_API_KEY
from celery import shared_task
from celery import Celery
from core.utils.log_and_progress import tracker_log_and_progress
from core.database import get_db_connection, session_pool

# Optional: import your DB connection for persistent summary logging
  # you’ll define this helper later
celery = Celery(
    "tasks.langchain_tasks",
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json']
)

# Initialize the OpenAI model (uses your environment API key)
llm = OpenAI(
    temperature=0.3,
    openai_api_key=OPENAI_API_KEY
)

# In-memory storage during live runs (task_id -> list of messages)
reasoning_memory = {}

# ---------- LIVE TRAINING PHASE ----------

@shared_task
def log_reasoning_to_langchain(task_id: str, message: str, context: str = None):
    """
    Logs a reasoning step to LangChain memory during a live task.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "context": context or "Training Progress",
        "message": message
    }

    # Store in in-memory log
    if task_id not in reasoning_memory:
        reasoning_memory[task_id] = []
    reasoning_memory[task_id].append(entry)

    logging.info(f"[LangChain] ({task_id}) {message}")

    # Send message to LangChain agent (keeps conversational reasoning trace)
    try:
        llm.invoke([
            SystemMessage(content=f"Task ID: {task_id} | Context: {context}"),
            HumanMessage(content=message)
        ])
    except Exception as e:
        logging.warning(f"LangChain log failed: {e}")


# ---------- POST TRAINING PHASE ----------
@shared_task
def get_langchain_summary(task_id: str):
    """
    Summarizes all reasoning logs for a completed task using LangChain.
    """
    logs = reasoning_memory.get(task_id, [])
    if not logs:
        return "No reasoning logs available for this task."

    combined_text = "\n".join(
        [f"[{x['timestamp']}] {x['context']}: {x['message']}" for x in logs]
    )

    try:
        summary_prompt = (
            "Summarize the key steps, data insights, and reasoning behind this model training:\n\n"
            f"{combined_text}"
        )
        summary = llm.invoke([HumanMessage(content=summary_prompt)])
        return summary.content if hasattr(summary, 'content') else str(summary)
    except Exception as e:
        logging.error(f"LangChain summary generation failed: {e}")
        return "Error generating reasoning summary."

@shared_task
def save_reasoning_log_to_db(model_name: str, section_id: str, task_id: str):
    """
    Persists summarized reasoning to a database table for chatbot use.
    """
    summary = get_langchain_summary(task_id)
    record = {
        "model_name": model_name,
        "section_id": section_id,
        "task_id": task_id,
        "summary": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Convert to JSON if needed
    try:
        insert_reasoning_log(record)
        logging.info(f"[LangChain] Summary stored for model={model_name}, section={section_id}")
    except Exception as e:
        logging.error(f"Failed to store reasoning summary: {e}")

    return summary

@shared_task
def insert_reasoning_log(record: dict):
    """
    Inserts reasoning logs (from LangChain Agent or training pipeline)
    into the MODEL_REASONING_LOGS table asynchronously.
    """

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            logging.error("❌ Database connection failed for reasoning log insertion.")
            return None

        cursor = conn.cursor()

        # Extract values safely
        model_name = record.get("model_name", "Unknown")
        section_id = record.get("section_id")
        task_id = record.get("task_id")
        summary = record.get("summary", "")
        timestamp = record.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        insert_sql = """
            INSERT INTO MIS_USER.MODEL_REASONING_LOGS (
                MODEL_NAME, SECTION_ID, TASK_ID, SUMMARY, TIMESTAMP
            )
            VALUES (:1, :2, :3, :4, TO_DATE(:5, 'YYYY-MM-DD HH24:MI:SS'))
        """

        cursor.execute(insert_sql, (model_name, section_id, task_id, summary, timestamp))
        conn.commit()

        logging.info(f"🧠 Reasoning log inserted for task_id={task_id}, model={model_name}")
        if task_id:
            tracker_log_and_progress(task_id, f"🧠 Reasoning log inserted for model {model_name}")

    except Exception as e:
        logging.error(f"❌ Failed to insert reasoning log: {e}")
        if record.get("task_id"):
            tracker_log_and_progress(record["task_id"], f"❌ Failed to insert reasoning log: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            session_pool.release(conn)