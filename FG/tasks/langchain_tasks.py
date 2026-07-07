import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from celery import shared_task
from celery import Celery

from FG.core.database import get_db_connection, session_pool
from FG.core.utils.log_and_progress import tracker_log_and_progress
from FG.core.config import OPENAI_API_KEY  # keep only if you want fallback


# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DEFAULT_OLLAMA_MODEL = "llama3.1"   # Make sure model exists in your machine
USE_OPENAI_FALLBACK = False         # Keep False unless needed


# -------------------------------------------------------------
# LLM CLIENTS
# -------------------------------------------------------------
# PRIMARY → OLLAMA
ollama_llm = ChatOllama(
    model=DEFAULT_OLLAMA_MODEL,
    temperature=0.2
)

# OPTIONAL FALLBACK → OpenAI (only if you want)
if USE_OPENAI_FALLBACK:
    from langchain_openai import OpenAI
    openai_llm = OpenAI(
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY
    )


# Initialize Celery for this module
celery = Celery(
    "tasks.langchain_tasks",
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery.conf.update(
    task_serializer='pickle',
    accept_content=['pickle', 'json']
)

# In-memory storage (task_id → list of log messages)
reasoning_memory = {}



# =======================================================================
#  LIVE TRAINING LOGGING
# =======================================================================
@shared_task
def log_reasoning_to_langchain(task_id: str, message: str, context: str = None):
    """
    Logs a reasoning step to an in-memory store + pushes to LLM for reasoning chain.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "timestamp": timestamp,
        "context": context or "Training Progress",
        "message": message
    }

    # Store locally
    if task_id not in reasoning_memory:
        reasoning_memory[task_id] = []
    reasoning_memory[task_id].append(entry)

    logging.info(f"[LangChain] ({task_id}) {message}")

    # Send message to LLM (Ollama) for maintaining conversational trace
    try:
        ollama_llm.invoke([
            SystemMessage(content=f"Task ID: {task_id} | Context: {context}"),
            HumanMessage(content=message)
        ])
    except Exception as e:
        logging.warning(f"⚠️ Ollama log failed: {e}")

        if USE_OPENAI_FALLBACK:
            try:
                openai_llm.invoke([
                    SystemMessage(content=f"Task ID: {task_id} | Context: {context}"),
                    HumanMessage(content=message)
                ])
            except Exception as e2:
                logging.error(f"⚠️ Fallback OpenAI failed also: {e2}")



# =======================================================================
#  SUMMARY GENERATOR
# =======================================================================
@shared_task
def get_langchain_summary(task_id: str):
    """
    Builds a full reasoning summary from stored logs using Ollama.
    """
    logs = reasoning_memory.get(task_id, [])
    if not logs:
        return "No reasoning logs available for this task."

    combined_text = "\n".join(
        f"[{x['timestamp']}] {x['context']}: {x['message']}"
        for x in logs
    )

    summary_prompt = (
        "You are an expert ML training analyst. Summarize the key reasoning, "
        "training details, improvements, and outcomes based on the logs:\n\n"
        f"{combined_text}"
    )

    try:
        summary = ollama_llm.invoke([HumanMessage(content=summary_prompt)])
        return summary.content

    except Exception as e:
        logging.error(f"❌ Ollama summary generation failed: {e}")

        if USE_OPENAI_FALLBACK:
            try:
                summary = openai_llm.invoke([HumanMessage(content=summary_prompt)])
                return summary.content
            except Exception as e2:
                logging.error(f"❌ OpenAI fallback also failed: {e2}")

        return "Error generating reasoning summary."



# =======================================================================
#  STORE SUMMARY INTO DATABASE
# =======================================================================
@shared_task
def save_reasoning_log_to_db(model_name: str, section_id: str, task_id: str):
    """
    Saves the generated summary into MODEL_REASONING_LOGS DB table.
    """
    summary = get_langchain_summary(task_id)

    record = {
        "model_name": model_name,
        "section_id": section_id,
        "task_id": task_id,
        "summary": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        insert_reasoning_log(record)
        logging.info(f"🧠 Reasoning summary inserted for task_id={task_id}")
    except Exception as e:
        logging.error(f"❌ Failed to save reasoning log: {e}")

    return summary



# =======================================================================
# DB INSERTION
# =======================================================================
@shared_task
def insert_reasoning_log(record: dict):
    """
    Inserts reasoning summary into Oracle DB.
    """

    conn = None
    cursor = None

    try:
        conn = get_db_connection()
        if conn is None:
            logging.error("❌ Database connection failed for reasoning log insertion.")
            return None

        cursor = conn.cursor()

        insert_sql = """
            INSERT INTO MIS_USER.MODEL_REASONING_LOGS (
                MODEL_NAME, SECTION_ID, TASK_ID, SUMMARY, TIMESTAMP
            )
            VALUES (:1, :2, :3, :4, TO_DATE(:5, 'YYYY-MM-DD HH24:MI:SS'))
        """

        cursor.execute(
            insert_sql,
            (
                record.get("model_name", "Unknown"),
                record.get("section_id"),
                record.get("task_id"),
                record.get("summary", ""),
                record.get("timestamp"),
            )
        )
        conn.commit()

        logging.info(
            f"🧠 Reasoning log inserted for model={record.get('model_name')} task={record.get('task_id')}"
        )

        if record.get("task_id"):
            tracker_log_and_progress(record["task_id"], "🧠 Reasoning summary stored in DB.")

    except Exception as e:
        logging.error(f"❌ Failed to insert reasoning log: {e}")

    finally:
        if cursor:
            cursor.close()
        if conn:
            session_pool.release(conn)
