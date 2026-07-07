import json
import threading
import time
from langchain_openai import ChatOpenAI
from FG.core.utils.redis_client import redis_client
from FG.core.config import OPENAI_API_KEY


class ProgressListener:
    """
    Listens to live training progress updates from Redis (Pub/Sub)
    and uses LangChain's ChatOpenAI model to summarize insights in real-time.

    Works in two modes:
      1. Passive polling via `get_latest_progress()`
      2. Active listening via Redis Pub/Sub `start_listening()`
    """

    def __init__(self, channel: str = "task_progress_channel", task_id: str = None):
        self.channel = channel
        self.task_id = task_id
        self.pubsub = redis_client.pubsub()
        self.pubsub.subscribe(self.channel)

        # 🧠 Initialize LLM (ChatOpenAI replaces initialize_agent in v1.x)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )

        # Store message history (for debugging or conversational continuity)
        self.messages = []

    # -------------------------------------------------------------------------
    # Passive Mode — Read latest from Redis
    # -------------------------------------------------------------------------
    def get_latest_progress(self):
        """Fetch the most recent task progress snapshot from Redis."""
        if not self.task_id:
            raise ValueError("task_id must be provided for passive progress polling.")
        data = redis_client.get(f"progress:{self.task_id}")
        return json.loads(data) if data else None

    # -------------------------------------------------------------------------
    # Active Mode — Listen to Pub/Sub Channel
    # -------------------------------------------------------------------------
    def start_listening(self, interval: int = None):
        """
        Starts a background thread to listen for Redis Pub/Sub updates
        and trigger LangChain reasoning per update.
        Optionally sleeps between cycles if interval is provided.
        """
        def listen():
            print(f"👂 Listening for live updates on '{self.channel}' ...")
            for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        self.messages.append(data)

                        if self.task_id and data.get("task_id") != self.task_id:
                            continue

                        msg_text = data.get("message", "")
                        print(f"🧩 Progress Update: {msg_text}")
                        self.summarize_progress(msg_text)

                        if interval:
                            time.sleep(interval)

                    except Exception as e:
                        print(f"[LangChain Listener Error] {e}")

        thread = threading.Thread(target=listen, daemon=True)
        thread.start()
    # -------------------------------------------------------------------------
    # Summarization with ChatOpenAI
    # -------------------------------------------------------------------------
    def summarize_progress(self, message: str):
        """
        Uses the ChatOpenAI model to summarize or interpret a training log message.
        """
        try:
            prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant summarizing machine learning training updates."
                },
                {
                    "role": "user",
                    "content": f"Summarize this ML training update briefly: '{message}'"
                }
            ]
            response = self.llm.invoke(prompt)
            print(f"[LangChain Insight] {response.content}")

        except Exception as e:
            print(f"[LangChain Summarization Error] {e}")

    # -------------------------------------------------------------------------
    # Manual Monitoring Loop
    # -------------------------------------------------------------------------
    def start_polling(self, interval: int = 10):
        """
        Polls Redis periodically for the latest progress (fallback if Pub/Sub is unavailable).
        """
        def monitor():
            print(f"🕒 Polling Redis every {interval}s for task_id={self.task_id}")
            while True:
                progress = self.get_latest_progress()
                if progress:
                    msg = progress.get("message", "")
                    print(f"📊 {msg}")
                    self.summarize_progress(msg)
                time.sleep(interval)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
