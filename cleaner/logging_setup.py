import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler


class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("job_id", "operation", "seconds", "peak_rss", "error_code"):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data["exception_type"] = record.exc_info[0].__name__
        return json.dumps(data)


def configure(root):
    directory = root / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cleaner")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(directory / "app.jsonl", maxBytes=5 * 1024 * 1024, backupCount=3)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
