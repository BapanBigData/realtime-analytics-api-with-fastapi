import logging
from src.config.config import LOG_FILE


file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logging.getLogger().handlers.clear()
logging.getLogger().addHandler(file_handler)
logging.getLogger().setLevel(logging.INFO)