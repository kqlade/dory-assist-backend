import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional for production, but useful locally

class Settings:
    # --- Database ---
    DATABASE_URL = os.environ.get("DATABASE_URL")
    DATABASE_PUBLIC_URL = os.environ.get("DATABASE_PUBLIC_URL")

    # --- Redis ---
    REDIS_URL = os.environ.get("REDIS_URL")

    # --- OpenAI / LLM ---
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    OPENAI_REASON_MODEL = os.environ.get("OPENAI_REASON_MODEL", OPENAI_MODEL)
    OPENAI_SEARCH_MODEL = os.environ.get("OPENAI_SEARCH_MODEL", OPENAI_MODEL)
    OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "30"))

    # --- Telnyx (SMS) ---
    TELNYX_API_KEY = os.environ.get("TELNYX_API_KEY")
    TELNYX_PUBLIC_KEY = os.environ.get("TELNYX_PUBLIC_KEY")
    TELNYX_FROM_NUMBER = os.environ.get("TELNYX_FROM_NUMBER")

    # --- Railway metadata (optional) ---
    RAILWAY_ENVIRONMENT = os.environ.get("RAILWAY_ENVIRONMENT")
    RAILWAY_ENVIRONMENT_ID = os.environ.get("RAILWAY_ENVIRONMENT_ID")
    RAILWAY_ENVIRONMENT_NAME = os.environ.get("RAILWAY_ENVIRONMENT_NAME")
    RAILWAY_PROJECT_ID = os.environ.get("RAILWAY_PROJECT_ID")
    RAILWAY_PROJECT_NAME = os.environ.get("RAILWAY_PROJECT_NAME")
    RAILWAY_PRIVATE_DOMAIN = os.environ.get("RAILWAY_PRIVATE_DOMAIN")
    RAILWAY_PUBLIC_DOMAIN = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    RAILWAY_SERVICE_DORY_ASSIST_BACKEND_URL = os.environ.get("RAILWAY_SERVICE_DORY_ASSIST_BACKEND_URL")
    RAILWAY_SERVICE_ID = os.environ.get("RAILWAY_SERVICE_ID")
    RAILWAY_SERVICE_NAME = os.environ.get("RAILWAY_SERVICE_NAME")
    RAILWAY_STATIC_URL = os.environ.get("RAILWAY_STATIC_URL")

    # --- Agent Asset Paths and Prompt Files ---
    AGENT_ASSET_DIR = os.environ.get("AGENT_ASSET_DIR", "app/services/agent_assets")
    AGENT_PROMPT_FILE = os.environ.get("AGENT_PROMPT_FILE", "prompt.txt")
    AGENT_MODULES_FILE = os.environ.get("AGENT_MODULES_FILE", "modules.txt")
    AGENT_LOOP_FILE = os.environ.get("AGENT_LOOP_FILE", "agent_loop.txt")

    # --- Default Limits and Timezone ---
    DEFAULT_MAX_CHARS = int(os.environ.get("DEFAULT_MAX_CHARS", "10000"))
    DEFAULT_MAX_RESULTS = int(os.environ.get("DEFAULT_MAX_RESULTS", "3"))
    DEFAULT_LOOKUP_LIMIT = int(os.environ.get("DEFAULT_LOOKUP_LIMIT", "5"))
    DEFAULT_TIMEZONE = os.environ.get("DEFAULT_TIMEZONE", "America/Los_Angeles")

    # --- Photo Metadata Timeout ---
    PHOTO_METADATA_TIMEOUT = int(os.environ.get("PHOTO_METADATA_TIMEOUT", "10"))

    # --- Add more as needed ---

settings = Settings()
