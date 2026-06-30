from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Agent identity (used in system prompts — FCC 24-17 requires AI disclosure)
    agent_name: str = "Alex"
    company_name: str = ""           # required before going live — set in .env
    agent_persona: str = "professional, confident, and genuinely helpful"
    call_objective: str = "book a 15-minute discovery call"

    # LLM
    qwen_vllm_endpoint: str = "http://localhost:8000/v1"
    qwen_model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    qwen_api_key: str = "EMPTY"   # set to Bearer token when endpoint requires auth

    # Voice (InWorld)
    inworld_api_key: str = ""
    inworld_stt_endpoint: str = ""   # wss://... — confirm from InWorld docs
    inworld_tts_endpoint: str = ""   # wss://... — confirm from InWorld docs

    # Voice variants — InWorld voice_id for each persona (leave empty to disable)
    inworld_voice_formal: str = ""
    inworld_voice_warm: str = ""
    inworld_voice_energetic: str = ""
    inworld_voice_calm: str = ""
    inworld_voice_casual: str = ""
    inworld_voice_test: str = ""     # fallback for local dev — set to any known voice_id

    # Telephony (FreePBX — separate server)
    freepbx_host: str = ""
    freepbx_ari_user: str = ""
    freepbx_ari_secret: str = ""
    audiosocket_port: int = 9092           # AI agent mode
    audiosocket_voicemail_port: int = 9093 # Voicemail/AMD-detected machine mode

    # Database
    postgres_url: str = "postgresql+asyncpg://coldcallai:password@localhost:5432/coldcallai"
    qdrant_url: str = "http://localhost:6333"

    # Worker queue
    redis_url: str = "redis://localhost:6379/0"

    # Recording storage (MinIO)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    recordings_bucket: str = "coldcallai-recordings"

    # CRM
    activecampaign_api_key: str = ""
    activecampaign_api_url: str = ""

    # Compliance
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""

    # Cost caps
    cost_cap_daily_usd: float = 100.0
    cost_cap_monthly_usd: float = 2000.0
    alert_email: str = "fahadfahim13@gmail.com"


settings = Settings()
