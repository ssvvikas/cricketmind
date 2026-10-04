"""API configuration, read from environment variables (prefix ``CRICKETMIND_API_``).

Defaults are safe for running on your own machine; before exposing the server to other people set
``CRICKETMIND_API_ACCESS_TOKEN`` (every chat request then needs it) and keep the limits modest,
because every chat message spends API credit.
"""

from pydantic_settings import BaseSettings

EXAMPLES = [
    "Who are the top 5 batters in the club leagues right now?",
    "Who should open the bowling to Kohli in the powerplay: Bumrah, Narine or Rashid Khan?",
    "Who is more likely to win a T20I between India and Australia?",
    "How does Kohli's strike rate in the death overs compare with the powerplay?",
    "Who won the Orange Cap in IPL 2016 and how many runs did they score?",
    "Which IPL matches were decided on the last ball, and how many have there been?",
    "What is the biggest successful run chase in IPL history?",
    "Which of Kohli, Rohit Sharma, Buttler and Suryakumar Yadav would make the best opening pair?",
]


class ApiSettings(BaseSettings):
    host: str = "127.0.0.1"  # loopback only until you decide otherwise
    port: int = 8000
    access_token: str | None = None  # when set, chat endpoints require it
    max_message_chars: int = 1000
    max_sessions: int = 200
    session_ttl_seconds: int = 1800  # idle conversations are forgotten after 30 minutes
    max_turns_per_session: int = 20  # bounds both context size and spend per conversation
    rate_limit_per_minute: int = 10  # chat requests per client address
    trust_proxy_headers: bool = False  # honour X-Forwarded-For (only behind your own proxy)
    model: str = "claude-opus-5-5"
    refusal_fallbacks: bool = True
    warmup_on_start: bool = True

    model_config = {"env_prefix": "CRICKETMIND_API_"}
