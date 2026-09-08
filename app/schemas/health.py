from datetime import datetime
from typing import Dict
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    components: Dict[str, str]
