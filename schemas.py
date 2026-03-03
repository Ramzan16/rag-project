from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any

class PaperData(BaseModel):
    """A Pydantic model for holding a paper's metadata and content stream."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    title: str
    filename: str
    authors: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    
    # Payload fields for streaming
    stream: Optional[Any] = None 
    content_length: Optional[int] = None
    pdf_binary_data: Optional[bytes] = None
