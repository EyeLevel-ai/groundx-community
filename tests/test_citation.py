# tests/test_citation.py
import os
import logging

# Set up logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True  # Force override if something else already configured it
)

import os
import logging
import pytest
from dotenv import load_dotenv
from chat_utils.citing import generate_cited_response

# Load env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Configure logging once (at module level)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_generate_cited_response_minimal():
    openai_key = os.getenv("OPENAI_API_KEY")

    chunks = [
        {
            "text": "Paris is the capital of France.",
            "uuid": "123e4567-e89b-12d3-a456-426614174000",
            "render_name": "example.txt",
            "source_data": {
                "url": "https://example.com/paris",
                "filename": "example.txt",
                "file_type": "txt",
                "document_uuid": "doc-001"
            }
        }
    ]

    system_prompt = "You are a helpful assistant."
    query = "What is the capital of France?"

    result = await generate_cited_response(
        chunks=chunks,
        system_prompt=system_prompt,
        query=query,
    )

    logger.info("Test result:\n%s", result)

    assert isinstance(result, str)
    assert any(keyword in result for keyword in ("Paris", "France", "<InTextCitation"))
