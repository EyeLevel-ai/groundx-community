import os
import re
from urllib import parse
from typing import List, TypedDict, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

class Chunk(TypedDict):
    text: str
    uuid: str
    render_name: str
    source_data: Dict[str, Any]

def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or not key.startswith("sk-"):
        raise ValueError(
            "Invalid or missing OPENAI_API_KEY environment variable. "
            "Ensure it exists and starts with 'sk-'."
        )
    return key

async def generate_cited_response(
    chunks: List[Chunk],
    system_prompt: str,
    query: str,
    llm: Optional[BaseChatModel] = None,
) -> str:
    if llm is None:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=get_openai_api_key(),
        )

    ref_mapping = {}
    chunk_texts = []
    for ref in chunks:
        full_uuid = ref["uuid"]
        chunk = ref["text"]
        chunk_texts.append(f"**ID:** {full_uuid}\n**Text:** {chunk}\n")
        ref_mapping[full_uuid] = ref.copy()

    context = "\n---\n\n".join(chunk_texts) if chunk_texts else ""

    human_prompt = f"""
I am going to ask a question in my next message. Here are some excerpts uniquely identified by an ID that may or may not be relevant.

You need to perform 2 tasks:
    1) Generate a response to answer the question. If these excerpts relate to my question, use them in your response. If not, ignore them, and rely on our conversation context.
    2) If any excerpt is used, generate in-text citation using the excerpt ID as follows - $REF: ID$. The formatting must be strictly followed. For example, if excerpt corresponding to ID 03b994bc-2fae-4e1e-a4cd-f0f3e6db2d90 is used in generating the response, source attribution must be -
    $REF: 03b994bc-2fae-4e1e-a4cd-f0f3e6db2d90$

Strictly follow the instructions of the above tasks.

Do not mention whether the content or previous context was used or not; respond seamlessly. Avoid phrases like "the provided content" or similar.

Take into account everything we've discussed so far, without assuming everything is relevant unless it clearly supports your answer.

-----

{context}
"""

    answer = await llm.ainvoke(
        input=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
            HumanMessage(content=query),
        ]
    )
    answer = answer.content

    pattern = r"\$REF: ([a-f0-9\-]+)\$"
    for match in re.finditer(pattern=pattern, string=answer):
        to_replace = match.group()
        ref_id = match.groups()[0]

        if ref_id in ref_mapping:
            ref = ref_mapping[ref_id]
            props = {
                "chunkId": ref["uuid"],
                "renderName": ref["render_name"],
                **ref.get("source_data", {}),
            }
            prop_str = " ".join(
                f'{key}="{parse.quote(str(value))}"' for key, value in props.items()
            )
            citation = f'\n<InTextCitation {prop_str}></InTextCitation>'
            answer = answer.replace(to_replace, citation)
        else:
            answer = answer.replace(to_replace, "")

    return answer
