# Installation

To install the most recent version:
```
pip install git+https://github.com/EyeLevel-ai/groundx_community.git
```

To install from a specific commit hash:
```
pip install git+https://github.com/EyeLevel-ai/groundx_community.git@<commit-hash>
```

# Chat Utils

## In-Text Citation
`generate_cited_response` is an augmentation and generation tool designed to generate in-text citations within a RAG system.

```
from groundx_community.chat_utils.citing import generate_cited_response
```


The major input to the `generate_cited_response` function is chunks, which consists of three fields:
1. `text`, which is passed to the LLM
1. `uuid` associated with each chunk, used to reference the in-text citation
1. `render_name`, used to render the in-text citation
1. `source_data`, which is an optional set of keys and values that will be injected into the final output upon generation of the in-text citation.

properly configured chunks have the following general structure:
```
chunks=[
        {
            "text": "text the LLM should use"
            "uuid": "11111111-aaaa-bbbb-cccc-000000000001",
            "render_name": "example.txt",
            "source_data": {
                "key1": "value1".
                "key2": "value2"
            }
        },
        ...
    ]
```

There are also two additional arguments for generating the in-text citation:
1. `system_prompt`, which functions as a system prompt defining application level guidance. Note, additional prompting is done to guide the model to use in-text citations. This prompt should guide the model in terms of application logic, not logic around in-text citations.
1. `query`, the query which the model should answer, based on the chunks

In-text citation defaults to using `GPT-4o`, and expects an `OPENAI_API_KEY` to be defined as an environment variable. If you wish to use OpenAI with GPT-4o, the following code will suffice:

```
# Use default GPT-4o
response = await generate_cited_response(
    chunks=my_chunks,
    system_prompt="You are a helpful assistant.",
    query="What is the verdict?",
)
```

If you want to specify another model as the completion model, the `generate_cited_response` function accepts any langchain chat model that inherits from `langchain_core.language_models.chat_models.BaseChatModel`. For instance, here's an example of using Claude 3 haiku.

```
# Use a custom Claude model
from langchain_anthropic import ChatAnthropic

custom_llm = ChatAnthropic(model="claude-3-haiku-20240307", api_key="your-anthropic-key")

response = await generate_cited_response(
    chunks=my_chunks,
    system_prompt="You are a helpful assistant.",
    query="What is the verdict?",
    llm=custom_llm,
)
```

The end result is a string with the following in-text citations injected:
```
This is a response that needs to be cited<InTextCitation chunkId="11111111-aaaa-bbbb-cccc-000000000001" renderName="example.txt" key1="value1" key2="value2"></InTextCitation>
```

`InTextCitation` can be configured as necessary to allow for arbitrary functionality. For example, in [this example](https://github.com/EyeLevel-ai/groundx_community/blob/main/examples/rag_in_text_citation.ipynb), you can see an example of turning in-text citations into clickable links which open the respective PDF being referenced.

# Testing

to test, `tests` should have a properly configured .env variable. Then run

```uv run -m pytest tests/```

