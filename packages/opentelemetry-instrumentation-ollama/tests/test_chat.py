import pytest
import ollama
from opentelemetry.semconv_ai import SpanAttributes


# Helper function to check span attributes
def check_ollama_span(ollama_span, model, content, response, streaming=False):
    assert ollama_span.name == "ollama.chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_SYSTEM}") == "Ollama"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_TYPE}") == "chat"
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_IS_STREAMING}") == streaming
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_REQUEST_MODEL}") == model
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_PROMPTS}.0.content") == content
    assert ollama_span.attributes.get(f"{SpanAttributes.LLM_COMPLETIONS}.0.content") == response
    assert ollama_span.attributes.get(SpanAttributes.LLM_USAGE_PROMPT_TOKENS) == 17
    assert ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    ) == ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_COMPLETION_TOKENS
    ) + ollama_span.attributes.get(
        SpanAttributes.LLM_USAGE_PROMPT_TOKENS
    )


@pytest.fixture
def test_input():
    return {
        "model": "llama3",
        "messages": [
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            },
        ]
    }


@pytest.mark.vcr
def test_ollama_chat(exporter, test_input):
    response = ollama.chat(**test_input)

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    check_ollama_span(
        ollama_span,
        model="llama3",
        content=test_input["messages"][0]["content"],
        response=response["message"]["content"],
        streaming=False
    )


@pytest.mark.vcr
def test_ollama_streaming_chat(exporter, test_input):
    gen = ollama.chat(**test_input, stream=True)

    response = "".join(res["message"]["content"] for res in gen)

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    check_ollama_span(
        ollama_span,
        model="llama3",
        content=test_input["messages"][0]["content"],
        response=response,
        streaming=True
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_chat(exporter, test_input):
    client = ollama.AsyncClient()
    response = await client.chat(**test_input)

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    check_ollama_span(
        ollama_span,
        model="llama3",
        content=test_input["messages"][0]["content"],
        response=response["message"]["content"],
        streaming=False
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_ollama_async_streaming_chat(exporter, test_input):
    client = ollama.AsyncClient()
    gen = await client.chat(**test_input, stream=True)

    response = ""
    async for res in gen:
        response += res["message"]["content"]

    spans = exporter.get_finished_spans()
    ollama_span = spans[0]
    check_ollama_span(
        ollama_span,
        model="llama3",
        content=test_input["messages"][0]["content"],
        response=response,
        streaming=True
    )
