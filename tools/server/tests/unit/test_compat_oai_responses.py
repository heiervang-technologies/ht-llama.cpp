import pytest
from openai import OpenAI
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

def test_responses_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
    )
    assert res.id.startswith("resp_")
    assert res.output[0].id is not None
    assert res.output[0].id.startswith("msg_")
    assert match_regex("(Suddenly)+", res.output_text)

def test_responses_stream_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    stream = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
        stream=True,
    )

    gathered_text = ''
    resp_id = ''
    msg_id = ''
    for r in stream:
        if r.type == "response.created":
            assert r.response.id.startswith("resp_")
            resp_id = r.response.id
        if r.type == "response.in_progress":
            assert r.response.id == resp_id
        if r.type == "response.output_item.added":
            assert r.item.id is not None
            assert r.item.id.startswith("msg_")
            msg_id = r.item.id
        if (r.type == "response.content_part.added" or
            r.type == "response.output_text.delta" or
            r.type == "response.output_text.done" or
            r.type == "response.content_part.done"):
            assert r.item_id == msg_id
        if r.type == "response.output_item.done":
            assert r.item.id == msg_id

        if r.type == "response.output_text.delta":
            gathered_text += r.delta
        if r.type == "response.completed":
            assert r.response.id.startswith("resp_")
            assert r.response.output[0].id is not None
            assert r.response.output[0].id.startswith("msg_")
            assert gathered_text == r.response.output_text
            assert match_regex("(Suddenly)+", r.response.output_text)


# Issue #19: truncated responses must signal status=incomplete +
# incomplete_details, otherwise agentic clients can't tell the response was
# cut short and may try to use partial output as conversation history.
def test_responses_truncation_emits_incomplete_status():
    global server
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": "tinyllama-2",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        # tinyllama2 has no eos in this generation pattern, so a small cap
        # reliably trips STOP_TYPE_LIMIT before any natural stop.
        "max_output_tokens": 2,
        "temperature": 0.8,
    })
    assert res.status_code == 200, res.body
    body = res.body
    assert body["status"] == "incomplete", body
    assert body.get("incomplete_details", {}).get("reason") == "max_output_tokens", body
    # downstream items inherit the same status so agent clients can detect
    # partial tool_calls / partial messages at the per-item level
    for item in body["output"]:
        assert item.get("status") == "incomplete", item


def test_responses_truncation_stream_emits_incomplete_event():
    global server
    server.start()
    # use the raw POST so we get raw SSE; the openai client coerces
    # response.incomplete into something different.
    res = server.make_stream_request("POST", "/v1/responses", data={
        "model": "tinyllama-2",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 2,
        "temperature": 0.8,
        "stream": True,
    })
    saw_incomplete = False
    final_status = None
    final_reason = None
    for chunk in res:
        if not chunk:
            continue
        # chunks come as parsed dicts from the helper
        ctype = chunk.get("type")
        if ctype == "response.incomplete":
            saw_incomplete = True
            r = chunk.get("response", {})
            final_status = r.get("status")
            final_reason = (r.get("incomplete_details") or {}).get("reason")
    assert saw_incomplete, "no response.incomplete event in stream"
    assert final_status == "incomplete"
    assert final_reason == "max_output_tokens"
