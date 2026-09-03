#!/usr/bin/env python3
import pytest

from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_alias = "tinyllama-2-gemini"
    server.server_port = 8082
    server.n_slots = 1
    server.n_ctx = 8192
    server.n_batch = 2048

def test_gemini_generate_content_basic():
    """Test basic Gemini generateContent endpoint"""
    server.start()

    res = server.make_request("POST", "/v1beta/models/test:generateContent", data={
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Hello, world!"}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 50,
            "temperature": 0.0
        }
    })
    
    assert res.status_code == 200
    
    data = res.body
    assert "candidates" in data
    assert len(data["candidates"]) > 0
    candidate = data["candidates"][0]
    
    assert "content" in candidate
    assert candidate["content"]["role"] == "model"
    assert "parts" in candidate["content"]
    assert len(candidate["content"]["parts"]) > 0
    assert "text" in candidate["content"]["parts"][0]
    
    text_content = candidate["content"]["parts"][0]["text"]
    assert len(text_content) > 0
    
    assert "usageMetadata" in data
    assert "promptTokenCount" in data["usageMetadata"]
    assert "candidatesTokenCount" in data["usageMetadata"]

def test_gemini_generate_content_stream():
    """Test streaming Gemini streamGenerateContent endpoint"""
    server.start()

    res = server.make_stream_request("POST", "/v1beta/models/test:streamGenerateContent?alt=sse", data={
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Count to 5"}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 20
        }
    })
    
    # Process SSE stream
    chunks = []
    for data in res:
        chunks.append(data)
    
    assert len(chunks) > 0
    
    has_text = False
    has_finish = False
    
    for chunk in chunks:
        if "candidates" in chunk and len(chunk["candidates"]) > 0:
            candidate = chunk["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        has_text = True
            
            if "finishReason" in candidate:
                has_finish = True
    
    assert has_text
    assert has_finish


def test_gemini_count_tokens():
    server.start()

    res = server.make_request("POST", "/v1beta/models/test:countTokens", data={
        "contents": [{
            "role": "user",
            "parts": [{"text": "Hello world"}],
        }],
    })

    assert res.status_code == 200
    assert set(res.body) == {"totalTokens"}
    assert isinstance(res.body["totalTokens"], int)
    assert res.body["totalTokens"] > 0


def test_gemini_count_tokens_generate_content_request():
    server.start()

    res = server.make_request("POST", "/v1beta/models/test:countTokens", data={
        "generateContentRequest": {
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello world"}],
            }],
        },
    })

    assert res.status_code == 200
    assert res.body["totalTokens"] > 0


def test_gemini_rejects_missing_contents():
    server.start()

    res = server.make_request("POST", "/v1beta/models/test:generateContent", data={})

    assert res.status_code == 400
    assert "contents" in res.body["error"]["message"]
