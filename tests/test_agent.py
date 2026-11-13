"""Agente: guardie SQL, tool, loop di streaming con tool-call (LLM simulato, nessuna rete esterna)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chatautopsy.agent import ChatAutopsyAgent


class _MockLLM(BaseHTTPRequestHandler):
    """Server OpenAI-compatible minimale: 1° giro tool-call nativi (2 in parallelo), 2° giro tool-call testuale
    stile Qwen, 3° giro risposta finale con blocco <think> da rimuovere."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "mock-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n_tool = sum(1 for m in req["messages"] if m["role"] == "tool")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(delta, finish=None):
            chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "mock",
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        if req.get("tools") and n_tool == 0:
            send({"role": "assistant", "tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                       "function": {"name": "query_sql_database", "arguments": ""}}]})
            for part in ['{"sql_query": "SELECT sender, COUNT(*) ', 'AS n FROM messages GROUP BY sender"}']:
                send({"tool_calls": [{"index": 0, "function": {"arguments": part}}]})
            send({"tool_calls": [{"index": 1, "id": "c2", "type": "function", "function": {
                "name": "search_chat_sessions", "arguments": '{"query":"\\"bar Sport\\"","top_k":2}'}}]})
            send({}, "tool_calls")
        elif req.get("tools") and n_tool == 2:
            send({"content": '<tool_call>{"name":"analyze_specific_red_flag","arguments":{"red_flag_type":"stonewalling"}}</tool_call>'})
            send({}, "stop")
        else:
            for tok in ["<think>ragiono</think>", "**Prova:** ", "risultati ricevuti. ", "Fine."]:
                send({"content": tok})
            send({}, "stop")
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture(scope="module")
def mock_llm():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_sql_tool_is_read_only(couple_db):
    a = ChatAutopsyAgent(str(couple_db))
    assert "rows" in a.tool_query_sql_database("SELECT COUNT(*) FROM messages")
    assert "error" in a.tool_query_sql_database("DELETE FROM messages")
    assert "error" in a.tool_query_sql_database("SELECT 1; DROP TABLE messages")


def test_tools_return_data(couple_db):
    a = ChatAutopsyAgent(str(couple_db))
    assert a.tool_search_chat_sessions('"bar Sport"', top_k=3)["sessions"]
    for flag in ["gaslighting", "stonewalling", "love_bombing_devaluation", "passive_aggressive", "guilt_tripping"]:
        assert "error" not in a.tool_analyze_specific_red_flag(flag)
    assert len(a.tool_build_persona_profile("Giulia", "pizza")["representative_messages"]) >= 10
    payload, _ = a.execute_tool("get_global_psychometric_dossier", {})
    assert json.loads(payload)


def test_agent_loop_with_streaming_tool_calls(couple_db, mock_llm):
    ag = ChatAutopsyAgent(str(couple_db), base_url=mock_llm, model="mock-model")
    events = list(ag.run("chi scrive di più?"))
    tools = [e["trace"]["name"] for e in events if e["type"] == "tool_result"]
    assert tools == ["query_sql_database", "search_chat_sessions", "analyze_specific_red_flag"]
    assert events[-1]["type"] == "final" and "think" not in events[-1]["content"]


def test_persona_simulator(couple_db, mock_llm):
    ag = ChatAutopsyAgent(str(couple_db), base_url=mock_llm, model="mock-model")
    assert list(ag.simulate("Giulia", "stasera faccio tardi"))[-1]["type"] == "final"


def test_list_models(mock_llm):
    assert ChatAutopsyAgent.list_models(mock_llm) == ["mock-model"]
