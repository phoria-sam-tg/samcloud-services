"""Ollama output truncation survives both OpenAI chat response paths.

Run: python -m unittest -v ollama.test_ollama_finish_reason
Real ASGI route/renderer; provider and admission are offline stubs. No lifespan,
GPU allocation, registry call or external network access.
"""
import json
import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ['AUTH_ENABLED'] = '0'
os.environ.setdefault('SC_TOKEN', 'offline-test-not-a-credential')
import httpx
from . import server


class FinishReasonTests(unittest.IsolatedAsyncioTestCase):
    async def response(self, stream, reason, tools=False):
        message = {'content': 'partial answer'}
        if tools:
            message['tool_calls'] = [{'function': {'name': 'example', 'arguments': {'x': 1}}}]
        terminal = {'done': True, 'message': {'content': ''},
                    'eval_count': 4, 'prompt_eval_count': 3}
        if reason is not None:
            terminal['done_reason'] = reason
        chunks = [{'message': message, 'done': False}, terminal]
        calls = []

        def chat(*args, **kwargs):
            calls.append(kwargs)
            return iter(chunks)

        async def chat_stream(*args, **kwargs):
            calls.append(kwargs)
            for chunk in chunks:
                yield chunk

        manager = SimpleNamespace(
            ollama=SimpleNamespace(chat=chat, chat_stream=chat_stream),
            serving=lambda _: nullcontext())
        model = SimpleNamespace(name='test-model', backend=server.Backend.OLLAMA)
        with patch.object(server, 'mgr', manager), \
             patch.object(server, '_resolve_model', AsyncMock(return_value=model)), \
             patch.object(server.prompt_size, 'count', return_value=(3, 'fixture')):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                         base_url='http://offline') as client:
                response = await client.post('/v1/chat/completions', json={
                    'model': 'test-model', 'messages': [{'role':'user','content':'hello'}],
                    'stream': stream, 'max_tokens': 4})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(calls[0]['options']['num_predict'], 4)
        if stream:
            lines = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
            self.assertEqual(lines[-1], '[DONE]')
            frames = [json.loads(line) for line in lines[:-1]]
            self.assertEqual(sum(f['choices'][0]['finish_reason'] is not None for f in frames), 1)
            self.assertEqual(frames[0]['choices'][0]['delta']['content'], 'partial answer')
            return frames[-1]['choices'][0]['finish_reason']
        result = response.json()
        self.assertEqual(result['choices'][0]['message']['content'], 'partial answer')
        self.assertEqual(result['usage']['completion_tokens'], 4)
        return result['choices'][0]['finish_reason']

    async def test_stream_length(self):
        self.assertEqual(await self.response(True, 'length'), 'length')

    async def test_collected_length(self):
        self.assertEqual(await self.response(False, 'length'), 'length')

    async def test_stream_length_wins_over_tool_calls(self):
        self.assertEqual(await self.response(True, 'length', tools=True), 'length')

    async def test_collected_length_wins_over_tool_calls(self):
        self.assertEqual(await self.response(False, 'length', tools=True), 'length')

    async def test_normal_stop_and_tools_unchanged(self):
        for stream in (False, True):
            for tools in (False, True):
                with self.subTest(stream=stream, tools=tools):
                    self.assertEqual(await self.response(stream, 'stop', tools),
                                     'tool_calls' if tools else 'stop')

    async def test_legacy_missing_reason_unchanged(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assertEqual(await self.response(stream, None), 'stop')


if __name__ == '__main__':
    unittest.main()
