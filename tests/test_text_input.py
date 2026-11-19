"""Check that text-only models never receive or read image attachments."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core import request
from game_theory import llm_player


class TextInputTests(unittest.TestCase):
    def test_text_mode_calls_api_without_opening_an_image(self):
        client = Mock()
        answer = '{"ranking":["B","A"]}'
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer), logprobs=None)])
        with patch.dict(os.environ, {"LLM_INPUT_MODE": "text"}), \
             patch.object(request, "_client", return_value=client), \
             patch.object(request, "_backend", return_value="ollama"), \
             patch.object(request, "_model", return_value="llama3.2:3b"), \
             patch.object(request, "encode_image") as encode:
            result, _ = request.send_image_to_model_openai_logprobs(
                "/nonexistent/grid.png", "unchanged grid and distances", temperature=0)
        encode.assert_not_called()
        self.assertEqual(result, answer)
        sent = client.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["messages"], [{"role": "user", "content": "unchanged grid and distances"}])
        self.assertEqual(sent["model"], "llama3.2:3b")

    def test_image_mode_keeps_text_and_image(self):
        with patch.dict(os.environ, {"LLM_INPUT_MODE": "image"}), \
             patch.object(request, "encode_image", return_value="IMAGE"):
            content = request.ranking_content("grid.png", "same prompt")
        self.assertEqual(content, [{"type": "text", "text": "same prompt"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE"}}])

    def test_invalid_mode_cannot_silently_drop_images(self):
        with patch.dict(os.environ, {"LLM_INPUT_MODE": "typo"}):
            with self.assertRaises(ValueError):
                request.ranking_content("grid.png", "prompt")

    def test_ranking_output_is_bounded_when_configured(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ranking":["A"]}'), logprobs=None)])
        with patch.dict(os.environ, {"LLM_INPUT_MODE": "text", "LLM_MAX_OUTPUT_TOKENS": "2048"}), \
             patch.object(request, "_client", return_value=client), \
             patch.object(request, "_backend", return_value="ollama"), \
             patch.object(request, "_model", return_value="llama3.2:3b"):
            request.send_image_to_model_openai_logprobs("missing.png", "prompt")
        self.assertEqual(client.chat.completions.create.call_args.kwargs["max_tokens"], 2048)

    def test_native_thinking_control_retains_the_image(self):
        client = Mock()
        with patch.dict(os.environ, {"LLM_INPUT_MODE": "image", "LLM_THINK": "false", "LLM_MAX_OUTPUT_TOKENS": "2048"}), \
             patch.object(request, "_client", return_value=client), \
             patch.object(request, "_backend", return_value="ollama"), \
             patch.object(request, "_model", return_value="gemma4:12b"), \
             patch.object(request, "encode_image", return_value="IMAGE"), \
             patch.object(request.ollama, "chat", return_value={"message": {"content": '{"ranking":["A"]}'}}) as chat:
            request.send_image_to_model_openai_logprobs("grid.png", "same prompt", temperature=0)
        args = chat.call_args.kwargs
        self.assertEqual(args["messages"], [{"role": "user", "content": "same prompt", "images": ["IMAGE"]}])
        self.assertFalse(args["think"])
        self.assertEqual(args["options"]["num_predict"], 2048)
        client.chat.completions.create.assert_not_called()

    def test_conflict_output_uses_the_same_limit_and_thinking_control(self):
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama", "LLM_THINK": "false", "LLM_MAX_OUTPUT_TOKENS": "2048"}), \
             patch.object(request.ollama, "chat", return_value={"message": {"content": "answer"}}) as chat:
            self.assertEqual(llm_player.call_model("same conflict prompt"), "answer")
        self.assertFalse(chat.call_args.kwargs["think"])
        self.assertEqual(chat.call_args.kwargs["options"]["num_predict"], 2048)

    def test_nonpositive_limit_is_rejected_before_model_call(self):
        with patch.dict(os.environ, {"LLM_BACKEND": "ollama", "LLM_MAX_OUTPUT_TOKENS": "0"}), \
             patch.object(request.ollama, "chat") as chat:
            with self.assertRaises(ValueError):
                llm_player.call_model("prompt")
        chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
