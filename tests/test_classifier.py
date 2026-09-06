import unittest
import threading

from agent.classifier import RequestType, classify_request
from agent.controller import AIController
from llm.provider import ProviderEvent


class RequestClassifierTests(unittest.TestCase):
    def assert_type(self, request: str, expected: RequestType) -> None:
        result = classify_request(request)
        self.assertEqual(result.type, expected, request)
        self.assertEqual(result.requires_tools, expected is RequestType.SYSTEM_TASK)
        self.assertEqual(result.as_dict()["type"], expected.value)

    def test_conversation_examples(self) -> None:
        for request in (
            "hi",
            "hello",
            "thanks",
            "what is Linux?",
            "explain DNS",
            "what is Wi-Fi?",
            "what is high CPU usage?",
            "How do I install Chrome?",
            "Can you tell me how to install Chrome?",
            "How do I troubleshoot my Wi-Fi?",
            "Explain this command: `ip addr`",
        ):
            self.assert_type(request, RequestType.CONVERSATION)

    def test_system_task_examples(self) -> None:
        for request in (
            "install blender",
            "install Google Chrome",
            "download Google Chrome",
            "download VLC",
            "install Firefox",
            "get Chrome",
            "check Chrome",
            "troubleshoot Chrome",
            "check my Wi-Fi",
            "why is my internet not working?",
            "check CPU usage",
            "list files in my home directory",
            "create a folder",
            "remove this package",
            "restart Bluetooth service",
            "configure my Wi-Fi",
            "configure software for me",
            "what is using my CPU?",
            "what is my CPU usage?",
            "run `ip addr` and check my network",
        ):
            self.assert_type(request, RequestType.SYSTEM_TASK)

    def test_controller_bypasses_task_runtime_for_conversation(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.chat_calls = 0

            def stream_chat(self, messages, cancel_event):
                self.chat_calls += 1
                yield ProviderEvent.text_chunk("hello")
                yield ProviderEvent.done()

        class MustNotRun:
            def matches(self, _request):
                return True

            def stream(self, *_args, **_kwargs):
                raise AssertionError("conversation entered the task workflow")

        provider = Provider()
        controller = AIController(
            provider,
            troubleshooter=MustNotRun(),
            software_manager=MustNotRun(),
            runtime=MustNotRun(),
        )
        events = list(controller.stream_response("what is Linux?", threading.Event()))
        self.assertEqual([event.kind for event in events], ["text", "done"])
        self.assertEqual(provider.chat_calls, 1)


if __name__ == "__main__":
    unittest.main()
