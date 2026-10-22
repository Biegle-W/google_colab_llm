import threading
import time
import unittest

from colab_llm_hub import hub as hubmod
from .helpers import KEY, HubCase, entry


class AuthAndModels(HubCase):
    def test_requires_bearer_key(self):
        self.assertEqual(self.request("GET", "/v1/models", key=None)[0], 401)
        self.assertEqual(self.request("GET", "/v1/models", key="wrong")[0], 401)
        status, _, headers = self.request("GET", "/v1/models", key="wrong")
        self.assertIn("Bearer", headers.get("WWW-Authenticate", ""))
        self.assertEqual(self.request("GET", "/v1/models")[0], 200)
        self.assertEqual(self.chat("a", key="wrong")[0], 401)

    def test_models_only_those_that_fit(self):
        _, data, _ = self.request("GET", "/v1/models")
        ids = [m["id"] for m in data["data"]]
        self.assertEqual(ids, ["a", "b"])                   # "big" needs 60 GB; this GPU has 24
        m = data["data"][0]
        self.assertEqual((m["context_length"], m["loaded"], m["downloaded"]), (4096, False, False))
        self.assertEqual(self.chat("big")[0], 404)
        self.assertEqual(self.chat("nope")[0], 404)

    def test_bad_requests(self):
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"messages": []})[0], 400)
        self.assertEqual(self.request("POST", "/v1/embeddings", {"model": "a"})[0], 404)
        self.assertEqual(self.request("GET", "/nothing")[0], 404)


class Loading(HubCase):
    def test_loads_on_demand_then_reuses_engine(self):
        status, data, _ = self.chat("a")
        self.assertEqual((status, data["choices"][0]["message"]["content"]), (200, "reply from a"))
        self.assertEqual(self.hub.active, "a")
        self.chat("a")
        self.assertEqual(len(self.launches), 1)
        _, models, _ = self.request("GET", "/v1/models")
        self.assertTrue(next(m for m in models["data"] if m["id"] == "a")["loaded"])

    def test_switching_stops_the_previous_engine(self):
        self.chat("a")
        first = self.launches[0]
        status, data, _ = self.chat("b")
        self.assertEqual((status, data["choices"][0]["message"]["content"]), (200, "reply from b"))
        self.assertFalse(first.alive)
        self.assertEqual(self.hub.active, "b")

    def test_switch_waits_for_inflight_stream(self):
        type(self).chunk_delay = 0.4
        try:
            result = {}

            def slow_stream():
                result["raw"] = self.stream_raw("a")[1]
            self.chat("a")                      # load a
            t = threading.Thread(target=slow_stream)
            t.start()
            time.sleep(0.5)                     # a's stream is mid-flight
            started = time.time()
            status, data, _ = self.chat("b")    # must wait for the stream to finish, not cut it off
            t.join()
            self.assertEqual(status, 200)
            self.assertIn("a-0", result["raw"])
            self.assertIn("a-2", result["raw"])
            self.assertIn("[DONE]", result["raw"])
            self.assertEqual(self.hub.active, "b")
        finally:
            type(self).chunk_delay = 0.0

    def test_unload(self):
        self.chat("a")
        status, _, _ = self.request("POST", "/v1/hub/unload", {})
        self.assertEqual(status, 200)
        self.assertIsNone(self.hub.active)
        self.assertFalse(self.launches[0].alive)

    def test_crashed_engine_is_reloaded(self):
        self.chat("a")
        self.launches[0].stop(grace=2)
        status, data, _ = self.chat("a")
        self.assertEqual(status, 200)
        self.assertEqual(len(self.launches), 2)


class SlowLoad(HubCase):
    load_delay = 3.0

    def test_nonstreaming_gets_retry_after_then_succeeds(self):
        old = hubmod.NONSTREAM_WAIT
        hubmod.NONSTREAM_WAIT = 1
        try:
            status, data, headers = self.chat("a")
            self.assertEqual(status, 503)
            self.assertEqual(data["error"]["code"], "model_loading")
            self.assertIn("Retry-After", headers)
            for _ in range(30):                 # the load kept going in the background
                status, data, _ = self.chat("a")
                if status == 200:
                    break
                time.sleep(0.5)
            self.assertEqual(status, 200)
            self.assertEqual(len(self.launches), 1)
        finally:
            hubmod.NONSTREAM_WAIT = old

    def test_streaming_gets_keepalives_while_loading(self):
        old = hubmod.KEEPALIVE
        hubmod.KEEPALIVE = 0.3
        try:
            status, raw, _ = self.stream_raw("a")
            self.assertEqual(status, 200)
            self.assertIn(": loading a", raw)
            self.assertIn("a-0", raw)
            self.assertIn("[DONE]", raw)
            self.assertLess(raw.index(": loading a"), raw.index("a-0"))
        finally:
            hubmod.KEEPALIVE = old


class Failures(HubCase):
    fail = {"a"}

    def test_failed_start_is_reported_and_cached_briefly(self):
        status, data, _ = self.chat("a")
        self.assertEqual(status, 503)
        self.assertEqual(data["error"]["code"], "backend_start_failed")
        self.assertIn("could not load the model", data["error"]["message"])
        n = len(self.launches)
        self.assertEqual(self.chat("a")[0], 503)           # within the TTL: no new launch
        self.assertEqual(len(self.launches), n)
        self.assertEqual(self.chat("b")[0], 200)           # other models still work

    def test_streaming_failure_is_an_sse_error_event(self):
        status, raw, _ = self.stream_raw("a")
        self.assertEqual(status, 200)
        self.assertIn('"code": "backend_start_failed"', raw)
        self.assertTrue(raw.rstrip().endswith("[DONE]"))

    def test_retry_after_ttl_relaunches(self):
        self.chat("a")
        type(self).fail = set()
        old = hubmod.ERROR_TTL
        hubmod.ERROR_TTL = 0
        try:
            self.assertEqual(self.chat("a")[0], 200)
        finally:
            hubmod.ERROR_TTL = old
            type(self).fail = {"a"}


class StoreIntegration(HubCase):
    def test_download_then_drive_copy_then_reuse(self):
        self.assertEqual(self.chat("a")[0], 200)
        self.assertEqual(self.downloads, ["org/a"])
        self.store.wait_for_drive_copies()
        self.assertTrue(self.store.state(self.hub.entries["a"])["on_drive"])
        self.assertEqual(self.chat("b")[0], 200)
        self.assertEqual(self.downloads, ["org/a", "org/b"])
        # Switching back reuses the local copy: no second download.
        self.assertEqual(self.chat("a")[0], 200)
        self.assertEqual(self.downloads, ["org/a", "org/b"])

    def test_prefetch_saves_to_drive_without_loading(self):
        status, _, _ = self.request("POST", "/v1/hub/prefetch", {"model": "b"})
        self.assertEqual(status, 202)
        for _ in range(50):
            if self.store.state(self.hub.entries["b"])["on_drive"]:
                break
            time.sleep(0.1)
        self.assertTrue(self.store.state(self.hub.entries["b"])["on_drive"])
        self.assertIsNone(self.hub.active)
        _, models, _ = self.request("GET", "/v1/models")
        self.assertTrue(next(m for m in models["data"] if m["id"] == "b")["downloaded"])
        self.assertEqual(self.request("POST", "/v1/hub/prefetch", {"model": "big"})[0], 404)

    def test_status_endpoint(self):
        self.chat("a")
        _, st, _ = self.request("GET", "/v1/hub/status")
        self.assertEqual((st["active"], st["gpu"], st["inflight"]), ("a", "fake", 0))
        self.assertIn("a", st["downloads"])


if __name__ == "__main__":
    unittest.main()
