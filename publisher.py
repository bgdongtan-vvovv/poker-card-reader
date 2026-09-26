"""Publishes recognized table state to Firebase Realtime Database for the online page.

Writes authenticate with a service-account key kept outside the repo
(~/.card-reader/service-account.json); the database rules allow public reads of
/table only and deny all client writes, so only this program can update the page.
"""
import os
import queue
import threading
import time
from collections import OrderedDict

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

DATABASE_URL = "https://card-reader-vvovv-default-rtdb.asia-southeast1.firebasedatabase.app"
KEY_PATH = os.path.join(os.path.expanduser("~"), ".card-reader", "service-account.json")
SCOPES = [
    "https://www.googleapis.com/auth/firebase.database",
    "https://www.googleapis.com/auth/userinfo.email",
]
HISTORY_LIMIT = 50
HEARTBEAT_SEC = 5


class Publisher:
    def __init__(self, on_error=None):
        self.on_error = on_error
        self._credentials = service_account.Credentials.from_service_account_file(KEY_PATH, scopes=SCOPES)
        self._session = requests.Session()
        self._history = OrderedDict()
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def is_configured():
        return os.path.exists(KEY_PATH)

    def publish(self, table, source):
        """Queue a state change; it becomes the current state and a history entry."""
        self._queue.put(("state", table, source, int(time.time() * 1000)))

    def close(self):
        self._stop.set()

    def _token(self):
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return self._credentials.token

    def _request(self, method, path, body=None):
        response = self._session.request(
            method,
            f"{DATABASE_URL}/{path}.json",
            json=body,
            headers={"Authorization": f"Bearer {self._token()}"},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def _run(self):
        try:
            existing = self._request("GET", "table/history") or {}
            for key in sorted(existing)[-HISTORY_LIMIT:]:
                self._history[key] = existing[key]
        except Exception as exc:  # noqa: BLE001 — network/auth errors surface in the GUI log
            self._report(exc)

        last_beat = 0.0
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=1)
            except queue.Empty:
                item = None
            try:
                if item is not None:
                    self._write_state(*item[1:])
                if time.time() - last_beat >= HEARTBEAT_SEC:
                    self._request("PUT", "table/heartbeat", int(time.time() * 1000))
                    last_beat = time.time()
            except Exception as exc:  # noqa: BLE001
                self._report(exc)

    def _write_state(self, table, source, timestamp):
        entry = {**table, "at": timestamp}
        self._history[str(timestamp)] = entry
        while len(self._history) > HISTORY_LIMIT:
            self._history.popitem(last=False)
        self._request("PUT", "table", {
            "state": {**table, "updatedAt": timestamp, "source": source},
            "history": dict(self._history),
            "heartbeat": timestamp,
        })

    def _report(self, exc):
        if self.on_error:
            self.on_error(f"온라인 발행 오류: {exc}")
