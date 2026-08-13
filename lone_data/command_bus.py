"""Non-blocking robot command dispatch.

Every link call blocks for as long as the board takes to answer -- one raw-REPL
round trip over serial, or one TCP round trip over WiFi. The old recorder made
those calls on the Tk main thread, which is the same thread the record tick
runs on, so a slow board directly stalled frame capture. That capped recording
at ~14.8 Hz median with individual gaps up to 0.86s.

Commands are queued here and drained by one worker thread instead. Pending
commands coalesce per channel key: if the link is slower than the operator,
only the newest command for a channel is still worth sending, which is also
where the robot would have ended up. That bounds the backlog to one command per
channel no matter how far behind the link falls.
"""

import threading
import time
from collections import OrderedDict

_LATENCY_WINDOW = 512

# A queued command older than this was overtaken by the operator's next input
# long ago. Dispatching it drives the arm to a position they've already moved
# on from, so drop it -- coalescing bounds the backlog to one per channel but
# says nothing about how stale that one is.
#
# Only ever applied to commands the caller marks droppable (see submit()).
_MAX_COMMAND_AGE = 0.25


class CommandBus:
    def __init__(self, on_error=None, on_link_dead=None, fatal_errors=()):
        """fatal_errors: exception types that mean the link itself is finished,
        not that one command failed. Passed in rather than imported so this
        package stays independent of any particular link implementation."""
        self._fatal_errors = tuple(fatal_errors)
        self._link = None
        self._pending = OrderedDict()  # key -> (method, args, enqueued_monotonic, droppable)
        self._cond = threading.Condition()
        self._running = False
        self._thread = None
        self._on_error = on_error
        self._on_link_dead = on_link_dead
        self._latencies = []
        self._coalesced = 0
        self._sent = 0
        self._dropped = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_link(self, link):
        """Swaps the active link; drops anything queued for the old one."""
        with self._cond:
            self._link = link
            self._pending.clear()

    def submit(self, key, method, *args, droppable=False):
        """Queues link.<method>(*args), replacing any pending command for `key`.

        droppable marks a command that a later one will supersede anyway -- a
        held-joint speed that a heartbeat re-sends every tick. Only those may be
        discarded for being stale. It defaults to False because the dangerous
        commands are exactly the un-refreshed ones: drop a stop and the joint
        keeps driving until the board's deadman timer fires seconds later.
        """
        with self._cond:
            if key in self._pending:
                self._coalesced += 1
            self._pending[key] = (method, args, time.monotonic(), droppable)
            self._pending.move_to_end(key)
            self._cond.notify()

    def cancel_pending(self):
        """Drops everything queued. Used before an emergency stop so a stale
        speed command can't land after it."""
        with self._cond:
            self._pending.clear()

    def send_now(self, method, *args):
        """Dispatches on the caller's thread, bypassing the queue. For stops and
        teardown, where blocking is fine and actually stopping matters more."""
        link = self._link
        if link is None:
            return
        try:
            getattr(link, method)(*args)
        except Exception as e:
            self._dispatch_error(f"{method}{args}: {e}", e)

    def _run(self):
        while True:
            with self._cond:
                while self._running and not self._pending:
                    self._cond.wait(0.1)
                if not self._pending:
                    if not self._running:
                        return
                    continue
                _key, (method, args, queued_at, droppable) = self._pending.popitem(last=False)
                link = self._link

            if link is None:
                continue
            if droppable and time.monotonic() - queued_at > _MAX_COMMAND_AGE:
                with self._cond:
                    self._dropped += 1
                continue
            try:
                getattr(link, method)(*args)
            except Exception as e:
                self._dispatch_error(f"{method}{args}: {e}", e)
                continue

            with self._cond:
                self._sent += 1
                self._latencies.append(time.monotonic() - queued_at)
                if len(self._latencies) > _LATENCY_WINDOW:
                    del self._latencies[: _LATENCY_WINDOW // 2]

    def _report(self, message):
        if self._on_error is not None:
            self._on_error(message)

    def _dispatch_error(self, message, exc):
        """A fatal error can't be recovered by retrying -- everything queued for
        this link would fail the same way, so drop the backlog and hand the
        problem to the owner rather than logging one failure per command."""
        if not isinstance(exc, self._fatal_errors):
            self._report(message)
            return
        with self._cond:
            self._pending.clear()
        if self._on_link_dead is not None:
            self._on_link_dead(message)
        else:
            self._report(message)

    def stats(self):
        """Dispatch latency in seconds (enqueue -> board acknowledged)."""
        with self._cond:
            lat = list(self._latencies)
            sent, coalesced = self._sent, self._coalesced
            dropped, backlog = self._dropped, len(self._pending)
        base = {"sent": sent, "coalesced": coalesced, "dropped": dropped, "backlog": backlog}
        if not lat:
            return {**base, "latency_mean": 0.0, "latency_max": 0.0}
        return {**base, "latency_mean": sum(lat) / len(lat), "latency_max": max(lat)}

    def close(self, timeout=1.0):
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
