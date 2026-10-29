# Dispatch control progress

The current specialized transport keeps one TX control warp alive as the
channel's guardian until both TX and RX have finished. Payload workers can
retire after TX-finished; RX control and its workers use the shared abort
verdict. This removes duplicate full peer scans from the RX main loop while
retaining a resident detector when TX has no outgoing tokens.

## Health and terminal state

`AdvancedHealthCheck` checks the local GPU timer each bounded iteration. Full
mapped abort scans are due at `min(1 ms, timeout_ns / 8)` after the preceding
scan completed. All 32 lanes read independent flags and join with a warp vote.
The due decision is broadcast before any collective; no warp collective is
called only by lane 0. There is no GPU read-modify-write on mapped host memory.

A failed check sets the device error and shared abort. The TX leader remains
the single mapped channel-abort writer. Its unified exit re-acquires abort
after observing RX-finished, covering a tail RX failure that raced the initial
loop check. Terminal mapped flags are never cleared, including Graph replay.
Only device-local per-dispatch diagnostics and epoch-ready state are reset.

Prefix preparation has an independent cooperative poll/timeout path, because
the homogeneous mode prepares prefixes before its transport kernel runs.
The producer still captures every source's rank/expert/channel counts, fences
all writers and joins before ACKing the mailbox. Prefix waiters read GPU-local
READY/errors; no cross-CTA scheduling dependency was added.

## Tail credit

`AdvancedTxDrain` starts at `max(0, N - Q)`. Earlier payload generations were
already acquired before slot reuse. Once all N source tokens are published,
the remaining final generations cannot be reused until transport completion.
Each step acquires at most one pending endpoint ACK, clears only a matching
epoch/token bit, and advances the ordinal only after its mask is empty.
Successful confirmations are retained in lane-0 registers. A missing signal
sets the next poll to completion time + 1 us; guardian checks remain active
during that wait.

After the payload tail, all NUMA relay domains must also report drained for
this epoch, including domains with no payload. Source payload ACK does not
replace relay-drained: the latter also covers replica consumers and endpoint
count-mailbox consumption. Descriptor credit, host payload ACK and local tile
FREE remain separate lifecycle events.

## Previous count mailbox

`advanced_begin_epoch` clears diagnostics, parallel-checks terminal flags,
advances epoch and uses a complete-CTA barrier to publish its entry verdict.
Only then do threads outside warp 0 exit. The first warp acquires each remote
GPU's preceding-epoch counts-consumed value, remembering successful lanes.
The stream's kernel boundary completes this guard before any later kernel
writes channel, endpoint or total counts. This replaces repeated guards in
every channel CTA and `publish_counts`.

The previous transport's relay-drained check protects old CPU endpoint-count
readers. The new guard protects old GPU rank/expert/channel-count readers.
The count publication still fences each writer, joins the CTA and releases
counts READY only afterward. No CPU readback or synchronization is involved.

## Testing and instrumentation

`tests/test_control_progress.py` checks bounded failure/normal progress using
fresh buffers, CPU references and explicit completion diagnostics outside the
production path. Private native test hooks can inject an abort from its legal
TX owner or suppress one receiver-owned counts ACK; neither hook lets the CPU
write GPU-owned mapped flags. Defaults disable both hooks. Other tests cover
cached/Graph replay, small rings, FP8, relay, consumer lifetime and full T8192
outputs.

In an instrumented build, TX role fields 10–12 describe cooperative health
scans; RX main-loop values are zero. Fields 13–17 describe bounded tail steps.
TX role field 24 aliases the otherwise RX-only finish timestamp to record a
guardian's abort observation. The dedicated fault test proves
`TX-finished < abort-observed < TX-exit` on the same GPU. Ordinary profiling
analysis still accepts only successful pipelined dispatches. See
[instrumentation](profiling-instrumentation.md) and
[measured results](../../RESULTS_CONTROL_PROGRESS.md).
