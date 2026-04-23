# 实时 WebSocket 准实时升级 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将实时 WebSocket 从“整段历史重转写”升级为“队列解耦 + 当前语音段增量识别 + 静音提交”的准实时模式。

**Architecture:** 保留现有 WebSocket 路径，升级协议和 CLI。服务端为每个会话维护独立队列、当前语音段缓存和已确认文本；通过周期解码与静音提交避免整段历史反复重跑。测试先行，确保协议与行为可验证。

**Tech Stack:** FastAPI, websockets, numpy, unittest, existing OpenAI-compatible vLLM client

---

### Task 1: 写协议级失败测试

**Files:**
- Modify: `tests/test_server_api.py`
- Test: `tests/test_server_api.py`

- [ ] **Step 1: 写新的 WebSocket 协议测试**

```python
def test_realtime_websocket_protocol_upgrade(self):
    with self.client.websocket_connect("/ws/v1/realtime/transcribe") as ws:
        ready = ws.receive_json()
        self.assertEqual(ready["event"], "ready")
        self.assertIn("protocol_version", ready)
        self.assertIn("realtime_alignment_supported", ready)

        ws.send_json({"event": "start", "context": "meeting"})
        started = ws.receive_json()
        self.assertEqual(started["event"], "started")
        self.assertIn("session", started)
```

- [ ] **Step 2: 运行单测确认失败**

Run: `python -m unittest tests.test_server_api.ServerApiTests.test_realtime_websocket_protocol_upgrade`

Expected: FAIL，提示返回字段与新协议不一致

- [ ] **Step 3: 写实时拒绝对齐参数测试**

```python
def test_realtime_websocket_rejects_forced_aligner_field(self):
    with self.client.websocket_connect("/ws/v1/realtime/transcribe") as ws:
        ws.receive_json()
        ws.send_json({"event": "start", "use_forced_aligner": True})
        error = ws.receive_json()
        self.assertEqual(error["event"], "error")
```

- [ ] **Step 4: 运行该测试确认失败**

Run: `python -m unittest tests.test_server_api.ServerApiTests.test_realtime_websocket_rejects_forced_aligner_field`

Expected: FAIL，提示当前服务没有显式拒绝

### Task 2: 写实时会话状态失败测试

**Files:**
- Modify: `tests/test_offline_and_streaming.py`
- Test: `tests/test_offline_and_streaming.py`

- [ ] **Step 1: 写静音提交测试**

```python
def test_streaming_session_emits_segment_final_after_silence(self):
    streaming = StreamingTranscriber(...)
    session = StreamingSession(context="demo")
    speech = np.ones(16000, dtype=np.float32) * 0.02
    silence = np.zeros(16000, dtype=np.float32)

    streaming.push_audio(session, speech)
    event = streaming.push_audio(session, silence)

    self.assertIn(event["event"], {"partial", "segment_final"})
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests.test_offline_and_streaming.OfflineAndStreamingTests.test_streaming_session_emits_segment_final_after_silence`

Expected: FAIL，说明当前状态机没有静音提交能力

- [ ] **Step 3: 写最终文本聚合测试**

```python
def test_streaming_finish_returns_committed_and_live_text(self):
    final = streaming.finish(session)
    self.assertEqual(final["event"], "final")
    self.assertIn("text", final)
```

- [ ] **Step 4: 运行测试确认失败**

Run: `python -m unittest tests.test_offline_and_streaming.OfflineAndStreamingTests.test_streaming_finish_returns_committed_and_live_text`

Expected: FAIL，说明当前状态结构不满足新协议

### Task 3: 实现新的 StreamingTranscriber

**Files:**
- Modify: `qwen3_asr_toolkit/streaming_transcriber.py`
- Test: `tests/test_offline_and_streaming.py`

- [ ] **Step 1: 扩展会话状态**

```python
@dataclass
class StreamingSession:
    context: str = ""
    session_id: Optional[str] = None
    segment_audio: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    committed_text: str = ""
    live_text: str = ""
    language: str = ""
    silence_samples: int = 0
    received_samples: int = 0
    segment_index: int = 0
    decode_interval_samples: int = 9600
    min_chunk_samples: int = 3200
    finalize_silence_samples: int = 9600
    max_segment_samples: int = 480000
    pending_decode_samples: int = 0
    closed: bool = False
```

- [ ] **Step 2: 实现最小 `push_audio()`**

```python
def push_audio(self, session, pcm16k):
    x = self._normalize_audio(pcm16k)
    session.segment_audio = np.concatenate([session.segment_audio, x], axis=0)
    session.received_samples += len(x)
    session.pending_decode_samples += len(x)
    self._update_silence(session, x)
    return self._maybe_emit(session)
```

- [ ] **Step 3: 实现静音提交和强制提交**

```python
if self._should_finalize_segment(session):
    self._decode_segment(session)
    self._commit_segment(session)
    return self._build_segment_final(session)
```

- [ ] **Step 4: 跑流式相关单测**

Run: `python -m unittest tests.test_offline_and_streaming -v`

Expected: PASS

### Task 4: 升级 WebSocket 服务端

**Files:**
- Modify: `qwen3_asr_toolkit/server.py`
- Test: `tests/test_server_api.py`

- [ ] **Step 1: 改造 `ready` 和 `start` 协议**

```python
await websocket.send_json({
    "event": "ready",
    "protocol_version": "2.0",
    "session_id": session.session_id,
    "sample_rate": WAV_SAMPLE_RATE,
    "audio_format": "float32le_pcm_mono_16k",
    "realtime_alignment_supported": False,
})
```

- [ ] **Step 2: 拆分 `receiver` 与 `worker`**

```python
audio_queue = asyncio.Queue()
receiver_task = asyncio.create_task(_receiver_loop(websocket, audio_queue))
worker_task = asyncio.create_task(_worker_loop(websocket, audio_queue, session))
```

- [ ] **Step 3: 显式拒绝对齐字段**

```python
if payload.get("use_forced_aligner"):
    await websocket.send_json({"event": "error", "message": "Realtime transcription does not support forced aligner."})
    continue
```

- [ ] **Step 4: 跑服务端测试**

Run: `python -m unittest tests.test_server_api -v`

Expected: PASS

### Task 5: 升级实时 CLI

**Files:**
- Modify: `qwen3_asr_toolkit/realtime_cli.py`
- Test: `tests/test_server_api.py`

- [ ] **Step 1: 发送新的 `start` 参数**

```python
await ws.send(json.dumps({
    "event": "start",
    "context": args.context,
    "decode_interval_ms": args.decode_interval_ms,
    "finalize_silence_ms": args.finalize_silence_ms,
}))
```

- [ ] **Step 2: 兼容 `segment_final` 输出**

```python
if message.get("event") == "segment_final":
    print(f"[segment_final] {message.get('text', '')}")
```

- [ ] **Step 3: 为 CLI 增加新参数**

```python
parser.add_argument("--decode-interval-ms", type=int, default=600)
parser.add_argument("--finalize-silence-ms", type=int, default=600)
```

- [ ] **Step 4: 跑帮助命令检查**

Run: `python -m qwen3_asr_toolkit.realtime_cli --help`

Expected: 输出包含新参数

### Task 6: 更新文档与验收

**Files:**
- Modify: `README.md`
- Modify: `doc/USAGE.md`
- Modify: `doc/TECH_IMPROVEMENTS.md`

- [ ] **Step 1: 更新实时协议说明**

```markdown
- `ready`：服务端返回会话能力
- `start`：客户端初始化实时参数
- `partial`：当前未完成段的增量结果
- `segment_final`：一次语音段确认完成
- `final`：整次会话最终结果
```

- [ ] **Step 2: 说明实时不支持对齐**

```markdown
实时 WebSocket 转写不支持 `use_forced_aligner`，若传入会返回错误。
```

- [ ] **Step 3: 跑完整验证**

Run: `python -m unittest discover -s tests -p 'test_*.py'`

Expected: 全部通过
