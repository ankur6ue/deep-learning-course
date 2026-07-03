# v7: Async GPU/CPU Control Plane

`v7` keeps FlashAttention, the v5 local kernels, and the v6 compile boundary, then reduces decode-side CPU synchronization.

## What Changed From v6

- `DecodeGpuState` stores the latest sampled token for each active request on GPU.
- Stable integer request slots map Python request IDs to GPU rows.
- `prepare_decode_inputs()` fills decode `input_ids`, positions, lengths, slot mappings, and int32 block tables on GPU.
- `update_decode_state()` writes sampled tokens back into the GPU table.
- `AsyncTokenCopy` copies sampled tokens to CPU without blocking the next decode step.
- `attention_backends.py` still contains only `FlashAttentionPagedBackend`.

## Async Decode State

`v7` splits decode state into CPU scheduling state and GPU feedback state. The
goal is simple: after the GPU samples token `T`, the next decode step should be
able to feed `T` back into the model without waiting for `T.item()` on CPU.

### CPU-Owned Structures

```text
RequestState
  request_id
      Stable Python id used by the scheduler.

  block_ids
      CPU list mapping this request's logical KV pages to physical KV pages.
      Example: logical block 0 -> physical block 7.

  generated_ids
      CPU-visible output tokens that have already been materialized.

  deferred_generated_token_copies
      List of pending `(AsyncTokenCopy, row_index)` pairs. These are sampled
      tokens whose D2H copy has started but whose Python int has not been read.

  generated_tokens_in_cache
      Count of generated tokens whose K/V has actually been written by decode.

  next_input_token_id
      In async mode this is only a non-None readiness marker. The real next
      token id lives in `DecodeGpuState.last_token_ids` on GPU.

DecodeGpuState.request_id_to_slot
  CPU dict: request id -> stable row in GPU decode-state tensors.
  Example: {"req-a": 0, "req-b": 1}

DecodeGpuState.free_slots
  CPU list of reusable slot ids. Slots are stable while a request is active and
  returned here when the request finishes.

DecodeGpuState.slot_cpu
  Pinned CPU staging tensor for the current decode batch's slot ids.
  Example for batch [req-b, req-a]: [1, 0]

AsyncTokenCopy.cpu_tokens
  CPU tensor that receives sampled token ids from an async D2H copy.

AsyncTokenCopy.event
  CUDA event recorded after the D2H copy is queued. `tolist()` synchronizes on
  this event only when the CPU finally needs the token values.
```

### GPU-Owned Structures

```text
DecodeGpuState.last_token_ids
  GPU tensor indexed by stable request slot.
  Example:
      last_token_ids[0] = latest token for req-a
      last_token_ids[1] = latest token for req-b

DecodeGpuState.slot_workspace
  GPU copy of `slot_cpu` for the current decode batch. Decode prep kernels use
  it to gather from `last_token_ids`; decode postprocess uses the same staged
  slots to scatter newly sampled tokens back into `last_token_ids`.

DecodeEagerWorkspace.input_ids
  GPU model input for the current decode step. Filled from `last_token_ids`,
  not from CPU token values.

DecodeEagerWorkspace.positions
  GPU decode positions. For one-token decode, each row is the request's current
  cached sequence length.

DecodeEagerWorkspace.cached_seq_lens
  GPU copy of the CPU-staged cached sequence lengths.

DecodeEagerWorkspace.block_tables
  GPU copy of the CPU-staged request block tables.

DecodeEagerWorkspace.slot_mapping
  GPU tensor mapping each decode row to the flattened physical KV slot where
  the new token's K/V should be written.

PagedKVCache.k_layers / v_layers
  GPU paged KV cache. Decode writes the K/V for the token that entered the
  model on this step.

next_tokens
  GPU result of `sample_greedy(logits)`, shaped `[batch]`.
```

### Concrete Example

Assume two active requests:

```text
request_id_to_slot = {"req-a": 0, "req-b": 1}

last_token_ids[0] = 50   # req-a's next decode input
last_token_ids[1] = 60   # req-b's next decode input

decode batch order = [req-b, req-a]
```

The current batch uses slots `[1, 0]`, because row 0 is `req-b` and row 1 is
`req-a`.

### Transfer Timeline

These diagrams focus only on generated-token handling. They intentionally omit
attention metadata such as block tables, sequence lengths, and KV slot mappings.
`H2D` means host-to-device. `D2H` means device-to-host.

#### v6: Blocking Token Readback

In `v6`, the CPU owns the actual next token id. After each decode step, it must
read `next_tokens[row]` back to a Python int before it can set
`req.next_input_token_id` and requeue the request.

```text
time --------------------------------------------------------------->

CPU thread:
  build batch | enqueue H2D | enqueue forward | enqueue argmax |
                                                              BLOCK on .item()
                                                              update RequestState
                                                              prepare next batch

default CUDA stream:
               | copy input_ids | ===== transformer forward ===== | argmax | D2H token |

GPU data:
                 input_ids=[[60],[50]]                            next_tokens=[61,51]
```

The visible wait is the `.item()` path:

```python
sampled = int(next_tokens[idx].item())
```

That read forces the CPU to wait until the GPU has produced `next_tokens`.

State changes in `v6`:

```text
0. Before decode
   CPU:
     req-b.next_input_token_id = 60
     req-a.next_input_token_id = 50
     req-b.generated_ids = []
     req-a.generated_ids = []
   GPU:
     no GPU-resident per-request next-token table

1. CPU builds and enqueues `input_ids`
   CPU:
     request state unchanged
   GPU after the H2D copy runs:
     input_ids = [[60], [50]]

2. CPU enqueues model forward
   CPU:
     immediately continues to enqueue later work
   GPU when the queued kernels execute:
     transformer kernels run for a relatively long time
     logits is produced

3. CPU enqueues `sample_greedy`
   CPU:
     still does not know the sampled token values
   GPU after argmax executes:
     next_tokens = [61, 51]

4. CPU executes `int(next_tokens[0].item())`
   CPU:
     blocks until model forward, argmax, and the scalar D2H read finish
     sampled token for req-b becomes Python int 61
   GPU:
     next_tokens remains [61, 51]

5. CPU executes `int(next_tokens[1].item())`
   CPU:
     sampled token for req-a becomes Python int 51
   GPU:
     no new GPU state needed for generated-token feedback

6. CPU updates request state
   CPU:
     req-b.generated_ids = [61]
     req-a.generated_ids = [51]
     req-b.next_input_token_id = 61
     req-a.next_input_token_id = 51
   GPU:
     unchanged

7. CPU can finally prepare the next decode batch
   CPU:
     next input tokens are available as Python ints
   GPU:
     waits for the CPU to enqueue the next batch
```

#### v7: Async GPU Feedback And Deferred CPU Readback

In `v7`, the CPU still chooses which requests are in the next decode batch, but
the actual next token id stays on GPU in `last_token_ids`.

The timeline below separates CPU enqueue work from GPU execution. The bar
widths are illustrative: transformer forward is long, while slot copies,
prepare, argmax, and decode-state update are much shorter. Values such as
`input_ids`, `last_token_ids`, and `next_tokens` live in GPU data, not "inside"
a stream.

```text
time ------------------------------------------------------------------------------->

CPU thread:
  build batch | slot_cpu=[1,0] | enqueue slots | enqueue prepare | enqueue forward |
  enqueue argmax | enqueue async D2H | enqueue update | CPU bookkeeping |
  scheduler picks/prepares next batch | enqueue next slots | enqueue next prepare |
  ... later resolve AsyncTokenCopy for output/EOS ...

default CUDA stream:
                | copy slots | prepare | ===== transformer forward ===== | argmax | update last_token_ids |
                                                                                                      | next prepare |

copy_stream:
                                                  wait for default stream through argmax | D2H next_tokens |

GPU data:
  last_token_ids[1]=60, last_token_ids[0]=50
                slot_workspace=[1,0]
                             input_ids=[[60],[50]]
                                                                      next_tokens=[61,51]
                                                                                 last_token_ids[1]=61
                                                                                 last_token_ids[0]=51
                                                                                                      next input_ids
```

The important difference is not that the next GPU work has no dependency. It
does have a dependency:

```text
sample_greedy -> update_decode_state -> next prepare_decode_inputs
```

The difference is that this dependency stays on GPU streams. The CPU no longer
has to materialize `next_tokens[row]` before it can schedule the next decode
step.

State changes in `v7`:

```text
0. Before decode
   CPU:
     request_id_to_slot = {"req-a": 0, "req-b": 1}
     req-b.next_input_token_id = pad_token_id  # readiness marker
     req-a.next_input_token_id = pad_token_id  # readiness marker
     req-b.deferred_generated_token_copies may contain older copies
     req-a.deferred_generated_token_copies may contain older copies
   GPU:
     last_token_ids[1] = 60  # req-b's real next input token
     last_token_ids[0] = 50  # req-a's real next input token

1. CPU picks decode batch `[req-b, req-a]`
   CPU:
     slot_cpu = [1, 0]
   GPU:
     unchanged

2. CPU enqueues H2D slot copy
   CPU:
     immediately continues after enqueue
   GPU after the copy executes on the default stream:
     slot_workspace = [1, 0]

3. CPU enqueues `prepare_decode_inputs`
   CPU:
     still does not read token values
   GPU when the prepare kernel executes:
     reads slot_workspace = [1, 0]
     reads last_token_ids[1] = 60
     reads last_token_ids[0] = 50
     writes input_ids = [[60], [50]]

4. CPU enqueues model forward
   CPU:
     returns after launching kernels
   GPU when queued kernels execute:
     transformer forward runs for a relatively long time
     logits is produced

5. CPU enqueues `sample_greedy`
   CPU:
     still does not know the sampled token values
   GPU after argmax executes:
     next_tokens = [61, 51]

6. CPU calls `async_copy_tokens(next_tokens, 2)`
   CPU:
     creates an AsyncTokenCopy object
     receives it immediately
     does not call `.item()` or `tolist()`
   GPU/copy stream:
     waits until the default stream has produced next_tokens
     then copies next_tokens to AsyncTokenCopy.cpu_tokens
     records an event after the D2H copy in the copy stream

7. CPU enqueues `update_decode_state`
   CPU:
     still does not know token values as Python ints
   GPU when the update kernel executes on the default stream:
     reads slot_workspace = [1, 0]
     reads next_tokens = [61, 51]
     writes last_token_ids[1] = 61
     writes last_token_ids[0] = 51

8. CPU updates request bookkeeping
   CPU:
     req-b.deferred_generated_token_copies += [(copy, 0)]
     req-a.deferred_generated_token_copies += [(copy, 1)]
     req-b.generated_tokens_in_cache += 1
     req-a.generated_tokens_in_cache += 1
     req-b.next_input_token_id = pad_token_id  # marker only
     req-a.next_input_token_id = pad_token_id  # marker only
   GPU:
     last_token_ids is the source of truth for the next input token

9. CPU schedules the next decode step
   CPU:
     can choose the next batch and enqueue the next slot copy/prepare
     without materializing [61, 51] on CPU
   GPU:
     next prepare_decode_inputs is queued after update_decode_state
     if last_token_ids is not ready, it waits by CUDA stream ordering
     the CPU does not block on token values

10. CPU later resolves deferred output/EOS bookkeeping
    CPU:
      token = int(copy.tolist()[row])
      `tolist()` synchronizes on the copy event only if needed
      generated_ids is appended
      EOS is checked
    GPU:
      no new model input depends on this CPU materialization
```

### Why `slot_workspace` Exists

The CPU scheduler knows request objects and request ids. GPU kernels cannot
index Python dictionaries, so every decode batch needs a compact tensor saying:

```text
batch row -> stable request slot
```

For batch `[req-b, req-a]`, that tensor is:

```text
slot_workspace = [1, 0]
```

The same staged tensor is used twice in eager decode:

1. Decode prepare gathers the current input tokens:
   `input_ids[row] = last_token_ids[slot_workspace[row]]`
2. Decode postprocess scatters newly sampled tokens:
   `last_token_ids[slot_workspace[row]] = next_tokens[row]`

CUDA graph buckets in later versions use their own fixed-address `req_slots`
tensor instead of `DecodeGpuState.slot_workspace`, because captured graphs need
stable tensor addresses owned by the graph bucket.

### Why `AsyncTokenCopy` Exists

The GPU needs sampled tokens immediately for the next decode step. The CPU only
needs them later for user-visible output and EOS checks. `AsyncTokenCopy` lets
those two timelines diverge:

```text
GPU-critical path:
    next_tokens -> last_token_ids -> next decode input

CPU bookkeeping path:
    next_tokens -> async D2H copy -> generated_ids / EOS check
```

If the CPU calls `tolist()` immediately, it pays the synchronization cost
immediately. If it waits until a later scheduler step, the D2H copy may already
be complete, and the GPU has not been blocked on CPU token materialization.

## Next Version

`v8` adds CUDA graph replay around optimized decode and all-valid prefill paths.

## Validation

Short real-model sanity run, Llama 3.1 8B, shared six-request workload, `max_new_tokens=8`, prefix cache disabled.

| Version | Wall time (s) | Tok/s | Exact IDs vs previous | Exact IDs vs vLLM | Improvement vs previous | Remaining gap vs vLLM |
|---|---:|---:|---|---|---:|---:|
| v7 | 0.439 | 109.24 | yes | no | -0.1% | +38.2% |

`v7` matched `v6` exactly. On this short workload the async control-plane change is correctness-neutral but not measurably faster; its value tends to show up more when decode has enough steps to hide CPU readback latency.
