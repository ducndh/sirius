#!/usr/bin/env python3
"""
cuGraph worker process for Sirius graph table functions.
Communicates via binary protocol on stdin/stdout.
Returns results via CUDA IPC (zero-copy GPU memory sharing).

Persistent mode: handles multiple requests on the same pipe connection.
After each response + ACK cycle, loops back to read the next request header.
Exits when stdin reaches EOF (parent closes pipe).

Protocol (per request):
  stdin (binary):
    [4 bytes] algo: 0=BFS, 1=PAGERANK, 2=WCC
    [4 bytes] num_edges
    [4 bytes] num_vertices  (max vertex ID + 1)
    [16 bytes] params (algo-specific, see below)
    [num_edges * 4 bytes] src_ids (int32)
    [num_edges * 4 bytes] dst_ids (int32)

  BFS params:    [4 bytes] source_vertex, [4 bytes] depth_limit (-1=unlimited), [8 bytes] padding
  PAGERANK params: [4 bytes float] damping, [4 bytes float] tolerance, [4 bytes] max_iter, [4 bytes] padding
  WCC params:    [16 bytes] unused

  stdout (binary):
    [4 bytes] status: 0=ok, 1=error
    [4 bytes] num_result_rows
    [4 bytes] num_arrays
    For each array:
      [4 bytes] elem_size (bytes per element)
      [64 bytes] cudaIpcMemHandle_t
    Then waits for 4-byte ACK on stdin before freeing GPU memory.

  On error: status=1, num_result_rows=0, then UTF-8 error message bytes, then
            a NUL terminator byte so parent knows where message ends.
"""

import sys
import struct
import ctypes
import time
import numpy as np

# ── CUDA IPC helpers via ctypes ──────────────────────────────────────────────

_cudart = None

def _init_cuda():
    global _cudart
    if _cudart is not None:
        return
    for name in ['libcudart.so', 'libcudart.so.12', 'libcudart.so.11']:
        try:
            _cudart = ctypes.CDLL(name)
            return
        except OSError:
            continue
    raise RuntimeError("Cannot load libcudart.so")

def cuda_malloc(nbytes):
    _init_cuda()
    ptr = ctypes.c_void_p()
    err = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if err != 0:
        raise RuntimeError(f"cudaMalloc({nbytes}) failed: error {err}")
    return ptr.value

def cuda_memcpy_d2d(dst, src, nbytes):
    _init_cuda()
    err = _cudart.cudaMemcpy(
        ctypes.c_void_p(dst), ctypes.c_void_p(src),
        ctypes.c_size_t(nbytes), ctypes.c_int(3))
    if err != 0:
        raise RuntimeError(f"cudaMemcpy D2D failed: error {err}")

def cuda_ipc_get_handle(ptr):
    _init_cuda()
    handle = (ctypes.c_byte * 64)()
    err = _cudart.cudaIpcGetMemHandle(handle, ctypes.c_void_p(ptr))
    if err != 0:
        raise RuntimeError(f"cudaIpcGetMemHandle failed: error {err}")
    return bytes(handle)

def cuda_free(ptr):
    _init_cuda()
    _cudart.cudaFree(ctypes.c_void_p(ptr))

# ── Main ─────────────────────────────────────────────────────────────────────

ALGO_BFS = 0
ALGO_PAGERANK = 1
ALGO_WCC = 2
ALGO_NAMES = {0: 'BFS', 1: 'PageRank', 2: 'WCC'}

def send_error(msg):
    sys.stdout.buffer.write(struct.pack('<II', 1, 0))  # status=error, 0 rows
    sys.stdout.buffer.write(msg.encode('utf-8'))
    sys.stdout.buffer.write(b'\x00')  # NUL terminator
    sys.stdout.buffer.flush()

def log(msg):
    sys.stderr.write(f"[graph_worker] {msg}\n")
    sys.stderr.flush()

def handle_request(inp, cudf, cugraph):
    """Handle one request. Returns True to continue, False on EOF."""
    header = inp.read(28)
    if len(header) == 0:
        return False  # EOF — parent closed pipe
    if len(header) < 28:
        send_error("incomplete header")
        return False

    algo, num_edges, num_vertices = struct.unpack_from('<III', header, 0)
    params = header[12:28]

    t_start = time.perf_counter()

    src_bytes = inp.read(num_edges * 4)
    dst_bytes = inp.read(num_edges * 4)
    if len(src_bytes) < num_edges * 4 or len(dst_bytes) < num_edges * 4:
        send_error("incomplete edge data")
        return False

    t_read = time.perf_counter()

    src_ids = np.frombuffer(src_bytes, dtype=np.int32)
    dst_ids = np.frombuffer(dst_bytes, dtype=np.int32)

    # Build graph
    edge_df = cudf.DataFrame({'src': cudf.Series(src_ids), 'dst': cudf.Series(dst_ids)})
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(edge_df, source='src', destination='dst', renumber=True)

    t_graph = time.perf_counter()

    # Run algorithm
    ipc_ptrs = []
    try:
        if algo == ALGO_BFS:
            source_vertex, depth_limit_raw = struct.unpack_from('<ii', params, 0)
            depth_limit = None if depth_limit_raw < 0 else depth_limit_raw
            result = cugraph.bfs(G, source_vertex, depth_limit=depth_limit)
            result = result[result['distance'] < 2_147_483_647]
            columns = [
                ('vertex', result['vertex'].astype('int32')),
                ('distance', result['distance'].astype('int32')),
                ('predecessor', result['predecessor'].astype('int32')),
            ]
        elif algo == ALGO_PAGERANK:
            damping, tolerance, max_iter = struct.unpack_from('<ffI', params, 0)
            result = cugraph.pagerank(G, alpha=damping, tol=tolerance, max_iter=max_iter)
            columns = [
                ('vertex', result['vertex'].astype('int32')),
                ('pagerank', result['pagerank'].astype('float32')),
            ]
        elif algo == ALGO_WCC:
            G_und = G.to_undirected()
            result = cugraph.connected_components(G_und)
            columns = [
                ('vertex', result['vertex'].astype('int32')),
                ('labels', result['labels'].astype('int32')),
            ]
        else:
            send_error(f"unknown algorithm {algo}")
            return True  # keep running for next request

        t_algo = time.perf_counter()

        num_rows = len(columns[0][1])
        num_arrays = len(columns)

        # Export each column via CUDA IPC
        handles = []
        for name, col in columns:
            nbytes = len(col) * col.dtype.itemsize
            elem_size = col.dtype.itemsize
            ipc_ptr = cuda_malloc(nbytes)
            ipc_ptrs.append(ipc_ptr)
            cuda_memcpy_d2d(ipc_ptr, col._column.data.ptr, nbytes)
            handle = cuda_ipc_get_handle(ipc_ptr)
            handles.append((elem_size, handle))

        t_ipc = time.perf_counter()

        # Write response
        out = sys.stdout.buffer
        out.write(struct.pack('<III', 0, num_rows, num_arrays))
        for elem_size, handle in handles:
            out.write(struct.pack('<I', elem_size))
            out.write(handle)
        out.flush()

        # Wait for ACK before freeing GPU memory
        ack = inp.read(4)

        t_done = time.perf_counter()

        log(f"{ALGO_NAMES.get(algo, '?')}: {num_edges} edges, {num_rows} result rows | "
            f"read={1000*(t_read-t_start):.0f}ms "
            f"graph_build={1000*(t_graph-t_read):.0f}ms "
            f"algorithm={1000*(t_algo-t_graph):.0f}ms "
            f"ipc_export={1000*(t_ipc-t_algo):.0f}ms "
            f"total={1000*(t_done-t_start):.0f}ms")

        return True  # ready for next request

    finally:
        for ptr in ipc_ptrs:
            cuda_free(ptr)


def main():
    inp = sys.stdin.buffer

    # Import heavy GPU libraries once at startup
    t_init = time.perf_counter()
    import cudf
    import cugraph
    import rmm
    rmm.reinitialize(pool_allocator=True, initial_pool_size=512 * 1024**2)
    t_ready = time.perf_counter()
    log(f"initialized in {1000*(t_ready-t_init):.0f}ms (cudf+cugraph+rmm)")

    # Handle requests in a loop until EOF
    while handle_request(inp, cudf, cugraph):
        pass

    log("exiting (stdin closed)")


if __name__ == '__main__':
    main()
