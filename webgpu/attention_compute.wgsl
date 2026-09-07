// =============================================================================
// WebGPU / WGSL Compute Shader: Tiled Scaled Dot-Product Attention
// Target: Edge LLM Inference in Browser / WebGPU runtime (Transformers.js, WebLLM)
// =============================================================================

struct Uniforms {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    is_causal: u32,
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;

// Workgroup Shared Memory Tiles (Tiling across sequence dimension)
const BLOCK_M: u32 = 16u;
const BLOCK_N: u32 = 16u;
const MAX_HEAD_DIM: u32 = 64u;

var<workgroup> s_Q: array<f32, 1024>; // 16 * 64
var<workgroup> s_K: array<f32, 1024>; // 16 * 64
var<workgroup> s_V: array<f32, 1024>; // 16 * 64

@compute @workgroup_size(16, 1, 1)
fn main(
    @builtin(workgroup_id) block_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let block_m = block_id.x;
    let batch_head = block_id.z;
    let tid = local_id.x;

    let B = params.batch_size;
    let H = params.num_heads;
    let N = params.seq_len;
    let D = params.head_dim;
    let scale = params.scale;

    let global_row = block_m * BLOCK_M + tid;
    let base_offset = batch_head * N * D;

    // Running online softmax statistics
    var m_i: f32 = -1e30;
    var l_i: f32 = 0.0;
    var acc_o: array<f32, 64>;

    for (var d: u32 = 0u; d < D; d = d + 1u) {
        acc_o[d] = 0.0;
    }

    // 1. Load Query Tile into Workgroup Memory
    for (var d: u32 = 0u; d < D; d = d + 1u) {
        if (global_row < N && d < D) {
            s_Q[tid * D + d] = Q[base_offset + global_row * D + d];
        } else {
            s_Q[tid * D + d] = 0.0;
        }
    }
    workgroupBarrier();

    // 2. Iterate over Key & Value blocks
    let num_blocks_n = (N + BLOCK_N - 1u) / BLOCK_N;
    for (var kv_b: u32 = 0u; kv_b < num_blocks_n; kv_b = kv_b + 1u) {
        // Load K and V
        for (var d: u32 = 0u; d < D; d = d + 1u) {
            let kv_row = kv_b * BLOCK_N + tid;
            if (kv_row < N && d < D) {
                s_K[tid * D + d] = K[base_offset + kv_row * D + d];
                s_V[tid * D + d] = V[base_offset + kv_row * D + d];
            } else {
                s_K[tid * D + d] = 0.0;
                s_V[tid * D + d] = 0.0;
            }
        }
        workgroupBarrier();

        if (global_row < N) {
            var s_row: array<f32, 16>;
            var tile_max: f32 = -1e30;

            // Compute dot product S = (Q * K^T) * scale
            for (var j: u32 = 0u; j < BLOCK_N; j = j + 1u) {
                let kv_col = kv_b * BLOCK_N + j;
                if (kv_col < N && (params.is_causal == 0u || global_row >= kv_col)) {
                    var dot: f32 = 0.0;
                    for (var d: u32 = 0u; d < D; d = d + 1u) {
                        dot = dot + s_Q[tid * D + d] * s_K[j * D + d];
                    }
                    let score = dot * scale;
                    s_row[j] = score;
                    tile_max = max(tile_max, score);
                } else {
                    s_row[j] = -1e30;
                }
            }

            // Online Softmax update
            let m_prev = m_i;
            let m_new = max(m_prev, tile_max);
            let correction = exp(m_prev - m_new);

            var sum_p: f32 = 0.0;
            for (var j: u32 = 0u; j < BLOCK_N; j = j + 1u) {
                if (s_row[j] > -1e25) {
                    s_row[j] = exp(s_row[j] - m_new);
                    sum_p = sum_p + s_row[j];
                } else {
                    s_row[j] = 0.0;
                }
            }

            l_i = l_i * correction + sum_p;
            m_i = m_new;

            // Update output accumulator
            for (var d: u32 = 0u; d < D; d = d + 1u) {
                var pv: f32 = 0.0;
                for (var j: u32 = 0u; j < BLOCK_N; j = j + 1u) {
                    pv = pv + s_row[j] * s_V[j * D + d];
                }
                acc_o[d] = acc_o[d] * correction + pv;
            }
        }
        workgroupBarrier();
    }

    // 3. Write output to global storage buffer
    if (global_row < N) {
        let inv_l = 1.0 / (l_i + 1e-6);
        for (var d: u32 = 0u; d < D; d = d + 1u) {
            O[base_offset + global_row * D + d] = acc_o[d] * inv_l;
        }
    }
}
