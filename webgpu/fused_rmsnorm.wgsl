// =============================================================================
// WebGPU / WGSL Compute Shader: Fused RMSNorm
// Performs single-pass mean-square reduction and normalization in workgroup memory
// =============================================================================

struct NormParams {
    total_tokens: u32,
    hidden_dim: u32,
    epsilon: f32,
};

@group(0) @binding(0) var<uniform> params: NormParams;
@group(0) @binding(1) var<storage, read> input_buf: array<f32>;
@group(0) @binding(2) var<storage, read> gamma_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_buf: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;
var<workgroup> s_reduction: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id) block_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = block_id.x;
    let tid = local_id.x;
    let D = params.hidden_dim;

    if (row >= params.total_tokens) {
        return;
    }

    let row_offset = row * D;

    // 1. Thread-local sum of squares
    var local_sum_sq: f32 = 0.0;
    for (var i: u32 = tid; i < D; i = i + WORKGROUP_SIZE) {
        let x = input_buf[row_offset + i];
        local_sum_sq = local_sum_sq + x * x;
    }
    s_reduction[tid] = local_sum_sq;
    workgroupBarrier();

    // 2. Tree reduction in workgroup memory
    for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            s_reduction[tid] = s_reduction[tid] + s_reduction[tid + stride];
        }
        workgroupBarrier();
    }

    // 3. Compute rsqrt factor
    let mean_sq = s_reduction[0] / f32(D);
    let rsqrt_val = inverseSqrt(mean_sq + params.epsilon);
    workgroupBarrier();

    // 4. Normalize and scale output
    for (var i: u32 = tid; i < D; i = i + WORKGROUP_SIZE) {
        let x = input_buf[row_offset + i];
        let g = gamma_buf[i];
        output_buf[row_offset + i] = (x * rsqrt_val) * g;
    }
}
