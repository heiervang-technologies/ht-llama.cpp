#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

static void check_rejection(ggml_backend_t cpu, ggml_type type, ggml_op op, bool transpose_dst, int width = 256) {
    auto * ctx = ggml_init({4*1024*1024, nullptr, true});
    const int heads = width == 32 ? 1 : 8;
    auto * raw = ggml_new_tensor_4d(ctx, type, width, heads, 256, 1);
    auto * perm = ggml_permute(ctx, raw, 0, 2, 1, 3);
    auto * trans = ggml_transpose(ctx, perm);
    auto * dst = ggml_new_tensor_4d(ctx, type, 256, width, heads, 1);
    auto * out = op == GGML_OP_CONT ? ggml_cont(ctx, trans) :
                op == GGML_OP_DUP ? ggml_dup(ctx, trans) :
                transpose_dst ? ggml_cpy(ctx, dst, trans) : ggml_cpy(ctx, trans, dst);
    std::vector<unsigned char> source(ggml_nbytes(raw), 0x3C);
    std::vector<unsigned char> target(ggml_nbytes(out), 0xA5);
    raw->data = perm->data = trans->data = transpose_dst ? target.data() : source.data();
    dst->data = transpose_dst ? source.data() : target.data();
    out->data = target.data();
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(!ggml_backend_supports_op(cpu, out));
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_FAILED);
    auto plan = ggml_backend_graph_plan_create(cpu, graph);
    GGML_ASSERT(plan != nullptr);
    GGML_ASSERT(ggml_backend_graph_plan_compute(cpu, plan) == GGML_STATUS_FAILED);
    ggml_backend_graph_plan_free(cpu, plan);
    GGML_ASSERT(std::all_of(target.begin(), target.end(), [](unsigned char x) { return x == 0xA5; }));
    ggml_free(ctx);
}

// Exercise row permutations plus a differently shaped destination. The old
// generic copy multiplied the row length by block bytes instead of row bytes.
static void check_valid(ggml_backend_t cpu, ggml_type type, int threads) {
    ggml_backend_cpu_set_n_threads(cpu, threads);
    auto * ctx = ggml_init({4*1024*1024, nullptr, true});
    auto * raw = ggml_new_tensor_3d(ctx, type, 256, 8, 4);
    auto * perm = ggml_permute(ctx, raw, 0, 2, 1, 3);
    auto * dst = ggml_new_tensor_3d(ctx, type, 64, 4, 32);
    auto * out = ggml_cpy(ctx, perm, dst);
    const size_t bytes = ggml_nbytes(raw);
    const size_t row = ggml_row_size(type, 256);
    std::vector<unsigned char> source(bytes), target(bytes + 64, 0xA5), expected(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        source[i] = i % 251;
    }
    for (size_t y = 0; y < 8; ++y) {
        for (size_t z = 0; z < 4; ++z) {
            std::memcpy(expected.data() + (y*4 + z)*row, source.data() + (z*8 + y)*row, row);
        }
    }
    raw->data = perm->data = source.data();
    dst->data = out->data = target.data();
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_supports_op(cpu, out));
    GGML_ASSERT(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(std::equal(expected.begin(), expected.end(), target.begin()));
    GGML_ASSERT(std::all_of(target.begin() + bytes, target.end(), [](unsigned char x) { return x == 0xA5; }));
    ggml_free(ctx);
}

int main() {
    auto cpu = ggml_backend_cpu_init();
    GGML_ASSERT(cpu);
    for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        for (auto op : {GGML_OP_CONT, GGML_OP_DUP, GGML_OP_CPY}) {
            check_rejection(cpu, type, op, false);
            check_rejection(cpu, type, op, false, 32);
        }
        check_rejection(cpu, type, GGML_OP_CPY, true);
        check_rejection(cpu, type, GGML_OP_CPY, true, 32);
    }
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        for (int threads : {1, 4}) {
            check_valid(cpu, type, threads);
        }
    }
    ggml_backend_free(cpu);
    std::puts("quantized copy rejection and valid reshaped copies: PASS");
}
