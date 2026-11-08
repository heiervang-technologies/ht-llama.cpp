#pragma once

#include "ggml.h"

// Quantized blocks cannot be transposed as individual scalar elements. Keep
// this check shared by scheduling and direct/planned CPU graph execution.
static inline bool ggml_cpu_quantized_rows_supported(const struct ggml_tensor * tensor) {
    if (!ggml_is_quantized(tensor->type)) {
        return true;
    }
    if (tensor->nb[0] != ggml_type_size(tensor->type)) {
        return false;
    }
    // A one-block row can have nb[0] == nb[1], so strides alone cannot
    // distinguish its invalid scalar transpose. Follow view provenance too.
    int axis = 0;
    for (const struct ggml_tensor * cur = tensor; cur != NULL; cur = cur->src[0]) {
        switch (cur->op) {
            case GGML_OP_TRANSPOSE:
                if (axis < 2) { axis = 1 - axis; }
                break;
            case GGML_OP_PERMUTE:
                for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                    if (cur->op_params[i] == axis) {
                        axis = i;
                        break;
                    }
                }
                break;
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
                if (axis != 0) { return false; }
                break;
            default:
                return axis == 0;
        }
    }
    return axis == 0;
}

static inline bool ggml_cpu_copy_layout_supported(const struct ggml_tensor * op) {
    if (op->op != GGML_OP_CPY && op->op != GGML_OP_CONT && op->op != GGML_OP_DUP) {
        return true;
    }
    const struct ggml_tensor * src = op->src[0];
    const struct ggml_tensor * dst = op->op == GGML_OP_CPY ? op->src[1] : op;
    return ggml_cpu_quantized_rows_supported(src) && ggml_cpu_quantized_rows_supported(dst);
}
