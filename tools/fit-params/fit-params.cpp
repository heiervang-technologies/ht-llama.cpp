#include "llama.h"
#include "../src/llama-ext.h"

#include "arg.h"
#include "common.h"
#include "fit.h"
#include "log.h"

#include <cinttypes>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// satisfies -Wmissing-declarations
int llama_fit_params(int argc, char ** argv);

int llama_fit_params(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FIT_PARAMS)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (params.fit_params_print_plan) {
        // Router admit-side dry-run (#66). Run common_fit_params with the per-device
        // byte plan output, then emit single-line JSON on stdout:
        //   {"per_device_bytes":[N0,N1,...],"n_devices":K,"total_bytes":T}
        // Entries are GPU/accelerator devices in tensor_split order; host memory
        // is intentionally excluded. Caller subprocesses can parse this without
        // depending on llama internals.
        std::vector<int64_t> plan;
        const common_params_fit_status status = common_fit_params(params.model.path.c_str(), &mparams, &cparams,
                params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_target.data(), params.fit_params_min_ctx,
                params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR,
                &plan);
        if (status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            LOG_ERR("%s: failed to fit CLI arguments to free memory, exiting...\n", __func__);
            // Emit a clean JSON failure marker on stdout so subprocess callers can
            // distinguish fit-failure from parse-failure.
            printf("{\"error\":\"fit_failed\",\"status\":%d}\n", int(status));
            return 1;
        }
        common_log_flush(common_log_main());

        int64_t total = 0;
        for (int64_t b : plan) total += b;

        printf("{\"per_device_bytes\":[");
        for (size_t i = 0; i < plan.size(); ++i) {
            printf("%s%" PRId64, i == 0 ? "" : ",", plan[i]);
        }
        printf("],\"n_devices\":%zu,\"total_bytes\":%" PRId64 "}\n", plan.size(), total);
        return 0;
    }

    if (!params.fit_params_print) {
        const common_params_fit_status status = common_fit_params(params.model.path.c_str(), &mparams, &cparams,
                params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_target.data(), params.fit_params_min_ctx,
                params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
        if (status != COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            LOG_ERR("%s: failed to fit CLI arguments to free memory, exiting...\n", __func__);
            exit(1);
        }

        LOG_INF("%s: printing fitted CLI arguments to stdout...\n", __func__);
        common_log_flush(common_log_main());
        printf("-c %" PRIu32 " -ngl %" PRIi32, cparams.n_ctx, mparams.n_gpu_layers);

        size_t nd = llama_max_devices();
        while (nd > 1 && mparams.tensor_split[nd - 1] == 0.0f) {
            nd--;
        }
        if (nd > 1) {
            for (size_t id = 0; id < nd; id++) {
                if (id == 0) {
                    printf(" -ts ");
                }
                printf("%s%" PRIu32, id > 0 ? "," : "", uint32_t(mparams.tensor_split[id]));
            }
        }

        const size_t ntbo = llama_max_tensor_buft_overrides();
        bool any_tbo = false;
        for (size_t itbo = 0; itbo < ntbo && mparams.tensor_buft_overrides[itbo].pattern != nullptr; itbo++) {
            if (itbo == 0) {
                printf(" -ot \"");
            }
            printf("%s%s=%s", itbo > 0 ? "," : "", mparams.tensor_buft_overrides[itbo].pattern, ggml_backend_buft_name(mparams.tensor_buft_overrides[itbo].buft));
            any_tbo = true;
        }
        printf("%s\n", any_tbo ? "\"" : "");
    } else {
        LOG_INF("%s: printing estimated memory in MiB to stdout (device, model, context, compute) ...\n", __func__);
        common_log_flush(common_log_main());

        common_fit_print(params.model.path.c_str(), &mparams, &cparams);
    }

    return 0;
}
