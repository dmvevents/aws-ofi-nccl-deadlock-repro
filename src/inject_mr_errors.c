/*
 * inject_mr_errors.c - Fault injection for fi_mr_reg
 *
 * This LD_PRELOAD library intercepts libfabric fi_mr_reg() calls and
 * injects failures to trigger the aws-ofi-nccl deadlock bug.
 *
 * Environment variables:
 *   MR_ERROR_RATE  - Inject error every N calls (default: 0 = disabled)
 *   MR_ERROR_START - Start injection after N calls (default: 500)
 *
 * Build:
 *   gcc -shared -fPIC -O2 -o inject_mr_errors.so inject_mr_errors.c -ldl -lpthread
 *
 * Usage:
 *   LD_PRELOAD=/path/to/inject_mr_errors.so MR_ERROR_RATE=50 ./your_app
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <errno.h>

/* libfabric types (minimal definitions) */
struct fid_domain;
struct fid_mr;

typedef int (*fi_mr_reg_fn)(struct fid_domain *domain, const void *buf,
                            size_t len, uint64_t access, uint64_t offset,
                            uint64_t requested_key, uint64_t flags,
                            struct fid_mr **mr, void *context);

static fi_mr_reg_fn real_fi_mr_reg = NULL;
static pthread_mutex_t init_lock = PTHREAD_MUTEX_INITIALIZER;
static int initialized = 0;

/* Configuration */
static int error_rate = 0;      /* Inject every N calls (0 = disabled) */
static int error_start = 500;   /* Start after N calls */

/* Counters */
static volatile uint64_t call_count = 0;
static volatile uint64_t error_count = 0;

static void init_interceptor(void) {
    pthread_mutex_lock(&init_lock);
    if (!initialized) {
        /* Load real fi_mr_reg */
        real_fi_mr_reg = (fi_mr_reg_fn)dlsym(RTLD_NEXT, "fi_mr_reg");
        if (!real_fi_mr_reg) {
            fprintf(stderr, "[MR_INJECT] ERROR: Cannot find fi_mr_reg: %s\n", dlerror());
            pthread_mutex_unlock(&init_lock);
            return;
        }

        /* Parse configuration */
        const char *rate_env = getenv("MR_ERROR_RATE");
        const char *start_env = getenv("MR_ERROR_START");

        if (rate_env) error_rate = atoi(rate_env);
        if (start_env) error_start = atoi(start_env);

        fprintf(stderr, "[MR_INJECT] Initialized: rate=1/%d, start_after=%d calls\n",
                error_rate, error_start);

        initialized = 1;
    }
    pthread_mutex_unlock(&init_lock);
}

/*
 * Intercepted fi_mr_reg
 *
 * Injects -FI_ENOMEM errors to simulate memory registration failures
 * that trigger the deadlock bug in aws-ofi-nccl < 1.17.2
 */
int fi_mr_reg(struct fid_domain *domain, const void *buf, size_t len,
              uint64_t access, uint64_t offset, uint64_t requested_key,
              uint64_t flags, struct fid_mr **mr, void *context) {

    if (!initialized) {
        init_interceptor();
    }

    if (!real_fi_mr_reg) {
        return -12; /* -FI_ENOMEM */
    }

    uint64_t count = __sync_add_and_fetch(&call_count, 1);

    /* Check if we should inject an error */
    if (error_rate > 0 && count > error_start && (count % error_rate) == 0) {
        uint64_t errs = __sync_add_and_fetch(&error_count, 1);

        fprintf(stderr, "[MR_INJECT] Injecting fi_mr_reg failure #%lu at call %lu "
                        "(buf=%p, len=%zu)\n", errs, count, buf, len);

        /* Return -FI_ENOMEM to simulate memory registration failure */
        return -12;
    }

    /* Call the real function */
    return real_fi_mr_reg(domain, buf, len, access, offset, requested_key,
                          flags, mr, context);
}

/*
 * Statistics reporting (called at program exit)
 */
__attribute__((destructor))
static void report_stats(void) {
    if (initialized && error_rate > 0) {
        fprintf(stderr, "[MR_INJECT] Final stats: %lu calls, %lu errors injected\n",
                call_count, error_count);
    }
}
