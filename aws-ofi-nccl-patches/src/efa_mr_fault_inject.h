/*
 * EFA Memory Registration Fault Injection
 * For scientific testing of aws-ofi-nccl deadlock bug (PR #968)
 *
 * Environment variables:
 *   FI_EFA_MR_INJECT_ENABLE=1   Enable fault injection
 *   FI_EFA_MR_INJECT_START=500  Start after N successful calls
 *   FI_EFA_MR_INJECT_RATE=50    Inject error every N calls
 */

#ifndef EFA_MR_FAULT_INJECT_H
#define EFA_MR_FAULT_INJECT_H

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

static atomic_int _fi_mr_call_count = 0;
static int _fi_mr_inject_enabled = -1;
static int _fi_mr_inject_start = 500;
static int _fi_mr_inject_rate = 50;

static void _fi_mr_inject_init(void)
{
    const char *env;
    _fi_mr_inject_enabled = 0;

    env = getenv("FI_EFA_MR_INJECT_ENABLE");
    if (env && atoi(env) == 1) {
        _fi_mr_inject_enabled = 1;

        env = getenv("FI_EFA_MR_INJECT_START");
        if (env) _fi_mr_inject_start = atoi(env);

        env = getenv("FI_EFA_MR_INJECT_RATE");
        if (env) _fi_mr_inject_rate = atoi(env);

        fprintf(stderr, "[EFA_MR_INJECT] ENABLED: start=%d rate=%d\n",
                _fi_mr_inject_start, _fi_mr_inject_rate);
    }
}

static int _fi_mr_should_inject(void)
{
    int count;

    if (_fi_mr_inject_enabled < 0)
        _fi_mr_inject_init();

    if (!_fi_mr_inject_enabled)
        return 0;

    count = atomic_fetch_add(&_fi_mr_call_count, 1);

    if (count < _fi_mr_inject_start)
        return 0;

    return ((count - _fi_mr_inject_start) % _fi_mr_inject_rate) == 0;
}

#define EFA_MR_FAULT_INJECT_CHECK() \
    do { \
        if (_fi_mr_should_inject()) { \
            fprintf(stderr, "[EFA_MR_INJECT] Injecting -FI_ENOMEM at call %d\n", \
                    atomic_load(&_fi_mr_call_count)); \
            free(efa_mr); \
            return -FI_ENOMEM; \
        } \
    } while(0)

#endif /* EFA_MR_FAULT_INJECT_H */
