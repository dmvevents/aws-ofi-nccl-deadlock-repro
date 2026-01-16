/*
 * fi_mr_inject.c - Fault injection for libfabric memory registration
 *
 * This LD_PRELOAD library intercepts fi_mr_regattr() calls and injects
 * -FI_ENOMEM errors at a configurable rate to trigger the MR deadlock
 * bug in aws-ofi-nccl < 1.17.0 (PR #968).
 *
 * The bug: When reg_mr_on_device() encounters an error, it calls dereg_mr()
 * while still holding the mr_cache lock, causing deadlock.
 *
 * Environment variables:
 *   FI_MR_INJECT_ENABLE=1     - Enable fault injection (default: 0)
 *   FI_MR_INJECT_START=500    - Start injecting after N calls (default: 500)
 *   FI_MR_INJECT_RATE=50      - Inject every N calls after start (default: 50)
 *
 * Compile:
 *   gcc -shared -fPIC -O2 -o libfi_mr_inject.so fi_mr_inject.c -ldl -lpthread
 *
 * Use:
 *   export LD_PRELOAD=/path/to/libfi_mr_inject.so
 *   export FI_MR_INJECT_ENABLE=1
 *
 * Based on methodologies from:
 *   - Mycroft (SOSP 2025): Automated RDMA fault injection
 *   - R²CCL (2025): NCCL resilience testing
 *   - bpftime (OSDI 2025): Userspace fault injection
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>

/* libfabric error codes */
#define FI_ENOMEM       12
#define FI_EAGAIN       11

/* libfabric types (forward declarations) */
struct fid_domain;
struct fid_mr;
struct fi_mr_attr;

/* Function pointer types */
typedef int (*fi_mr_regattr_fn)(struct fid_domain *domain,
                                 const struct fi_mr_attr *attr,
                                 uint64_t flags,
                                 struct fid_mr **mr);

/* Global state */
static fi_mr_regattr_fn real_fi_mr_regattr = NULL;
static atomic_ulong call_count = 0;
static atomic_ulong inject_count = 0;
static int inject_enable = 0;
static unsigned long inject_start = 500;
static unsigned long inject_rate = 50;
static int initialized = 0;
static pthread_mutex_t init_lock = PTHREAD_MUTEX_INITIALIZER;

/* Get thread ID for logging - use system gettid() if available */
#ifndef __GLIBC_PREREQ
#define __GLIBC_PREREQ(x, y) 0
#endif
#if !__GLIBC_PREREQ(2, 30)
static inline pid_t my_gettid(void) {
    return syscall(SYS_gettid);
}
#define gettid my_gettid
#endif

/* Initialize from environment variables */
static void init_injection(void) {
    if (initialized) return;

    pthread_mutex_lock(&init_lock);
    if (!initialized) {
        char *val;

        val = getenv("FI_MR_INJECT_ENABLE");
        inject_enable = val ? atoi(val) : 0;

        val = getenv("FI_MR_INJECT_START");
        inject_start = val ? strtoul(val, NULL, 10) : 500;

        val = getenv("FI_MR_INJECT_RATE");
        inject_rate = val ? strtoul(val, NULL, 10) : 50;
        if (inject_rate < 1) inject_rate = 50;

        if (inject_enable) {
            fprintf(stderr, "[FI_MR_INJECT] === FAULT INJECTION ENABLED ===\n");
            fprintf(stderr, "[FI_MR_INJECT] Start after: %lu calls\n", inject_start);
            fprintf(stderr, "[FI_MR_INJECT] Inject every: %lu calls\n", inject_rate);
            fprintf(stderr, "[FI_MR_INJECT] Purpose: Trigger MR deadlock bug (PR #968)\n");
            fprintf(stderr, "[FI_MR_INJECT] ================================\n");
        }

        initialized = 1;
    }
    pthread_mutex_unlock(&init_lock);
}

/* Check if we should inject an error */
static int should_inject(unsigned long count) {
    if (!inject_enable) return 0;
    if (count < inject_start) return 0;
    return ((count - inject_start) % inject_rate) == 0;
}

/* Intercepted fi_mr_regattr - the main MR registration function */
int fi_mr_regattr(struct fid_domain *domain,
                  const struct fi_mr_attr *attr,
                  uint64_t flags,
                  struct fid_mr **mr) {

    init_injection();

    /* Load real function on first call */
    if (!real_fi_mr_regattr) {
        real_fi_mr_regattr = (fi_mr_regattr_fn)dlsym(RTLD_NEXT, "fi_mr_regattr");
        if (!real_fi_mr_regattr) {
            fprintf(stderr, "[FI_MR_INJECT] FATAL: Could not find real fi_mr_regattr\n");
            return -FI_ENOMEM;
        }
    }

    /* Increment call counter */
    unsigned long count = atomic_fetch_add(&call_count, 1);

    /* Check if we should inject an error */
    if (should_inject(count)) {
        unsigned long inj = atomic_fetch_add(&inject_count, 1);
        fprintf(stderr, "[FI_MR_INJECT] [tid=%d] Call #%lu: INJECTING -FI_ENOMEM "
                        "(total injections: %lu)\n",
                gettid(), count, inj + 1);
        return -FI_ENOMEM;
    }

    /* Call real function */
    return real_fi_mr_regattr(domain, attr, flags, mr);
}

/* Also intercept fi_mr_reg which is a wrapper around fi_mr_regattr */
int fi_mr_reg(struct fid_domain *domain,
              const void *buf,
              size_t len,
              uint64_t access,
              uint64_t offset,
              uint64_t requested_key,
              uint64_t flags,
              struct fid_mr **mr,
              void *context) {

    /* This is typically implemented as a macro calling fi_mr_regattr,
     * but some implementations have it as a function. We intercept both. */

    static int (*real_fi_mr_reg)(struct fid_domain*, const void*, size_t,
                                  uint64_t, uint64_t, uint64_t, uint64_t,
                                  struct fid_mr**, void*) = NULL;

    init_injection();

    if (!real_fi_mr_reg) {
        real_fi_mr_reg = dlsym(RTLD_NEXT, "fi_mr_reg");
        if (!real_fi_mr_reg) {
            /* fi_mr_reg might not exist as a separate function */
            return -FI_ENOMEM;
        }
    }

    unsigned long count = atomic_fetch_add(&call_count, 1);

    if (should_inject(count)) {
        unsigned long inj = atomic_fetch_add(&inject_count, 1);
        fprintf(stderr, "[FI_MR_INJECT] [tid=%d] fi_mr_reg call #%lu: INJECTING -FI_ENOMEM "
                        "(total: %lu)\n",
                gettid(), count, inj + 1);
        return -FI_ENOMEM;
    }

    return real_fi_mr_reg(domain, buf, len, access, offset, requested_key,
                          flags, mr, context);
}

/* Destructor - print statistics on exit */
__attribute__((destructor))
static void print_stats(void) {
    if (inject_enable) {
        fprintf(stderr, "\n[FI_MR_INJECT] === FINAL STATISTICS ===\n");
        fprintf(stderr, "[FI_MR_INJECT] Total fi_mr_reg calls: %lu\n",
                atomic_load(&call_count));
        fprintf(stderr, "[FI_MR_INJECT] Injected failures:     %lu\n",
                atomic_load(&inject_count));
        fprintf(stderr, "[FI_MR_INJECT] ========================\n\n");
    }
}
