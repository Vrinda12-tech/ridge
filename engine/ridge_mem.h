/*
 * ridge_mem.h -- logical-block memory manager: a small physical pool in one
 * mmap'd region, plus a spill file that makes the pool a cache for a larger
 * logical space.
 *
 * The split matters and is the whole point:
 *   - a LOGICAL block is a stable handle the caller keeps (a KV block for a token
 *     range, a layer's weights). There can be many.
 *   - a PHYSICAL slot is a fixed-size window into RAM. There are few.
 *
 * alloc() hands back a logical id and binds it to a physical slot, evicting and
 * spilling another logical block if the pool is full. touch() guarantees the
 * bytes are in RAM. So resident memory is a policy we set, not a consequence of
 * how many tokens the user typed.
 *
 * This is paged attention's block table, and it is the same shape as MNN's CPU KV
 * manager (in-memory tensor or mmap'd file, with a move-to-disk spill).
 */
#ifndef RIDGE_MEM_H
#define RIDGE_MEM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RIDGE_BLOCK_NONE (-1)

/* madvise advice, mirrored so callers need not include sys/mman.h. */
enum {
  RIDGE_ADVISE_WILLNEED = 0,
  RIDGE_ADVISE_SEQUENTIAL = 1,
  RIDGE_ADVISE_DONTNEED = 2,
  RIDGE_ADVISE_RANDOM = 3,
};

typedef struct {
  size_t allocs, frees, evictions, spills, recalls, touches;
  size_t spill_writes, spill_reads;
  size_t resident_bytes, peak_resident_bytes;
  size_t logical_in_use;
} ridge_mem_stats;

typedef struct ridge_mem {
  uint8_t *region;          /* the physical pool: one mapping */
  size_t region_bytes;      /* == n_phys * block_bytes */
  size_t block_bytes;
  size_t n_phys;

  int32_t n_logical;        /* how many logical blocks the caller may hold */
  int32_t *log_phys;        /* logical -> physical slot, or -1 */
  uint8_t *log_resident;    /* logical: pages currently in the pool */
  uint8_t *log_dirty;       /* logical: needs a write-back before eviction */
  uint8_t *log_inuse;
  uint64_t *log_lru;
  uint64_t clock;

  int32_t *phys_free;       /* stack of free physical slots */
  int32_t phys_free_top;

  int32_t *log_free;        /* stack of free logical ids */
  int32_t log_free_top;

  int spill_fd;
  size_t spill_bytes;
  char spill_path[256];

  ridge_mem_stats stats;
} ridge_mem;

/* Create a manager with a pool of pool_bytes (rounded down to whole blocks) and
 * max_logical logical blocks. spill_path may be NULL for memory-only. */
ridge_mem *ridge_mem_new(size_t pool_bytes, size_t block_bytes,
                         int32_t max_logical, const char *spill_path);

void ridge_mem_free(ridge_mem *m);

/* Allocate a logical block. Returns an id, or RIDGE_BLOCK_NONE if max_logical is
 * reached. May spill another block to make room. */
int32_t ridge_mem_alloc(ridge_mem *m);

/* Free a logical block and return its physical slot to the pool. */
void ridge_mem_release(ridge_mem *m, int32_t id);

/* Writable pointer to a logical block's bytes, recalling it from the spill file
 * and evicting if needed. Never returns a pointer to non-resident bytes. */
void *ridge_mem_touch(ridge_mem *m, int32_t id);

/* Mark a logical block as needing a write-back before it can be dropped. */
void ridge_mem_mark_dirty(ridge_mem *m, int32_t id);

int ridge_mem_is_resident(const ridge_mem *m, int32_t id);

/* Force a block out to the spill file and hand its slot back. Returns 0 on
 * success, -1 if the block is dirty and there is no spill file. */
int ridge_mem_spill(ridge_mem *m, int32_t id);

/* Page-granularity hints for a weight mapping. */
int ridge_mem_advise(ridge_mem *m, void *addr, size_t bytes, int advice);

size_t ridge_mem_resident_blocks(const ridge_mem *m);

/* Returns 0 on success, non-zero on the first failed invariant. */
int ridge_mem_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* RIDGE_MEM_H */
