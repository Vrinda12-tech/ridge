/*
 * ridge_mem.c -- logical-block memory manager over a small physical pool.
 *
 * The design rests on one fact from the kernel's page cache: a clean, file-backed
 * page can be dropped at any time and re-faulted later for the cost of one read,
 * because there is nothing to write back. That is what makes it safe to be
 * aggressive with madvise, and it is what makes a spill file a real extension of
 * RAM rather than a copy of it.
 *
 * Nothing here calls malloc in the hot path. The pool is one mmap; slots are
 * carved out of it; the maps are flat arrays.
 *
 * Build:
 *   gcc -O3 -march=native -c -o bench/ridge_mem.o engine/ridge_mem.c
 *
 * Contract: alloc returns a logical id or RIDGE_BLOCK_NONE; touch never returns a
 * pointer to non-resident bytes; a dirty block is always written before it is
 * dropped.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ridge_mem.h"

/* ------------------------------------------------------------ construction -- */

ridge_mem *ridge_mem_new(size_t pool_bytes, size_t block_bytes,
                         int32_t max_logical, const char *spill_path) {
  if (pool_bytes < block_bytes || block_bytes == 0 || max_logical <= 0) return NULL;

  ridge_mem *m = (ridge_mem *)calloc(1, sizeof(ridge_mem));
  if (!m) return NULL;

  m->block_bytes = block_bytes;
  m->n_phys = pool_bytes / block_bytes;
  m->region_bytes = m->n_phys * block_bytes;
  m->n_logical = max_logical;

  m->region = mmap(NULL, m->region_bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m->region == MAP_FAILED) {
    free(m);
    return NULL;
  }

  m->log_phys = (int32_t *)malloc((size_t)max_logical * sizeof(int32_t));
  m->log_resident = (uint8_t *)calloc((size_t)max_logical, 1);
  m->log_dirty = (uint8_t *)calloc((size_t)max_logical, 1);
  m->log_inuse = (uint8_t *)calloc((size_t)max_logical, 1);
  m->log_lru = (uint64_t *)calloc((size_t)max_logical, sizeof(uint64_t));
  m->phys_free = (int32_t *)malloc(m->n_phys * sizeof(int32_t));
  m->log_free = (int32_t *)malloc((size_t)max_logical * sizeof(int32_t));
  if (!m->log_phys || !m->log_resident || !m->log_dirty || !m->log_inuse ||
      !m->log_lru || !m->phys_free || !m->log_free) {
    ridge_mem_free(m);
    return NULL;
  }

  for (int32_t i = 0; i < max_logical; i++) {
    m->log_phys[i] = -1;
    m->log_free[i] = max_logical - 1 - i; /* so id 0 is handed out first */
  }
  m->log_free_top = max_logical;

  for (size_t i = 0; i < m->n_phys; i++) m->phys_free[i] = (int32_t)i;
  m->phys_free_top = (int32_t)m->n_phys;

  m->spill_fd = -1;
  if (spill_path) {
    snprintf(m->spill_path, sizeof(m->spill_path), "%s", spill_path);
    m->spill_fd = open(spill_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (m->spill_fd >= 0) {
      /* Preallocate the whole logical space so spill writes never allocate at
       * the wrong moment. The file is sparse until written. */
      off_t full = (off_t)max_logical * (off_t)block_bytes;
      if (ftruncate(m->spill_fd, full) != 0) {
        close(m->spill_fd);
        m->spill_fd = -1;
        m->spill_path[0] = '\0';
      }
    } else {
      m->spill_path[0] = '\0';
    }
  }
  return m;
}

void ridge_mem_free(ridge_mem *m) {
  if (!m) return;
  if (m->region && m->region != MAP_FAILED) munmap(m->region, m->region_bytes);
  if (m->spill_fd >= 0) {
    close(m->spill_fd);
    if (m->spill_path[0]) unlink(m->spill_path);
  }
  free(m->log_phys);
  free(m->log_resident);
  free(m->log_dirty);
  free(m->log_inuse);
  free(m->log_lru);
  free(m->phys_free);
  free(m->log_free);
  free(m);
}

/* ------------------------------------------------------------------ spill --- */

static int write_block(ridge_mem *m, int32_t id, int32_t slot) {
  off_t off = (off_t)id * (off_t)m->block_bytes;
  const uint8_t *src = m->region + (size_t)slot * m->block_bytes;
  size_t left = m->block_bytes;
  while (left) {
    ssize_t w = pwrite(m->spill_fd, src, left, off);
    if (w <= 0) return -1;
    src += w;
    off += w;
    left -= (size_t)w;
  }
  m->stats.spill_writes++;
  m->spill_bytes += m->block_bytes;
  return 0;
}

static int read_block(ridge_mem *m, int32_t id, int32_t slot) {
  off_t off = (off_t)id * (off_t)m->block_bytes;
  uint8_t *dst = m->region + (size_t)slot * m->block_bytes;
  size_t left = m->block_bytes;
  while (left) {
    ssize_t r = pread(m->spill_fd, dst, left, off);
    if (r <= 0) return -1;
    dst += r;
    off += r;
    left -= (size_t)r;
  }
  m->stats.spill_reads++;
  return 0;
}

int ridge_mem_spill(ridge_mem *m, int32_t id) {
  if (id < 0 || id >= m->n_logical) return -1;
  if (!m->log_inuse[id]) return -1;
  if (!m->log_resident[id]) return 0;               /* already out */
  if (m->log_dirty[id] && m->spill_fd < 0) return -1;

  int32_t slot = m->log_phys[id];
  if (m->log_dirty[id]) {
    if (write_block(m, id, slot) != 0) return -1;
    m->log_dirty[id] = 0;
  }

  /* Hand the pages back to the kernel. A clean page costs nothing to lose. */
  madvise(m->region + (size_t)slot * m->block_bytes, m->block_bytes, MADV_DONTNEED);
  m->log_resident[id] = 0;
  m->phys_free[m->phys_free_top++] = slot;
  m->log_phys[id] = -1;
  m->stats.resident_bytes -= m->block_bytes;
  m->stats.spills++;
  return 0;
}

/* --------------------------------------------------------------- eviction --- */

static int32_t choose_victim(ridge_mem *m) {
  /* LRU over resident, evictable logical blocks. The lru counter is monotonic,
   * so the minimum is the oldest. A dirty block needs a spill file. */
  int32_t best = -1;
  uint64_t best_t = UINT64_MAX;
  for (int32_t i = 0; i < m->n_logical; i++) {
    if (!m->log_inuse[i] || !m->log_resident[i]) continue;
    if (m->log_dirty[i] && m->spill_fd < 0) continue;
    if (m->log_lru[i] < best_t) {
      best_t = m->log_lru[i];
      best = i;
    }
  }
  return best;
}

/* Make one physical slot available, spilling a victim if the pool is full. */
static int32_t acquire_slot(ridge_mem *m) {
  if (m->phys_free_top > 0) return m->phys_free[--m->phys_free_top];
  int32_t v = choose_victim(m);
  if (v < 0) return RIDGE_BLOCK_NONE;
  if (ridge_mem_spill(m, v) != 0) return RIDGE_BLOCK_NONE;
  m->stats.evictions++;
  if (m->phys_free_top <= 0) return RIDGE_BLOCK_NONE;
  return m->phys_free[--m->phys_free_top];
}

/* ------------------------------------------------------------------ alloc --- */

int32_t ridge_mem_alloc(ridge_mem *m) {
  if (m->log_free_top <= 0) return RIDGE_BLOCK_NONE;
  int32_t slot = acquire_slot(m);
  if (slot == RIDGE_BLOCK_NONE) return RIDGE_BLOCK_NONE;

  int32_t id = m->log_free[--m->log_free_top];
  m->log_inuse[id] = 1;
  m->log_phys[id] = slot;
  m->log_resident[id] = 1;
  m->log_dirty[id] = 0;
  m->log_lru[id] = ++m->clock;
  m->stats.resident_bytes += m->block_bytes;
  if (m->stats.resident_bytes > m->stats.peak_resident_bytes)
    m->stats.peak_resident_bytes = m->stats.resident_bytes;
  m->stats.allocs++;
  m->stats.logical_in_use++;
  return id;
}

void ridge_mem_release(ridge_mem *m, int32_t id) {
  if (id < 0 || id >= m->n_logical || !m->log_inuse[id]) return;
  if (m->log_resident[id]) {
    int32_t slot = m->log_phys[id];
    madvise(m->region + (size_t)slot * m->block_bytes, m->block_bytes, MADV_DONTNEED);
    m->phys_free[m->phys_free_top++] = slot;
    m->stats.resident_bytes -= m->block_bytes;
  }
  m->log_inuse[id] = 0;
  m->log_resident[id] = 0;
  m->log_dirty[id] = 0;
  m->log_phys[id] = -1;
  m->log_free[m->log_free_top++] = id;
  m->stats.frees++;
  m->stats.logical_in_use--;
}

/* ------------------------------------------------------------------ touch --- */

void *ridge_mem_touch(ridge_mem *m, int32_t id) {
  if (id < 0 || id >= m->n_logical || !m->log_inuse[id]) return NULL;

  if (!m->log_resident[id]) {
    int32_t slot = acquire_slot(m);
    if (slot == RIDGE_BLOCK_NONE) return NULL;
    if (m->spill_fd >= 0) read_block(m, id, slot);
    m->log_phys[id] = slot;
    m->log_resident[id] = 1;
    m->stats.resident_bytes += m->block_bytes;
    if (m->stats.resident_bytes > m->stats.peak_resident_bytes)
      m->stats.peak_resident_bytes = m->stats.resident_bytes;
    m->stats.recalls++;
  }
  m->log_lru[id] = ++m->clock;
  m->stats.touches++;
  return m->region + (size_t)m->log_phys[id] * m->block_bytes;
}

void ridge_mem_mark_dirty(ridge_mem *m, int32_t id) {
  if (id < 0 || id >= m->n_logical || !m->log_inuse[id]) return;
  m->log_dirty[id] = 1;
}

int ridge_mem_is_resident(const ridge_mem *m, int32_t id) {
  if (id < 0 || id >= m->n_logical || !m->log_inuse[id]) return 0;
  return m->log_resident[id];
}

/* ------------------------------------------------------------- weight hint -- */

int ridge_mem_advise(ridge_mem *m, void *addr, size_t bytes, int advice) {
  (void)m;
  if (!addr || bytes == 0) return -1;
  switch (advice) {
    case RIDGE_ADVISE_WILLNEED:   return madvise(addr, bytes, MADV_WILLNEED);
    case RIDGE_ADVISE_SEQUENTIAL: return madvise(addr, bytes, MADV_SEQUENTIAL);
    case RIDGE_ADVISE_DONTNEED:   return madvise(addr, bytes, MADV_DONTNEED);
    case RIDGE_ADVISE_RANDOM:     return madvise(addr, bytes, MADV_RANDOM);
    default:                      return -1;
  }
}

size_t ridge_mem_resident_blocks(const ridge_mem *m) {
  return m->stats.resident_bytes / m->block_bytes;
}

/* ---------------------------------------------------------------- selftest -- */

int ridge_mem_selftest(void) {
  const size_t blk = 4096;
  const int32_t nlog = 64;
  const size_t pool = 16 * blk;   /* only 16 of 64 logical blocks fit at once */
  char path[256];
  snprintf(path, sizeof(path), "/tmp/.ridge_mem_selftest_%d.bin", (int)getpid());

  ridge_mem *m = ridge_mem_new(pool, blk, nlog, path);
  if (!m) return 1;

  /* Write a signature into every block. The small pool forces spills; the
   * signatures prove nothing was lost. Hold all 64 ids at once. */
  int32_t *ids = (int32_t *)malloc((size_t)nlog * sizeof(int32_t));
  for (int32_t i = 0; i < nlog; i++) {
    ids[i] = ridge_mem_alloc(m);
    if (ids[i] < 0) { free(ids); ridge_mem_free(m); return 2; }
    uint32_t *p = (uint32_t *)ridge_mem_touch(m, ids[i]);
    if (!p) { free(ids); ridge_mem_free(m); return 7; }
    p[0] = (uint32_t)((uint32_t)i * 2654435761u + 12345u);
    ridge_mem_mark_dirty(m, ids[i]);
  }

  /* Resident bytes must never exceed the pool, at any moment. */
  if (m->stats.resident_bytes > pool) { free(ids); ridge_mem_free(m); return 5; }
  if (ridge_mem_resident_blocks(m) > pool / blk) { free(ids); ridge_mem_free(m); return 3; }

  /* Read every block back; each must still hold its signature. */
  for (int32_t i = 0; i < nlog; i++) {
    uint32_t *p = (uint32_t *)ridge_mem_touch(m, ids[i]);
    if (!p) { free(ids); ridge_mem_free(m); return 8; }
    if (p[0] != (uint32_t)((uint32_t)i * 2654435761u + 12345u)) {
      free(ids); ridge_mem_free(m); return 4;
    }
  }
  if (m->stats.resident_bytes > pool) { free(ids); ridge_mem_free(m); return 6; }

  /* Release everything; the free lists must come back to full. */
  for (int32_t i = 0; i < nlog; i++) ridge_mem_release(m, ids[i]);
  if (m->stats.logical_in_use != 0) { free(ids); ridge_mem_free(m); return 9; }

  int ok = (m->stats.spills > 0) && (m->stats.recalls > 0);
  free(ids);
  ridge_mem_free(m);
  return ok ? 0 : 10;
}
