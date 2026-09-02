/* ds4_q4e_page.h -- qwen4exp paged-KV geometry, shared by the kernels and the
 * host.
 *
 * A page is DS4_Q4E_PAGE_TOKENS token positions of every KV row: QSA K and V
 * for each of the twelve attention layers, the QSA indexer's raw and pooled
 * keys, and the draft head's K and V.  That is 30 KiB per position, so 7.5 MiB
 * per page across all layers.
 *
 * Why 256.  It is a multiple of the indexer's 4-position pooling block, so a
 * pooled block never straddles a page and a page's pooled keys are exactly its
 * own quarter of the pooled rows; a multiple of the attention kernels' 32-key
 * tile; and a multiple of the indexer score kernel's 64-block tile (which is
 * then exactly one page).  So every tiled reader translates one position per
 * tile and walks the rest of the tile contiguously.
 *
 * A kernel turns a logical position p of the sequence whose page table is
 * `pages` into a physical cache row with
 *     pages[p / DS4_Q4E_PAGE_TOKENS] * DS4_Q4E_PAGE_TOKENS + p % DS4_Q4E_PAGE_TOKENS
 * (q4e_kv_row in ds4_qwen4exp_gpu.cuh).  Pages are immutable once the
 * sequence's frontier has moved past them, so rows a kernel is reading never
 * move under it.
 *
 * This header is macros only.  ds4.c includes it directly and outside its
 * DS4_NO_GPU guard, because the page pool and the span tree are host-side
 * bookkeeping that a CPU build compiles and ds4_test exercises;
 * ds4_qwen4exp_gpu.cuh includes it for the kernels.  ds4_gpu.h deliberately
 * does not, so the geometry does not reach the dozen translation units that
 * only want the tensor API.
 */
#ifndef DS4_Q4E_PAGE_H
#define DS4_Q4E_PAGE_H

#define DS4_Q4E_PAGE_TOKENS 256u
#define DS4_Q4E_PAGE_SHIFT  8u
#define DS4_Q4E_PAGE_MASK   (DS4_Q4E_PAGE_TOKENS - 1u)

#endif /* DS4_Q4E_PAGE_H */
