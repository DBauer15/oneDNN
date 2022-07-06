
## Convolution Type

The kernel seems to do direct convolution after all since only the OW dimension varies which suggests that the image is processed line-by-line similar to XeSS.
Inside of the kernel the convolution is performed in the GEMM-like form as described in the README.

## Algorithm Description
Overall the unoptimized IR produced by the generator follows this sequence of steps:

1. Allocate `A`,`B`, `C` buffers and fill `C` with zeros
2. Initiate compute loop, which is defined as
```python
for ic_outer in range(2) # 2 b/c we have 16 iterations and 32 IC
    for kd in range(1) # This is optional in 2D scenarios
        for kh in range(3) # 3 b/c we have 3x3 kernels
            for kw_outer in range(3) # 3 b/c we have 3x3 kernels
```

Obviously some values represented int `src`, `wei`, and `dst` are missing here. These dimensions are handled with offsets.
They are implicitely defined by the dispatch size of the kernel and are invariably calculated as:
```python
oc_outer = grid_idx0
mb_outer = grid_idx2
oc_inner_outer = tg_idx0
mb_inner_outer_ow_inner_outer_fused = tg_idx1
mb_inner_outer = (mb_inner_outer_ow_inner_outer_fused / 16)
ow_inner_outer = ((mb_inner_outer_ow_inner_outer_fused / 1) % 16)
g_outer_od_oh_ow_outer_fused = grid_idx1
g_outer = (g_outer_od_oh_ow_outer_fused / KERNEL_GRID_SIZE) # Kernel grid size is defined in the problem definition and equals OW_GRID_DIMENSION * IMAGE_HEIGHT
od = ((g_outer_od_oh_ow_outer_fused / KERNEL_GRID_SIZE) % 1)
oh = ((g_outer_od_oh_ow_outer_fused / OW_GRID_DIMENSION) % IMAGE_HEIGHT) # OW_GRID_DIMENSION varies per image size
ow_outer = ((g_outer_od_oh_ow_outer_fused / 1) % OW_GRID_DIMENSION)
ic_inner_outer = tg_idx2
``` 
3. Load `src` and `wei` GRF using above offsets into intermediate buffer and store in SLM
4. Put SLM contents into respective `A` and `B` buffers
5. Run `dpas(w)` to calculate the convolution step
```python
dpasw.8x8(c[0], c[0], b[0], a[0]) {Atomic}
dpasw.8x8(c[256], c[256], b[0], a[128])
dpasw.8x8(c[512], c[512], b[256], a[0]) {Atomic}
dpasw.8x8(c[768], c[768], b[256], a[128])
dpasw.8x8(c[1024], c[1024], b[512], a[0]) {Atomic}
dpasw.8x8(c[1280], c[1280], b[512], a[128])
dpasw.8x8(c[1536], c[1536], b[768], a[0]) {Atomic}
dpasw.8x8(c[1792], c[1792], b[768], a[128])
```
This command is invariably the same for most image sizes but indices for `A` might vary for different IC/OC counts. However there are always exactly 8 calls.

6. Optionally reorder `C`
7. Store (reordered) `C` into `dst` using the above offsets


## Optimized Algorithm




## Changing the problem size

When we vary the image size between 64 - 4096 pixels squared the code itself changes very little.
Grid and thread dimensions change, mostly in the `OW_GRID_DIMENSION`.

In the kernel itself, different divisors and modulos are used to determine offsets. However, the load/store operations stay the same unless the size of input is significantly lower (e.g. 64x64).
They don't become more or less in number, only the offsets vary. This suggests that input size differences are mainly managed on the dispatch level.

For uneven sized inputs (e.g. 1080p) the dispatch sizes sometimes (obviously) overshoot the actual input size, leaving some threads idle.
Stores are sometimes broken down into smaller chunks (OWORD4 vs OWORD8). Most likely to maximize performance if alignments can't be met with larger stores.