Human zone:

papers cited by Basalt for feature detection and tracking:

FAST corner detector:
https://arxiv.org/abs/0810.2434
KLT tracking: 
https://idl.uw.edu/living-papers-paper/lucas-kanade/
Inverse compositional image alignment:
https://www.ncorr.com/download/publications/bakerequivalence.pdf
Locally Scaled Sum of Squared Differences (LSSD) image alignment norm:
https://web.tecnico.ulisboa.pt/~ist14359/wordpress/nfvr_pubs/wspc02.pdf




Moshi bullshit:
Image pyramids, why do it?
    - Tracking all features in all scales at the same time.
    


---

## Basalt vs Odysseus Frontend Comparison

### Feature Detection

| Aspect | Basalt | Odysseus |
|--------|--------|----------|
| Detector | FAST (via OpenCV) | FAST-9 (custom) |
| Scoring | FAST response | Shi-Tomasi (min eigenvalue) |
| Threshold | Adaptive: starts at 40, falls back to 5 | Fixed: 20 |
| Grid NMS | 50px cells, 1 point per cell | 32px cells, 1 point per cell |
| Edge margin | 19 pixels | 4 pixels |
| Subpixel | No (integer positions) | Yes (quadratic fitting on Shi-Tomasi response) |

**Notes:**
- Basalt's adaptive threshold ensures consistent feature density regardless of texture
- Odysseus uses Shi-Tomasi scoring which directly measures "trackability" for LK
- Could adopt adaptive thresholding from Basalt

### Optical Flow Tracking

| Aspect | Basalt | Odysseus |
|--------|--------|----------|
| Method | Patch-based (Pattern52: ~52 sample points) | Traditional LK (window-based) |
| Window/Patch | ~52 points in cross pattern | 15x15 pixel window |
| Pyramid levels | 3 | 4 |
| Iterations | 5 per level | 30 total |
| Optimization | SE2 (translation + rotation) | Translation only |
| Normalization | Brightness-normalized patches | None |
| Forward-backward check | Yes, threshold 0.2 px | Yes, threshold 2.0 px |

**Basalt optical flow variants** (selected via `optical_flow_type` config):
- `frame_to_frame` (default): Creates patches at finest pyramid level only, tracks coarse-to-fine
- `multiscale_frame_to_frame`: Can create patches at multiple pyramid levels (for features that only appear at coarser scales)
- `patch`: Older implementation, stores patches per-landmark for re-detection

Default config: `optical_flow_type = "frame_to_frame"`, `optical_flow_pattern = 51`

**Notes:**
- Basalt's patch normalization makes it robust to illumination changes
- Basalt optimizes SE2 (includes rotation), Odysseus only translation
- Basalt's FB threshold is much tighter (0.2 vs 2.0 pixels)
- Could add brightness normalization to improve robustness

### Stereo Matching

| Aspect | Basalt | Odysseus |
|--------|--------|----------|
| Method | Optical flow with depth-based initial guess | Descriptor matching (BRIEF) |
| Epipolar check | Essential matrix constraint (0.005 threshold) | Vertical diff < 2px |
| Depth estimation | Projects using average depth for initial guess | Disparity from matched position |
| Ratio test | N/A (uses flow) | 0.7 ratio test |
| Descriptor | N/A | BRIEF (256-bit) |

**Notes:**
- Basalt treats stereo as another optical flow problem with good initialization
- Odysseus uses traditional descriptor matching (more robust to large baselines?)
- Basalt's approach is elegant but requires good depth estimate

### Basalt Feature Recall System

Basalt has an optional "landmark recall" feature that can recover tracks that were temporarily lost. Here's how it works:

**Setup (when a new feature is detected):**
1. When a feature is first detected, save its normalized patch at each pyramid level
2. Store these patches in a map keyed by landmark ID: `patches[landmark_id] = [patch_level_0, patch_level_1, ...]`

**Recall (each frame, after normal tracking):**
1. Get 3D positions of all landmarks from the backend (`latest_lm_bundle`)
2. Project each landmark into the current camera frame using current pose estimate
3. For landmarks not currently tracked:
   - Use the stored patch as a template
   - Track from the projected position using the saved patch (not frame-to-frame)
   - If patch matching succeeds AND position is close to projection → recovered!
4. Optionally update the patch template to the new viewpoint

**Key parameters:**
- `optical_flow_recall_enable`: Enable/disable recall (default: false)
- `optical_flow_recall_max_patch_dist`: Max distance between projected and tracked position (% of image width)
- `optical_flow_recall_max_patch_norms`: Max patch matching error per pyramid level
- `optical_flow_recall_update_patch_viewpoint`: Update template after successful recall

**Why it helps:**
- Features can be re-acquired after temporary occlusion
- Tracks survive motion blur spikes
- Longer effective track lifetimes
- Better loop closure potential (old landmarks become visible again)

**Tradeoffs:**
- Memory: patches are never deleted (noted as TODO in Basalt)
- CPU: extra work to project and match all stored patches
- Requires 3D landmark positions from backend (tight coupling)

### Key Techniques in Basalt We Could Adopt

1. **Adaptive detection threshold** - Start high, reduce if too few features
2. **Brightness-normalized patches** - Divide patch by mean for illumination invariance
3. **Tighter FB threshold** - Basalt uses 0.2px, we use 2.0px
4. **SE2 optimization** - Track rotation along with translation
5. **Pattern-based sampling** - 52 points in cross pattern vs dense window (faster)
6. **Landmark recall** - Re-detect old landmarks that reappear (requires backend integration)

### Current Odysseus Strengths

1. **Shi-Tomasi scoring** - Directly measures LK trackability
2. **Subpixel detection** - Quadratic fitting for precise corner localization
3. **Descriptor-based stereo** - May be more robust for large stereo baselines
4. **Simpler implementation** - Easier to understand and modify

---

## Implementation Details

### Image Pyramids

Why use pyramids?
- Handle large motions that exceed the search window at full resolution
- Coarse-to-fine: find approximate motion at coarse level, refine at fine level
- Basalt: 3 levels, Odysseus: 4 levels
- Scale factor: 2x between levels

### Forward-Backward Consistency Check

The key insight: if tracking is correct, tracking A→B→A should return to A.

```
1. Track point from frame 1 to frame 2: p1 → p2
2. Track back from frame 2 to frame 1: p2 → p1'
3. Compute error: |p1 - p1'|
4. Reject if error > threshold
```

This catches:
- Points that land on edges (aperture problem)
- Points near occlusion boundaries
- Tracking failures due to motion blur
- Ambiguous matches

### Shi-Tomasi Response

The minimum eigenvalue of the structure tensor M = [[Ix², IxIy], [IxIy, Iy²]]:

```
λ_min = (trace - sqrt(trace² - 4*det)) / 2
```

- High λ_min = corner (good for tracking)
- Low λ_min with high λ_max = edge (aperture problem)
- Both low = flat region (nothing to track)

Detection uses threshold 10.0, tracking landing check uses 10.0.

---