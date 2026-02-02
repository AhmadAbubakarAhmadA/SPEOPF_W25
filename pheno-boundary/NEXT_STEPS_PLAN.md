# Pheno-Boundary Project: Next Steps Plan

**Date:** 2026-01-12
**Status:** Phase 3 Complete ✅ | Moving to Phase 4 & 5

---

## 🎯 Project Overview

**Mission:** Multi-year agricultural field boundary detection and temporal stability analysis using Sentinel-2 and FTW neural network.

**Current Status:**
- ✅ **Phase 1 Complete:** STAC access, datacube formation, model download
- ✅ **Phase 2 Complete:** Cloud masking, seasonal composites, preprocessing
- ✅ **Phase 3 Complete:** FTW inference with corrected band order (50-55% field coverage)
- 🔄 **Phase 4 Pending:** Stability analysis (IoU, change detection)
- 🔄 **Phase 5 Pending:** Advanced visualizations

---

## 📊 Current Results Summary

### Inference Results (FIXED):
```
2020: 52.5% field coverage  (mean prob: 0.524)
2021: 55.0% field coverage  (mean prob: 0.550) ⭐ Best
2022: 55.0% field coverage  (mean prob: 0.549) ⭐ Best
2023: 51.5% field coverage  (mean prob: 0.515)
```

**Key Achievement:** Band order fix increased detection from 0-13% → 50-55%! 🎉

---

## 🗺️ Next Steps Roadmap

---

## **PHASE 1: Critical Infrastructure Fixes**

### **Task 1.1: Fix Main Pipeline Notebook**
**Priority:** 🔴 **CRITICAL** (Blocks all future work)
**Importance:** ⭐⭐⭐⭐⭐ (5/5)
**Time Estimate:** 30 minutes

**What to do:**
- Update `pheno_boundary_full_pipeline.ipynb` cell 28
- Change band order from `['b02', 'b03', 'b04', 'b08']` → `['b04', 'b03', 'b02', 'b08']`
- Re-run cells 28-29 to regenerate correct `ftw_inputs.pkl`
- Upload new `ftw_inputs.pkl` to Google Drive

**Why it's critical:**
- Current saved inputs have wrong band order
- All future analysis depends on correct inputs
- Quick win that prevents future errors

**Deliverable:**
- ✅ Corrected main pipeline notebook
- ✅ New `ftw_inputs.pkl` with correct [R,G,B,NIR] order

---

### **Task 1.2: Verify Corrected Results**
**Priority:** 🔴 **CRITICAL**
**Importance:** ⭐⭐⭐⭐⭐ (5/5)
**Time Estimate:** 15 minutes

**What to do:**
- Re-run inference with newly generated inputs
- Verify field coverage remains 50-55%
- Save `ftw_results_VERIFIED.pkl`

**Success Criteria:**
- Consistent 50-55% field coverage across all years
- High confidence predictions (mean prob > 0.5)

---

## **PHASE 2: Post-Processing & Smoothing**

### **Task 2.1: Implement Morphological Smoothing**
**Priority:** 🟡 **HIGH**
**Importance:** ⭐⭐⭐⭐ (4/5)
**Time Estimate:** 2 hours

**What to do:**
1. Add morphological operations (opening, closing, hole-filling)
2. Remove small objects (< 100 pixels)
3. Apply Gaussian smoothing to boundaries
4. Test different kernel sizes (3, 5, 7 pixels)

**Why it's important:**
- Removes noise and artifacts
- Creates professional-looking boundaries
- Makes results publication-ready
- Required for accurate IoU calculations

**Deliverable:**
- `smooth_field_mask()` function added to pipeline
- Smoothed masks for all 4 years

---

### **Task 2.2: Vectorization with Simplification**
**Priority:** 🟡 **HIGH**
**Importance:** ⭐⭐⭐⭐ (4/5)
**Time Estimate:** 2 hours

**What to do:**
1. Convert raster masks to vector polygons using `rasterio.features.shapes()`
2. Apply Douglas-Peucker simplification (tolerance: 10-20m)
3. Apply Chaikin's smoothing algorithm (2-3 iterations)
4. Export as GeoJSON/Shapefile for GIS

**Why it's important:**
- Enables GIS analysis and visualization
- Reduces file size while maintaining accuracy
- Standard format for sharing results
- Required for overlay with ground truth

**Deliverable:**
- Vector polygons for each year (GeoJSON format)
- `parcels_2020.geojson`, `parcels_2021.geojson`, etc.

---

### **Task 2.3: Visual Quality Assessment**
**Priority:** 🟢 **MEDIUM**
**Importance:** ⭐⭐⭐ (3/5)
**Time Estimate:** 1 hour

**What to do:**
- Create side-by-side comparison: raw mask vs smoothed vs vector
- Overlay on RGB composites
- Validate boundary quality visually

**Deliverable:**
- Quality assessment figures for all years

---

## **PHASE 3: Temporal Stability Analysis** ⭐ **CORE RESEARCH CONTRIBUTION**

### **Task 3.1: Compute Pairwise IoU Matrix**
**Priority:** 🔴 **CRITICAL**
**Importance:** ⭐⭐⭐⭐⭐ (5/5)
**Time Estimate:** 3 hours

**What to do:**
1. Compute Intersection over Union (IoU) between all year pairs
2. Create 4×4 IoU matrix
3. Identify stable vs changing parcels
4. Calculate per-pixel stability scores

**Formula:**
```
IoU(A, B) = |A ∩ B| / |A ∪ B|

Stability_Score(pixel) = mean(IoU across all year pairs)
```

**Why it's critical:**
- **This is the main research question!**
- Quantifies temporal stability of field boundaries
- Identifies which parcels are permanent vs temporary
- Key metric for agricultural monitoring

**Deliverable:**
- IoU matrix (CSV format)
- Stability score raster for entire study area
- Statistics: mean IoU, median IoU, stability distribution

---

### **Task 3.2: Change Detection Analysis**
**Priority:** 🟡 **HIGH**
**Importance:** ⭐⭐⭐⭐ (4/5)
**Time Estimate:** 2 hours

**What to do:**
1. Detect boundary changes between consecutive years
2. Classify changes:
   - **Expansion:** New field areas added
   - **Contraction:** Field areas removed
   - **Fragmentation:** Large parcel split into smaller
   - **Consolidation:** Small parcels merged
3. Quantify change magnitude (area in hectares)

**Analysis:**
```python
change_2020_2021 = mask_2021 - mask_2020
# Positive = expansion, Negative = contraction

change_area = np.sum(np.abs(change_2020_2021)) * pixel_area_m2
```

**Why it's important:**
- Identifies land use dynamics
- Detects agricultural intensification/abandonment
- Validates model consistency

**Deliverable:**
- Change detection maps for each year transition
- Change statistics table (expansion/contraction per year)

---

### **Task 3.3: Stability Zones Classification**
**Priority:** 🟢 **MEDIUM**
**Importance:** ⭐⭐⭐⭐ (4/5)
**Time Estimate:** 2 hours

**What to do:**
Classify study area into stability zones:
- **Highly Stable (IoU > 0.8):** Permanent field boundaries
- **Moderately Stable (0.5 < IoU < 0.8):** Seasonal changes
- **Unstable (IoU < 0.5):** Frequent boundary changes
- **New/Abandoned (IoU = 0):** Land use change

**Why it's important:**
- Identifies reliable vs unreliable boundaries
- Informs agricultural policy
- Validates FTW model consistency

**Deliverable:**
- Stability zone classification map
- Statistics on % area in each category

---

## **PHASE 4: Advanced Visualization**

### **Task 4.1: Multi-Year Comparison Panels**
**Priority:** 🟡 **HIGH**
**Importance:** ⭐⭐⭐⭐ (4/5)
**Time Estimate:** 2 hours

**What to do:**
- Create publication-quality 4-panel figure (one per year)
- Show: RGB composite + field boundaries overlay
- Add colorbar, north arrow, scale bar
- Export as high-res PNG/PDF

**Deliverable:**
- `Figure_1_MultiYear_FieldBoundaries.pdf`

---

### **Task 4.2: Stability Heatmap**
**Priority:** 🟡 **HIGH**
**Importance:** ⭐⭐⭐⭐⭐ (5/5)
**Time Estimate:** 2 hours

**What to do:**
- Create heatmap showing stability scores across study area
- Use diverging colormap (red = unstable, green = stable)
- Overlay with major parcels for context

**Why it's critical:**
- **Primary visualization of research findings**
- Shows spatial patterns of stability
- Publication-ready figure

**Deliverable:**
- `Figure_2_TemporalStability_Heatmap.pdf`

---

### **Task 4.3: Change Detection Animations**
**Priority:** 🟢 **MEDIUM**
**Importance:** ⭐⭐⭐ (3/5)
**Time Estimate:** 3 hours

**What to do:**
- Create animated GIF showing field evolution 2020→2023
- Highlight areas of change in red/yellow
- Add year label and statistics overlay

**Deliverable:**
- `Animation_FieldBoundaries_2020-2023.gif`

---

### **Task 4.4: Statistical Summary Dashboard**
**Priority:** 🟢 **MEDIUM**
**Importance:** ⭐⭐⭐ (3/5)
**Time Estimate:** 2 hours

**What to do:**
Create dashboard showing:
- Total field area per year (hectares)
- Number of parcels detected
- Mean parcel size
- IoU matrix visualization
- Change statistics bar charts

**Deliverable:**
- `Figure_3_Statistical_Summary.pdf`

---

## **PHASE 5: Documentation & Reporting**

### **Task 5.1: Results Summary Document**
**Priority:** 🟡 **HIGH**
**Importance:** ⭐⭐⭐⭐ (4/5)
**Time Estimate:** 3 hours

**What to do:**
Write comprehensive results document:
1. **Introduction:** Study area, methods, objectives
2. **Data & Methods:** Sentinel-2, FTW model, preprocessing
3. **Results:**
   - Field detection accuracy (50-55% coverage)
   - Temporal stability analysis
   - Change detection findings
4. **Discussion:** Interpretation, limitations, future work
5. **Conclusions:** Key findings and implications

**Deliverable:**
- `RESULTS_REPORT.md` or `.pdf` (5-10 pages)

---

### **Task 5.2: Code Documentation**
**Priority:** 🟢 **MEDIUM**
**Importance:** ⭐⭐⭐ (3/5)
**Time Estimate:** 2 hours

**What to do:**
- Add docstrings to all functions
- Create `USAGE_GUIDE.md` for reproducing results
- Document dependencies and environment setup

**Deliverable:**
- Well-documented, reproducible code

---

### **Task 5.3: Update README**
**Priority:** 🟢 **MEDIUM**
**Importance:** ⭐⭐⭐ (3/5)
**Time Estimate:** 1 hour

**What to do:**
- Update README with final results
- Add figures and key findings
- Document known issues and fixes (band order!)

**Deliverable:**
- Updated `README.md`

---

## 📅 Recommended Timeline

### **Week 1: Critical Infrastructure (3-4 hours)**
- ✅ Task 1.1: Fix main pipeline notebook
- ✅ Task 1.2: Verify corrected results

### **Week 2: Post-Processing (5-7 hours)**
- Task 2.1: Morphological smoothing
- Task 2.2: Vectorization
- Task 2.3: Quality assessment

### **Week 3: Stability Analysis (7-9 hours)** ⭐ **CORE RESEARCH**
- Task 3.1: IoU matrix computation
- Task 3.2: Change detection
- Task 3.3: Stability zones

### **Week 4: Visualization & Reporting (9-12 hours)**
- Task 4.1-4.4: Create all figures
- Task 5.1-5.3: Write documentation

**Total Estimated Time:** 24-32 hours

---

## 🎯 Prioritized Action Plan (If Time Limited)

### **Minimum Viable Product (8 hours):**
1. Fix main pipeline (30 min) 🔴
2. Compute IoU matrix (3 hours) 🔴
3. Create stability heatmap (2 hours) 🔴
4. Write 2-page results summary (2.5 hours) 🔴

### **Standard Deliverable (16 hours):** MVP +
5. Morphological smoothing (2 hours)
6. Change detection analysis (2 hours)
7. Multi-year comparison figure (2 hours)
8. Statistical dashboard (2 hours)

### **Publication-Ready (32 hours):** Standard +
9. Vectorization & GeoJSON export (2 hours)
10. Change animations (3 hours)
11. Full results report (5 pages) (3 hours)
12. Code documentation (2 hours)
13. Quality figures and polishing (5 hours)

---

## 🚨 Critical Dependencies

**Before starting Phase 3-5, you MUST:**
1. ✅ Fix band order in main pipeline
2. ✅ Regenerate `ftw_inputs.pkl` with correct order
3. ✅ Verify inference results are stable (50-55% coverage)

**Blocker:** If you don't fix the main pipeline, all future work will be based on incorrect inputs!

---

## 📊 Success Metrics

### **Scientific Success:**
- [ ] IoU matrix computed for all year pairs
- [ ] Temporal stability quantified with statistical significance
- [ ] Change detection identifies real land use dynamics
- [ ] Results reproducible by others

### **Technical Success:**
- [ ] 50-55% field coverage maintained across all years
- [ ] Smooth, publication-quality boundaries
- [ ] Vector outputs compatible with GIS software
- [ ] Code documented and reproducible

### **Impact Success:**
- [ ] Results presented in clear, compelling visualizations
- [ ] Findings documented in comprehensive report
- [ ] Repository ready for sharing/publication
- [ ] Methods reusable for other regions

---

## 💡 Recommendations

### **Priority Ranking:**
1. **🔴 CRITICAL:** Tasks 1.1, 1.2, 3.1, 4.2 (Infrastructure + Core Analysis)
2. **🟡 HIGH:** Tasks 2.1, 2.2, 3.2, 4.1, 5.1 (Quality + Deliverables)
3. **🟢 MEDIUM:** Everything else (Polish + Documentation)

### **Start Here:**
1. Fix main pipeline notebook (Task 1.1) - **DO THIS FIRST!**
2. Implement IoU computation (Task 3.1) - Core research question
3. Create stability heatmap (Task 4.2) - Show your findings!

### **Quick Wins:**
- Task 1.1: 30 minutes, huge impact
- Task 2.1: 2 hours, professional results
- Task 3.1: 3 hours, answers main research question

---

## 📝 Notes

**What's Working Well:**
- FTW model performs excellently after band order fix
- Consistent 50-55% field coverage is realistic and stable
- High confidence predictions (mean prob > 0.5)
- GPU inference on Colab works smoothly

**Known Issues:**
- Original band order was incorrect (FIXED! ✅)
- Need transform/CRS info for vectorization (available in datacube)
- IoU computation may be memory-intensive for full resolution

**Lessons Learned:**
- Always verify input band order against model expectations
- Use strict=True for model loading to catch issues early
- Debug with small test cases before full inference

---

## 🤝 Questions to Discuss

1. **Scope:** Do you want minimal (8 hrs) or full (32 hrs) implementation?
2. **Priority:** Is stability analysis (Phase 3) or smooth boundaries (Phase 2) more urgent?
3. **Output Format:** Do you need vector outputs (GeoJSON) or raster is sufficient?
4. **Timeline:** Do you have a deadline for results?
5. **Validation:** Do you have ground truth data to validate against?

---

**Next Action:** Shall we start with Task 1.1 (fixing the main pipeline notebook)?
