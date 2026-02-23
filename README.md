# DXA_scan_BCBA
This is the official code for Body Composition Biological Age – A deep learning sex-specific body composition ageing biomarker using DXA scan

Note on age reweighting:
Due to imbalance in the age distribution of the reference cohort, age-stratified weighting was applied during model training. Samples were grouped into age bins (5/10/15-year intervals, or under xx-age/over xx-age), and training weights were adjusted to reduce the influence of overrepresented age ranges. The binning strategy and weighting scheme should be determined based on the age distribution of the dataset under analysis. Morever, fineyuning can also improve this. Besdeis,regression-based age-bias correction can also solve (e.g., correcting regression-to-the-mean effects by regressing predicted age on chronological age and using residuals as age gap). These strategies may be considered depending on study design and analytical objectives.
