# MAR History Note

- The rows below are imported from the original `MAR_test/backup_2600.py` acceptance outputs.
- They are not recomputed by the new benchmark code path; they are the stored full-pipeline historical results.

## MAR-engineered-full

- Raw thermal GT: pixel_accuracy=0.998071, mean_iou=0.991410, binary_iou_foreground=0.877151
- Aligned GT: pixel_accuracy=0.992318, mean_iou=0.967495, binary_iou_foreground=0.739261
- geometry_baseline_ok=True, real_projected_ratio_on_redistort=0.914334

## MAR-paper-full

- Raw thermal GT: pixel_accuracy=0.996753, mean_iou=0.976560, binary_iou_foreground=0.805773
- Aligned GT: pixel_accuracy=0.993581, mean_iou=0.996271, binary_iou_foreground=0.769357
- geometry_baseline_ok=False, real_projected_ratio_on_redistort=0.161953

## Manual Binary Reference

- MAR-engineered-history: pixel_accuracy_total=0.987869, foreground_iou=0.427893
- MAR-paper-history: pixel_accuracy_total=0.987515, foreground_iou=0.434711

- These manual binary numbers are a different metric definition from the multi-class benchmark table.
