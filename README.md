# Gradient Boosting Decision Tree for Learning to Rank

Course Project for How To Write Fast Numerical Code, ETH Spring 2019

## Repository Structure

```bash
├── data 				# scripts to download data
├── include/			# header files
├── src/				# source code
├── docs/				# project-related documents generated along the course.
│	├── Report.pdf
│	└── Presentation.pdf
├── main.cpp			# starts the program
└── README.md
```
## Building the Source Code

```bash
mkdir build && cd build
cmake .. && make
ln -s ../data .
```

## Implementation branches

##### `master`
- The baseline implementation, that makes use of no optimization.
```python
for feature in features:
	for sample in samples:
		update()
	for candidate in candidates:
		cumulate()
		get_best_split()
```

#####  `feature_blocking_cumulator`
- Perform loop over all combinations of blocking & unrolling for `update` & `cumulate`, which are implemented using AVX.
- Allows for variable re-use for candidate in `update`, resulting in reduction of cost incurred due to branching.
```python
for feat_block in features:
	for sample_block in samples:
		for sample in sample_block:
			for feature in feat_block:
				update_using_avx()

	for feature in feat_block:
		for cand_block in candidates:
			for bin in cand_block:
				cumulate_using_avx()
	
	for candidate in candidates:
		for feature in feat_block:
			get_best_split() 
```

#####  `feature_blocking_cumulator_no_avx`
- Perform loop over all combinations of blocking & unrolling for `update` & `cumulate`
- Allows for variable re-use for candidate in `update`, resulting in reduction of cost incurred due to branching.
```python
for feat_block in features:
	for sample_block in samples:
		for sample in sample_block:
			for feature in feat_block:
				update()

	for feature in feat_block:
		for cand_block in candidates:
			for bin in cand_block:
				cumulate()
	
	for candidate in candidates:
		for feature in feat_block:
			get_best_split() 
```

#####  `feature_blocking_cumulator_float`
- Similar to `feature_blocking_cumulator`.
- Replace double by float.
- Change intrinsics accordingly.

#####  `feature_blocking_cumulator_float`
- Similar`feature_blocking_cumulator_no_avx`.
- Replace double by float.

#####  `sparse_bins`
- Algorithmic improvement.
- Reduces dataset to dense representation.