# Data placement:

 Detection dataset is placed in the following structure:
- data/train/good/

- data/test/good/
- data/test/color/
- data/test/cut/
- data/test/hole/
- data/test/metal_contamination/
- data/test/thread/

- data/ground_truth/color/
- data/ground_truth/cut/
- data/ground_truth/hole/
- data/ground_truth/metal_contamination/
- data/ground_truth/thread/

Expected training and evaluation commands in this repository use --data_path data/.
