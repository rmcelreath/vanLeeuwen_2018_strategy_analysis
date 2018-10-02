# vanLeeuwen et al 2018 reanalysis

This repository contains files for the strategy-based reanalysis of van Leeuwen et al (2018) "The development of human social learning across seven societies" (DOI: 10.1038/s41467-018-04468-2). This is a really nice paper that models proportions of behavior. The reanalysis here models instead proportions of strategies. The key difference is that different strategies can predict the same behavior in some cases.

The `script_trinomial_model.R` prepares the data and runs the models, as well as produces summary plots.

These models consider six learning strategies:
(1) Majority-rule conformity: Copy the majority choice
(2) Minority-rule: Copy the minority choice
(3) Unbiased copying: Copy in proportion to number of individuals demonstrating
(4) Maverick: Choose the undemonstrated option
(5) Random: Choose at random
(6) Copy first demonstrated option

