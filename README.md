**Date**: April 15nd, 2022

**Title**: Code for Paper: [A KNOWLEDGE/DATA ENHANCED METHOD FOR JOINT EVENT AND TEMPORAL RELATION EXTRACTION]

The TB-Dense and MATRES raw data files are saved in data fold. Download any other necessary files into other/ folder.

1. Before the experiment, modify the configuration file to your own path

2. Data processinng. run python addcommonsense_featurized_data_all_*.py first,the folder  all_joint/ are created

3. Featurize data. run python context_aggregator_*.py. The folder all_context/ are created: . 

the step 2 all_context contains the final files used in the model.

4. Local Model: run python joint_model_addcommonseStandModel_roberta_*.py


For convenience and ease of understanding,You can run it separately on both datasets.