import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import product

from pyitab.utils.matrix import copy_matrix, array_to_matrix
from pyitab.analysis.results.bids import filter_dataframe, get_results_bids
from pyitab.analysis.results.dataframe import apply_function, query_rows
from pyitab.plot.connectivity import plot_connectivity_circle_edited, plot_connectivity_lines

pl.style.use("seaborn")

path = "/media/robbis/Seagate_Pt1/data/working_memory/data/derivatives/"


pipeline="psi+review+singleband" # added sample ztransforming when loading
pipeline="psi+review+singleband+plain" # without sample ztransforming and k=1:1200:50

pipeline="multiband+channel+review"

pipeline = "PSICORRmultiband+channel+review"
pipeline = "PSICORRpsi+review+singleband+plain"

full_df = get_results_bids(path, pipeline=pipeline, field_list=['estimator__fsel', 
                                                                    'ds.a.task', 
                                                                    'ds.a.prepro',
                                                                    'ds.a.img_pattern',
                                                                    'sample_slicer'])

f = sns.relplot(x="k", y="score_score", col="band", hue="targets", row='ds.a.task', 
              height=5, aspect=.75, facet_kws=dict(sharex=False),
              kind="line", legend="full", data=full_df 
              )

