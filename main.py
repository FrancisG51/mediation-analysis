mediation_analysis_process = False
bootstrap_need = False
use_llm = False

from data_preprocess import *
process_raw_data_no_brain()

from have_brain import *
have_brain_data()

from have_data_all import *
have_data_all_function()

from statistical_analysis_log import *
statistical_function_logistic()

from mediation_analysis import *
if mediation_analysis_process == True:
    mediation_analysis()

from bootstrap_test import *
if bootstrap_need == True:
    bootstrap_function()

process_bootstrap_result()

from spss_validation import *
spss_valid()

from llm_app import *
if use_llm == True:
    use_llm_function()
