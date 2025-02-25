# Generic imports
import os
import sys
import time
import shutil
import numpy as np

# Custom imports
from sparkle.src.env.parallel    import parallel
from sparkle.src.model.generator import generate
from sparkle.src.plot.plot       import plot_avg
from sparkle.src.utils.prints    import liner

# Compute model
def model(json_file, pms):

    # Add paths to PATH
    base_path = os.path.abspath(os.getcwd())
    json_path = os.path.dirname(json_file)
    sys.path.append(base_path)
    sys.path.append(json_path)

    # Create paths for results and open repositories
    base_path     = os.path.abspath(os.getcwd())
    results_path  = 'results'
    os.makedirs(results_path, exist_ok=True)
    path          = folder_name(pms)
    results_path += '/'+path
    os.makedirs(results_path, exist_ok=True)

    # Copy json file to results folder
    shutil.copyfile(json_file, results_path+'/params.json')

    # Generate model
    model = generate(env_pms   = pms.environment,
                     pex_pms   = pms.pex,
                     model_pms = pms.model,
                     path      = results_path)


    # Finalize parallel process
    parallel.finalize()

# Generate results folder name
def folder_name(pms):

    name_env = ""
    if hasattr(pms.naming, "env"):
        if (pms.naming.env is True):
            name_env = pms.environment.name

    name_model = ""
    if hasattr(pms.naming, "model"):
        if (pms.naming.model is True):
            name_model = pms.model.name

    name_tag = ""
    if hasattr(pms.naming, "tag"):
        if (pms.naming.tag is not False):
            name_tag = pms.naming.tag

    name_time = ""
    if hasattr(pms.naming, "time"):
        if (pms.naming.time is True):
            name_time = str(time.strftime("%H-%M-%S", time.localtime()))

    path = ""
    if (name_env != ""):
        path += name_env
    if (name_model != ""):
        if (path != ""): path += "_"
        path += name_model
    if (name_tag != ""):
        if (path != ""): path += "_"
        path += name_tag
    if (name_time):
        if (path != ""): path += "_"
        path += name_time

    return path
