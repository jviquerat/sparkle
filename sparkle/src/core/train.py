import os
import sys
import time
import shutil

from sparkle.src.env.parallel    import parallel
from sparkle.src.trainer.trainer import trainer_factory
from sparkle.src.utils.data      import DataAvg
from sparkle.src.plot.plot       import plot_avg
from sparkle.src.utils.prints    import liner

# Average training over multiple runs
def train(json_file, pms):

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

    # Initialize trainer
    trainer = trainer_factory.create(pms.trainer.name,
                                     path      = results_path,
                                     pms       = pms.trainer)

    # Intialize averager
    averager = DataAvg(2, pms.n_avg)

    # Run
    for run in range(pms.n_avg):
        liner('Avg run #'+str(run))
        os.makedirs(results_path+'/'+str(run), exist_ok=True)
        trainer.reset(run)
        trainer.optimize()
        filename = results_path+'/'+str(run)+'/raw.dat'
        averager.store(filename, run)

    # Close environments
    trainer.env.close()

    # Write to file
    filename = results_path+'/avg.dat'
    data = averager.average(filename, pms.avg_type)

    # Plot
    filename = results_path+'/'+path
    plot_avg(data, filename, pms.avg_type)

    # Finalize parallel process
    parallel.finalize()

# Generate results folder name
def folder_name(pms):

    name_env = ""
    if hasattr(pms.naming, "env"):
        if (pms.naming.env is True):
            name_env = pms.trainer.environment.name

    name_agent = ""
    if hasattr(pms.naming, "agent"):
        if (pms.naming.agent is True):
            name_agent = pms.trainer.agent.name

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
    if (name_agent != ""):
        if (path != ""): path += "_"
        path += name_agent
    if (name_tag != ""):
        if (path != ""): path += "_"
        path += name_tag
    if (name_time):
        if (path != ""): path += "_"
        path += name_time

    return path
