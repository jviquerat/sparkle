# Generic imports
import os
import sys
import shutil

# Custom imports
from sparkle.src.utils.json       import *
from sparkle.src.utils.data       import *
from sparkle.src.env.environments import *
from sparkle.src.trainer.trainer  import *

###############################################
### Generic runner used in agent and trainer tests
def runner(json_file, agent_type, val_avg, val_bst):

    # Initial space
    print("")

    #########################
    # Initialize json parser and read test json file
    reader = json_parser()
    reader.read(json_file)

    # Initialize trainer
    trainer = trainer_factory.create(reader.pms.trainer.name,
                                     env_pms   = reader.pms.environment,
                                     agent_pms = reader.pms.agent,
                                     path      = ".",
                                     pms       = reader.pms.trainer)

    # Intialize averager
    averager = data_avg(2, trainer.agent.n_steps_total, reader.pms.n_avg)

    # Make two optimization runs and average
    print("Test "+agent_type)
    os.makedirs("0/", exist_ok=True)
    os.makedirs("1/", exist_ok=True)
    trainer.reset(0)
    trainer.optimize()
    averager.store("0/raw.dat", 0)
    trainer.reset(1)
    trainer.optimize()
    averager.store("1/raw.dat", 1)
    trainer.env.close()
    data = averager.average("avg.dat")

    # Check final average and best costs
    avg = data[-1,1]
    bst = data[-1,4]
    print("Avg: "+str(avg))
    print("Bst: "+str(bst))
    assert(avg < val_avg)
    assert(bst < val_bst)

    # Clean
    shutil.rmtree("0")
    shutil.rmtree("1")
    os.remove("avg.dat")
    print("")
