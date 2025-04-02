import os
import shutil

from sparkle.src.env.parallel import parallel
from sparkle.src.trainer.trainer import trainer_factory
from sparkle.src.utils.compare import compare
from sparkle.src.utils.data import DataAvg
from sparkle.src.utils.json import JsonParser
from sparkle.src.utils.seeds import set_seeds


###############################################
### Generic runner used in agent and trainer tests
def runner(json_file, val_avg, val_bst):

    # Set seed for reproducible test
    set_seeds(0)

    # Initial space
    print("")

    # Initialize json parser and read test json file
    reader = JsonParser()
    reader.read(json_file)

    # Initialize parallel framework
    parallel.set(reader.pms)

    # Initialize trainer
    trainer = trainer_factory.create(reader.pms.trainer.name,
                                     path = ".",
                                     pms  = reader.pms.trainer)

    # Intialize averager
    averager = DataAvg(2, reader.pms.n_avg)

    # Make two optimization runs and average
    print("Test "+reader.pms.trainer.agent.name)
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
    print("Avg: {:.15e}".format(avg))
    print("Bst: {:.15e}".format(bst))
    assert(compare(avg, val_avg, 1.0e-15))
    assert(compare(bst, val_bst, 1.0e-15))

    # Clean
    shutil.rmtree("0")
    shutil.rmtree("1")
    os.remove("avg.dat")
    print("")
