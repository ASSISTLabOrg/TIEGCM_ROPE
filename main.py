"""
Main program file.

Contact: Violet Player
Email: violet.player@noaa.gov
"""

import multiprocessing as mp

def run(x):
    print(x)

if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()

    #-n  NUMPROCS -m MODEL
    parser.add_argument("-n", "--numprocs", default=1, help="Number of processors")
    parser.add_argument("-m", "--model", default="sindy", help="Model type")

    cli_args = parser.parse_args()

    print("Initializing the server with settings: \
          \n\tNumber of processsors: {}\
          \n\tForecaster model     : {}".format(cli_args.numprocs, cli_args.model))

    # #### model setup
    # model = set_up_model(arg)

    #### multiprocessing setup
    mp.set_start_method('forkserver') # ensures model data will be copied over to child processes
    p = mp.Pool(int(cli_args.numprocs))
    p.map_async(run, [i for i in range(20)])
    p.close()
    p.join()