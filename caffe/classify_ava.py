#!/usr/bin/env python
"""
Trains a model using one or more GPUs.
"""
from multiprocessing import Process

import caffe
import signal
import sys
import traceback
import time

from google.protobuf import text_format
from caffe.proto import caffe_pb2

from ava.train import base as train
from ava.log.logger import logger
from ava.params import params
from ava.utils import utils
from ava.utils import cmd
from ava.utils import config
from ava.monitor import caffe as caffe_monitor


def signal_handler(signum, frame):
    logger.info("received signal: %s, do clean_up", signum)
    train_ins = frame.f_locals['train_ins']
    clean_up(train_ins)
    exit()


def start_new_training():
    # AVA-SDK training Instance
    SUPPORTED_SIGNALS = (signal.SIGINT, signal.SIGTERM,)
    for signum in SUPPORTED_SIGNALS:
        try:
            signal.signal(signum, signal_handler)
            logger.info("Bind signal '%s' success to %s",
                        signum, signal_handler)
        except Exception as identifier:
            logger.warning(
                "Bind signal '%s' failed, err: %s", signum, identifier)

    try:
        # parse args
        train_ins = train.TrainInstance()
        err_msg=''
        # add CALLBACK
        solver_param = caffe_pb2.SolverParameter()
        with open('/workspace/model/lenet_solver.prototxt','r') as f:
            text_format.Merge(f.read(),solver_param)
        solver_param.snapshot_prefix = train_ins.get_snapshot_base_path()
        logger.info("saving to  %s", solver_param.snapshot_prefix)

        fixed_solver = train_ins.get_base_path() + "/solver.prototxt"
        with open(fixed_solver, 'w') as f:
            f.write(str(solver_param))
        logger.info("write fixed solver to %s", fixed_solver)

        # AVA-SDK start caffe process
        training_cmd = ['caffe','train','-solver',fixed_solver,'-gpu','0']
        proc = cmd.startproc(training_cmd)
        logger.info("Started %s", proc)
        # AVA-SDK add caffe callback
        cmd.logproc(proc, [train_ins.get_monitor_callback("caffe")])
        exit_code = proc.wait()
        logger.info("Finished proc with code %s", exit_code)
        logger.info("Gracefully shutdown after 5s, wait cleaner ...")
        time.sleep(5)
        logger.info("Done.")
        if exit_code != 0:
            logger.error(
                "training exit code [%d] != 0, raise Exception", exit_code)
            raise Exception("training exit code [%d] != 0" % (exit_code))
        train_ins.done(err_msg=err_msg)
    except Exception as err:
        err_msg = "training failed, err: %s" % (err)
        logger.info(err_msg)
        traceback.print_exc(file=sys.stderr)

        if train_ins == None:
            return
        train_ins.done(err_msg=err_msg)





def main():
    start_new_training()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

#    parser.add_argument("--solver", required=True, help="Solver proto definition.")
#    parser.add_argument("--snapshot", help="Solver snapshot to restore.")
#    parser.add_argument("--gpus", type=int, nargs='+', default=[0],
#                        help="List of device ids.")
#    parser.add_argument("--timing", action='store_true', help="Show timing info.")
    args = parser.parse_args()

    main()
 #   train(args.solver, args.snapshot, args.gpus, args.timing)