import signal
import sys
import traceback
import time

from ava.train import base as train
from ava.log.logger import logger
from ava.params import params
from ava.utils import utils
from ava.utils import cmd
from ava.utils import config
from ava.monitor import caffe as caffe_monitor




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

        # AVA-SDK start caffe process
        out_dir = train_ins.get_snapshot_base_path()
        roidb_path = train_ins.get_trainset_base_path() + "/cache/gt_roidb.pkl"
        training_cmd = ['python', 'detect_py_faster_rcnn.py', '--solver', 'vgg_solver.prototxt', '--gpu', '0',
                        '--output_path', out_dir, '--ava_roidb_path', roidb_path,
                        '--train_base_path', train_ins.get_trainset_base_path()+'/cache']
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

if __name__ == '__main__':
    start_new_training()