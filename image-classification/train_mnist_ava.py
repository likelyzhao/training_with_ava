# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx
import signal
from ava.log.logger import logger
from ava.train import base as train
import traceback
import sys



def clean_up(train_ins, err_msg=""):
    # AVA-SDK
    if train_ins == None:
        return
    train_ins.done(err_msg=err_msg)


def signal_handler(signum, frame):
    logger.info("received signal: %s, do clean_up", signum)
    train_ins = frame.f_locals['train_ins']
    clean_up(train_ins)
    exit()

def start_new_training():
    # binding signals
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
        parser = argparse.ArgumentParser(description="train imagenet-1k",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        fit.add_fit_args(parser)
        data.add_data_args(parser)
        data.add_data_aug_args(parser)
        # use a large aug level
        data.set_data_aug_level(parser, 3)
        parser.set_defaults(
            # network
            network='resnet',
            num_layers=50,
            # data
            num_classes=10,
            num_examples=60000,
            image_shape='3,28,28',
            min_random_scale=1,  # if input image has min size k, suggest to use
            # 256.0/x, e.g. 0.533 for 480
            # train
            num_epochs=80,
            lr_step_epochs='30,60',
            dtype='float32',
            batch_size =32
        )
        args = parser.parse_args()

        # AVA-SDK  new an Instance
        train_ins = train.TrainInstance()
        # add CALLBACK
        batch_end_cb = train_ins.get_monitor_callback(
            "mxnet",
            batch_size=args.batch_size,
            batch_freq=10)
        args.batch_end_callback = batch_end_cb

        # load network
        from importlib import import_module
        net = import_module('symbols.' + args.network)
        sym = net.get_symbol(**vars(args))

        # train
        fit.fit(args, sym, data.get_rec_iter)

        logger.info("training finish")
        err_msg = ""
        if train_ins == None:
            return
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
    main()
