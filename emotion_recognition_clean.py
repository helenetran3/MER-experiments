#!/usr/bin/env python
################################################################################
#       The Edinburgh G25 Multimodal DNN Emotion Recognition System
#                https://github.com/rhoposit/emotionChallenge
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2017-2018
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################


def custom_split(train, valid):
    valid = list(valid)
    train = list(train)
    total = len(valid)
    half = total / 2
    valid_ids_list = valid[:half]
    test_ids_list = valid[half + 1:]
    # 5 % of training into test data
    five_p = int(len(train) * 0.05)
    train_ids_list = train[:-five_p]
    test_ids_list = test_ids_list + train[-five_p:]
    # 10% of leftover training into valid data
    ten_p = int(len(train_ids_list) * 0.1)
    train_ids_list = train_ids_list[:-ten_p]
    valid_ids_list = valid_ids_list + train_ids_list[-ten_p:]
    return train_ids_list, valid_ids_list, test_ids_list


