# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright Tor Vergata, University of Rome. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Data processor for the QC dataset

import os
import csv
import tensorflow as tf
import tokenization


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               label_mask=None,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example
    self.label_mask = label_mask


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_labeled_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_unlabeled_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class QcFineProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_labeled_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "labeled.tsv"), "train")

    def get_unlabeled_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "unlabeled.tsv"), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "test.tsv"), "test")

    def get_labels(self):
        """See base class."""
        return ["UNK_UNK", "ABBR_abb", "ABBR_exp", "DESC_def", "DESC_desc", "DESC_manner", "DESC_reason", "ENTY_animal", "ENTY_body", "ENTY_color", "ENTY_cremat", "ENTY_currency", "ENTY_dismed", "ENTY_event", "ENTY_food", "ENTY_instru", "ENTY_lang", "ENTY_letter", "ENTY_other", "ENTY_plant", "ENTY_product", "ENTY_religion", "ENTY_sport", "ENTY_substance", "ENTY_symbol", "ENTY_techmeth", "ENTY_termeq", "ENTY_veh", "ENTY_word", "HUM_desc", "HUM_gr", "HUM_ind", "HUM_title", "LOC_city", "LOC_country", "LOC_mount", "LOC_other", "LOC_state", "NUM_code", "NUM_count", "NUM_date", "NUM_dist", "NUM_money", "NUM_ord", "NUM_other", "NUM_perc", "NUM_period", "NUM_speed", "NUM_temp", "NUM_volsize", "NUM_weight"]

    def _create_examples(self, input_file, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        with open(input_file, 'r') as f:
            contents = f.read()
            file_as_list = contents.splitlines()
            for line in file_as_list[1:]:
                split = line.split(" ")
                question = ' '.join(split[1:])

                guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line))
                text_a = tokenization.convert_to_unicode(question)
                inn_split = split[0].split(":")
                label = inn_split[0] + "_" + inn_split[1]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            f.close()

        return examples
