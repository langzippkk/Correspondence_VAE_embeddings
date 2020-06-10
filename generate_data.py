import logging as log
import h5py
import numpy as np
import torch
import json
import torch.utils.data as tud
from utils.loader import load
from saver.saver import savez
from data.datasets import SpeechDataset


class MultiViewDataset(SpeechDataset):

  def __init__(self, feats, align, vocab, min_dur=2, max_dur=None,
               min_seg_dur=2, max_seg_dur=None, stack=False):
    #super(MultiViewDataset, self).__init__()

    feats = h5py.File(feats, "r")
    align = h5py.File(align, "r")

    num_ignored = 0
    examples = {}
    for uid, g in align.items():
      ind = []
      frames = len(feats[uid][:])
      if frames < min_dur or (max_dur is not None and frames > max_dur):
        continue
      words = g["words"][:]
      durs = g["ends"][:] - g["starts"][:] + 1
      for i, (w, d) in enumerate(zip(words, durs)):
        if d < min_seg_dur or (max_seg_dur is not None and d > max_seg_dur):
          num_ignored += 1
          continue
        if w not in vocab.w2i:
          continue
        ind.append(i)
      if len(ind) == 0:
        num_ignored += 1
        continue
      examples[(uid, tuple(ind))] = {
        "frames": frames // 2 if stack else frames,
        "words": len(ind)
      }

    seg = self.add_deltas(feats[list(examples)[0][0]][:])
    if stack:
      seg = self.stack_input_frames(seg)

    _, input_feat_dim = seg.shape
    input_num_subwords = len(vocab.s2i)

    log.info(f" >> feats: loading {feats.filename}")
    log.info(f" >> align: loading {align.filename}")
    log.info(f" >> stack= {stack}")
    log.info(f" >> durs= ({min_dur}, {max_dur})")
    log.info(f" >> seg_durs= ({min_seg_dur}, {max_seg_dur})")
    log.info(f" >> # utts= {len(examples)} (# ignored= {num_ignored})")
    log.info(f" >> # words= {sum(len(k[1]) for k in examples)}")
    log.info(f" >> input_feat_dim= {input_feat_dim}")
    log.info(f" >> input_num_subwords= {input_num_subwords}")

    self.feats = feats
    self.stack = stack
    self.align = align
    self.vocab = vocab
    self.examples = examples
    self.input_feat_dim = input_feat_dim
    self.input_num_subwords = input_num_subwords

  def __getitem__(self, ex):

    uid, ind = ex

    seg = self.feats[uid][:]
    starts = [self.align[uid]["starts"][:][i] for i in ind]
    ends = [self.align[uid]["ends"][:][i] for i in ind]
    seg = self.add_deltas(seg)
    if self.stack:
      seg = self.stack_input_frames(seg)
      starts = [start // 2 for start in starts]
      ends = [end // 2 for end in ends]

    words = np.array([self.align[uid]["words"][:][i] for i in ind])

    return {
      "seg": seg, "starts": starts, "ends": ends,
      **self.vocab.collate_fn([self.vocab[word] for word in words])
    }

class MultiViewDataset_IndividualWords(SpeechDataset):

  def __init__(self, feats, align, vocab,
               min_seg_dur=2, max_seg_dur=None, stack=False):
    #super(MultiViewDataset_IndividualWords, self).__init__()

    feats = h5py.File(feats, "r")
    align = h5py.File(align, "r")

    num_ignored = 0
    examples = {}
    for uid, g in align.items():
      words = g["words"][:]
      durs = g["ends"][:] - g["starts"][:] + 1
      for i, (w, d) in enumerate(zip(words, durs)):
        if d < min_seg_dur or (max_seg_dur is not None and d > max_seg_dur):
          num_ignored += 1
          continue
        if w not in vocab.w2i:
          continue
        examples[(uid, i)] = {"frames": d}
    print(feats[list(examples)[0][0]][:].shape)
    seg = self.add_deltas(feats[list(examples)[0][0]][:])
    ## add derivatives from 36 dim to 108

    if stack:
      seg = self.stack_input_frames(seg)

    _, input_feat_dim = seg.shape
    input_num_subwords = len(vocab.s2i)

    log.info(f" >> feats: loading {feats.filename}")
    log.info(f" >> align: loading {align.filename}")
    log.info(f" >> stack= {stack}")
    log.info(f" >> seg_durs= ({min_seg_dur}, {max_seg_dur})")
    log.info(f" >> # words= {len(examples)} (# ignored= {num_ignored})")
    log.info(f" >> input_feat_dim= {input_feat_dim}")
    log.info(f" >> input_num_subwords= {input_num_subwords}")

    self.feats = feats
    self.stack = stack
    self.align = align
    self.vocab = vocab
    self.examples = examples
    self.input_feat_dim = input_feat_dim
    self.input_num_subwords = input_num_subwords

  def __getitem__(self, ex):

    uid, i = ex

    start = self.align[uid]["starts"][:][i]
    end = self.align[uid]["ends"][:][i]
    seg = self.feats[uid][:][start:end]
    seg = self.add_deltas(seg)
    if self.stack:
      seg = self.stack_input_frames(seg)

    word = self.align[uid]["words"][:][i]
    return {"seg": seg,"word":word, **self.vocab[word]}

  def collate_fn(self, batch):

    durs = torch.LongTensor([len(ex["seg"]) for ex in batch])
    segs = torch.zeros(len(durs), max(durs), self.input_feat_dim)
    for i, ex in enumerate(batch):
      segs[i, :durs[i]] = torch.from_numpy(ex["seg"])

    ids = np.array([ex["id"] for ex in batch], dtype=np.int32)
    uids, ind, inv_ind = np.unique(
        ids, return_index=True, return_inverse=True)

    lens = torch.LongTensor([len(batch[j]["seq"]) for j in ind])
    seqs = torch.zeros(len(lens), max(lens), dtype=torch.long)
    for i, j in enumerate(ind):
      seqs[i, :lens[i]] = torch.from_numpy(batch[j]["seq"])

    return {
      "view1": segs, "view1_lens": durs,
      "view2": seqs, "view2_lens": lens,
      "ids": uids, "inv": torch.from_numpy(inv_ind)
    }


###################################################################################
class Dataset(tud.Dataset):

  @property
  def iter(self):
    return self.loader.batch_sampler.iter

  def __iter__(self):
    log.info(f"Iterating {self.__class__.__name__} (start_iter= {self.iter})")
    self.iterator = iter(self.loader)
    return self.iterator

  def __len__(self):
    return len(self.loader)

  def init_data_loader(self, batch_sampler):
    self.loader = tud.DataLoader(self, num_workers=1,
                                 batch_sampler=batch_sampler,
                                 collate_fn=self.collate_fn)
    log.info(f" >> {batch_sampler.__class__.__name__}; {len(self)} batches")
################################################################################

class MultiViewVocab(Dataset):

  def __init__(self, lexicon, subwords, counts=None, min_count=0):
    super(MultiViewVocab, self).__init__()

    with open(lexicon, "r") as f:
      lexicon = json.load(f)
      w2i = lexicon["words_to_ids"]
      # Note: add 1 since padding_idx=0
      s2i = {s: i + 1 for s, i in lexicon[f"{subwords}_to_ids"].items()}
      # Note: use first pronunciation
      w2s = {w: s[0] for w, s in lexicon[f"word_to_{subwords}"].items()}

    num_removed = 0
    if min_count > 0:
      with open(counts, "r") as f:
        counts = json.load(f)
      for w in list(w2i):
        if counts.get(w, 0) < min_count:
          num_removed += 1
          del w2i[w], w2s[w]

    log.info(f" >> min_count= {min_count}")
    log.info(f" >> # words= {len(w2i)} (# removed= {num_removed})")
    log.info(f" >> # subwords= {len(s2i)}")
    for s, i in s2i.items():
      log.info(f"    {s}= {i}")


    self.w2i = w2i
    self.i2w = {i: w for w, i in w2i.items()}
    self.s2i = s2i
    self.w2s = w2s
    self.examples = list(w2i)

  def __getitem__(self, word):

    seq = np.array([self.s2i[s] for s in self.w2s[word]], dtype=np.int32)
    id_ = self.w2i[word]

    return {"seq": seq, "id": id_}

  def collate_fn(self, batch):

    lens = torch.LongTensor([len(ex["seq"]) for ex in batch])
    seqs = torch.zeros(len(lens), max(lens), dtype=torch.long)
    for i, ex in enumerate(batch):
        seqs[i, :lens[i]] = torch.from_numpy(ex["seq"])

    ids = np.array([ex["id"] for ex in batch], dtype=np.int32)

    return {"view2": seqs, "view2_lens": lens, "ids": ids}

#####################################Contextual###########################

##############################################################################
feats = "C:/Users/59381/Desktop/agwe-recipe-master/data/fbank.test.hdf5"
align = "C:/Users/59381/Desktop/agwe-recipe-master/data/align-small.test.hdf5"
vocab = "C:/Users/59381/Desktop/agwe-recipe-master/data/vocab.json"
# vocab = load(vocab, config)
# vocab_sampler = load("vocab_sampler", config, examples=vocab.examples)
# vocab.init_data_loader(vocab_sampler)
with open(vocab) as json_file:
    data = json.load(json_file)
subwords = ['phones','chars','spoken_chars_to_ids']
vocab = MultiViewVocab(vocab,subwords[0])
dataset = MultiViewDataset_IndividualWords(feats,align,vocab)
multiview = MultiViewDataset(feats,align,vocab)
# print(vocab['benefits'])
## output the id
# feats = h5py.File(feats, "r")
# print("dataset1")
# print(dataset[('sw02062-A_000131-000595', 0)])
# ## dim: mfcc : 108, num_Frames: 73, total_num_frams:
# print(feats['sw02062-A_000131-000595'])
# ## dim: mfcc : 36, num_Frames: 73, total_num_frams:
# temp = (dataset[('sw02062-A_000131-000595', 0)])
# print(temp['seg'].shape)


dictionary = {}
counter = 0
for i in dataset.examples.keys():
  data_temp = dataset[i]
  word_temp = data_temp['word']
  feature_temp = data_temp['seg']
  id_temp = data_temp['id']
  key_temp = word_temp+"_"+str(id_temp)+"_"+str(counter)
  counter +=1
  dictionary[key_temp] = feature_temp
print(sorted(dictionary))
print(len(dictionary))
np.savez('C:/Users/59381/Desktop/test.npz', **dictionary)
