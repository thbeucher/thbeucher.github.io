* Separable Convolution: (I put it for information but I'm not sure about my implementation as I get fewer parameters as expected but it seems computationally not efficient)
```python
class SeparableConvBlock(nn.Module):
  def __init__(self, in_chan, out_chan, kernel=3, stride=1, pad=1, dil=1, dropout=0., k=1, **kwargs):
    super().__init__()
    assert k in [1, 2], 'Handle only k = 1 or 2'
    self.conv = nn.Sequential(nn.Conv1d(in_chan, k * in_chan, kernel, stride=stride, padding=pad, dilation=dil, groups=in_chan),
                              nn.BatchNorm1d(k * in_chan),
                              nn.ReLU(inplace=True) if k == 1 else nn.GLU(dim=1),
                              nn.Dropout(dropout),
                              nn.Conv1d(k * in_chan, out_chan, 1),
                              nn.BatchNorm1d(out_chan),
                              nn.ReLU(inplace=True),
                              nn.Dropout(dropout))
  
  def forward(self, x):  # [batch_size, in_chan, seq_len]
    return self.conv(x)  # [batch_size, out_chan, seq_len]
```

* Basic convolutional block:
```python
class ConvBlock(nn.Module):
  def __init__(self, in_chan, out_chan, kernel=3, stride=1, pad=1, dil=1, dropout=0., groups=1, k=1, **kwargs):
    super().__init__()
    assert k in [1, 2], 'Handle only k = 1 or 2'
    self.conv = nn.Sequential(nn.Conv1d(in_chan, out_chan, kernel, stride=stride, padding=pad, dilation=dil, groups=groups),
                              nn.BatchNorm1d(out_chan),
                              nn.ReLU(inplace=True) if k == 1 else nn.GLU(dim=1),
                              nn.Dropout(dropout))
  
  def forward(self, x):  # [batch_size, in_chan, seq_len]
    return self.conv(x)  # [batch_size, out_chan, seq_len] or [batch_size, out_chan // 2, seq_len] if k == 2
```

* Convolution attention from [Lightconv](https://openreview.net/pdf?id=SkVhlh09tX):
```python
class AttentionConvBlock(nn.Module):
  def __init__(self, in_chan, n_heads=8, kernel=5, dropout=0., pad=2, bias=True, **kwargs):
    super().__init__()
    assert in_chan // n_heads * n_heads == in_chan, 'in_chan must be evenly divisible by n_heads'
    self.n_heads = n_heads
    self.dropout = dropout
    self.pad = pad
    self.bias = None

    self.weight = nn.Parameter(torch.Tensor(n_heads, 1, kernel))
    nn.init.xavier_uniform_(self.weight)

    if bias:
      self.bias = nn.Parameter(torch.Tensor(in_chan))
      nn.init.constant_(self.bias, 0.)
  
  def forward(self, x):  # [batch_size, in_chan, seq_len]
    in_ = x.reshape(-1, self.n_heads, x.size(2))
    weight = F.dropout(F.softmax(self.weight, dim=-1), self.dropout, training=self.training)
    out = F.conv1d(in_, weight, padding=self.pad, groups=self.n_heads).reshape(x.shape)

    if self.bias is not None:
      out = out + self.bias.view(1, -1, 1)
    return out
```

* Combination of a ConvBlock and an AttentionConvBlock:
```python
class ConvAttentionConvBlock(nn.Module):
  def __init__(self, in_chan, out_chan, kernel_conv=3, stride_conv=1, pad_conv=1, dil_conv=1, dropout_conv=0., groups=1, k=1,
               n_heads=8, kernel_attn=5, dropout_attn=0., pad_attn=2, bias=True):
    super().__init__()
    self.conv = ConvBlock(in_chan, out_chan, kernel=kernel_conv, stride=stride_conv, pad=pad_conv, dil=dil_conv, dropout=dropout_conv,
                          groups=groups, k=k)
    self.attn_conv = AttentionConvBlock(out_chan//k, n_heads=n_heads, kernel=kernel_attn, dropout=dropout_attn, pad=pad_attn, bias=bias)
  
  def forward(self, x):  # [batch_size, in_chan, seq_len]
    return self.attn_conv(self.conv(x))
```

We can also add a simple feed-forward network:
```python
class FeedForward(nn.Module):
  def __init__(self, input_size, output_size, d_ff=2048, dropout=0., **kwargs):
    super().__init__()
    self.ff = nn.Sequential(nn.Linear(input_size, d_ff),
                            nn.ReLU(inplace=True),
                            nn.Dropout(dropout),
                            nn.Linear(d_ff, output_size),
                            nn.Dropout(dropout),
                            nn.LayerNorm(output_size))
  
  def forward(self, x):  # [batch_size, *, input_size]
    return self.ff(x)   # [batch_size, *, output_size]
```

```python
class STTModel(nn.Module):
  def __init__(self, config=None, residual=True, output_size=None, input_proj=None, wav2vec_frontend=True, **kwargs):
    super().__init__()
    self.config = config
    self.output_proj = None
    ## list all authorized blocks
    self.available_blocks = {'conv_block': ConvBlock, 'attention_conv_block': AttentionConvBlock, 'feed_forward': FeedForward}
    
    ## retrieve features extractor
    self.wav2vec = get_wav2vec_model() if wav2vec_frontend else None
    
    ## retrieve projection layer if desired
    self.input_proj = get_input_proj_layer(config=input_proj)  # will return None if input_proj is None
    
    ## retrieve default config if not given
    if config is None:
      self.config = get_stt_model_config(config='base')
    
    ## define the network
    layers = []
    for layer in self.config:
      blocks = []
      for block in layer:
        sub_blocks = []
        for parallel_sub_block_type, block_config in block:
          sub_blocks.append(self.available_blocks[parallel_sub_block_type](**block_config))
        blocks.append(nn.ModuleList(sub_blocks))
      layers.append(nn.ModuleList(blocks))
    self.network = nn.ModuleList(layers)
    
    ## create the output projection if desired
    if output_size is not None:
      key = [k for k in ['output_size'] if k in self.config[-1][-1][-1][1]][0]
      self.output_proj = nn.Linear(self.config[-1][-1][-1][1][key], output_size)
    
  @torch.no_grad()
  def _wav2vec_forward(self, x):  # x = [batch_size, signal_len]
    z = self.wav2vec.feature_extractor(x)  # n_feats = 512
    c = self.wav2vec.feature_aggregator(z)  # [batch_size, n_feats, seq_len]
    return c.permute(0, 2, 1)  # [batch_size, seq_len, n_feats]
    
  def forward(self, x, y=None):
    if self.wav2vec is not None:
      x = self._wav2vec_forward(x)
    
    if self.input_proj is not None:
      x = self.input_proj(x)
    
    for i, layer in enumerate(self.network):
      out = x
      for j, block in enumerate(layer):
        outs = []
        for k, sub_block in enumerate(block):
          if 'conv' in self.config[i][j][k][0]:
            outs.append(sub_block(out.permute(0, 2, 1)).permute(0, 2, 1))
          else:
            outs.append(sub_block(out))
        out = torch.cat(outs, dim=-1)
      x = x + out if self.residual and out.shape == x.shape else out
    
    if self.output_proj is not None:
      x = self.output_proj(x)
    return x
```

where **get_wav2vec_model**, **get_input_proj_layer** and **get_stt_model_config** functions are defined as follow : 

```python
from fairseq.models.wav2vec import Wav2VecModel

def get_wav2vec_model(filename='wav2vec_large.pt', eval_model=True):
  checkpoint = torch.load(filename)
  wav2vec_model = Wav2VecModel.build_model(checkpoint['args'], task=None)
  wav2vec_model.load_state_dict(checkpoint['model'])
  
  if eval_model:
    wav2vec_model.eval()
  
  return wav2vec_model

def get_input_proj_layer(config='base'):
  if config == 'base':
    input_proj = nn.Sequential(nn.Dropout(0.25), nn.Linear(512, 512), nn.ReLU(inplace=True), nn.LayerNorm(512))
  else:
    input_proj = None
  return input_proj


def get_stt_model_config(config='base'):
  if config == 'whatever':
    raise NotImplementedError
  else:
    cnet_config = [
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 2, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ],
                    [
                      [
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 1,
                                                       'dil_conv': 1, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 2,
                                                       'dil_conv': 2, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True}),
                        ('conv_attention_conv_block', {'in_chan': 512, 'out_chan': 512, 'kernel_conv': 3, 'stride_conv': 1, 'pad_conv': 3,
                                                       'dil_conv': 3, 'dropout_conv': 0.25, 'groups': 1, 'k': 1, 'n_heads': 8,
                                                       'kernel_attn': 5, 'dropout_attn': 0.25, 'pad_attn': 2, 'bias': True})
                      ],
                      [('feed_forward', {'input_size': 3 * 512, 'output_size': 512, 'd_ff': 2048, 'dropout': 0.25})]
                    ]
                  ]
  return cnet_config
```

Now, to hold and preprocess the data we can define a Data class : 

```python
import soundfile as sf
from jiwer import wer as wer_compute

class Data(object):
  def __init__(self):
    self.vars_to_save = ['ids_to_audiofile_train', 'ids_to_audiofile_test', 'max_signal_len', 'max_source_len',
                         'ids_to_transcript_train', 'ids_to_transcript_test', 'ids_to_encodedsources_train',
                         'ids_to_encodedsources_test', 'idx_to_tokens', 'tokens_to_idx', 'n_signal_feats']

    self.ids_to_audiofile_train = {}
    self.ids_to_audiofile_test = {}
    self.max_signal_len = 0
    self.n_signal_feats = 0
    
    self.max_source_len = 0
    self.ids_to_transcript_train = {}
    self.ids_to_transcript_test = {}
    self.ids_to_encodedsources_train = {}
    self.ids_to_encodedsources_train = {}
    self.idx_to_tokens = []
    self.tokens_to_idx = {}
  
  def save_metadata(self, save_name='_Data_metadata.pk'):
    metadata = {name: getattr(self, name) for name in self.vars_to_save}
    with open(save_name, 'wb') as f:
      pk.dump(metadata, f)
  
  def load_metadata(self, save_name='_Data_metadata.pk'):
    with open(save_name, 'rb') as f:
      metadata = pk.load(f)
    for k, v in metadata.items():
      setattr(self, k, v)
  
  @staticmethod
  def read_audio_file(filename):
    return sf.read(filename)
  
  @staticmethod
  def raw_signal(signal, **kwargs):
    return signal
  
  @staticmethod
  def read_n_slice_signal(filename, slice_fn=None, **kwargs):
    slice_fn = Data.raw_signal if slice_fn is None else slice_fn
    signal, sample_rate = Data.read_audio_file(filename)
    return slice_fn(signal, **kwargs)
  
  @staticmethod
  def get_openslr_files(folder, key=None):
    dataset_filelist = {'audio': [], 'transcript': []}

    for fname in os.listdir(folder):
      full_path_fname = os.path.join(folder, fname)
      if os.path.isdir(full_path_fname):
        for f2name in os.listdir(full_path_fname):
          full_path_f2name = os.path.join(full_path_fname, f2name)
          if os.path.isdir(full_path_f2name):
            for f3name in os.listdir(full_path_f2name):
              filename = os.path.join(full_path_f2name, f3name)
              if '.flac' in f3name:
                dataset_filelist['audio'].append(filename)
              elif '.txt' in f3name:
                dataset_filelist['transcript'].append(filename)
    return dataset_filelist if key is None else dataset_filelist[key]
  
  @staticmethod
  def letters_encoding(sources, idx_to_letters=None, letters_to_idx=None, blank_token='<blank>', **kwargs):
    sources = [s.lower() for s in sources]
    
    if idx_to_letters is None or letters_to_idx is None:
      letters = list(sorted(set([l for s in sources for l in s])))
      idx_to_letters = [blank_token] + letters
      letters_to_idx = {l: i for i, l in enumerate(idx_to_letters)}
    
    sources_encoded = [[letters_to_idx[l] for l in s] for s in sources]
    return sources_encoded, idx_to_letters, letters_to_idx
  
  @staticmethod
  def reconstruct_sentences(sentences, idx_to_tokens, blank_idx=0):
    reconstructed_sentences = [[i for i, _ in groupby(p)] for s in sentences]
    return [''.join([idx_to_tokens[i] for i in rs if i != blank_idx]) for rs in reconstructed_sentences]
  
  @staticmethod
  def compute_wer(targets, predictions, idx_to_tokens, reconstruct=True, blank_idx=0):
    if reconstruct:
      targets = Data.reconstruct_sentences(targets, idx_to_tokens, blank_idx=blank_idx)
      predictions = Data.reconstruct_sentences(predictions, idx_to_tokens, blank_idx=blank_idx)
    return np.mean([wer_compute(t, p) for t, p in zip(targets, predictions)])
  
  @staticmethod
  def read_openslr_transcript_file(filename):
    with open(filename, 'r') as f:
      transcripts = f.read().splitlines()
    return zip(*[t.split(' ', 1) for t in transcripts])
  
  def get_transcripts(self, folder, list_files_fn=None, parse_fn=None, var_name='idx_to_transcript'):
    list_files_fn = Data.get_openslr_files if list_files_fn is None else list_files_fn
    parse_fn = Data.read_openslr_transcript_file if parse_fn is None else parse_fn
    
    ids_to_transcript = {}
    for filename in list_files_fn(folder)
  
  def process_openslr_transcripts(self, train_folder, test_folder, encoding_fn=None, **kwargs):
    encoding_fn = Data.letters_encoding if encoding_fn is None else encoding_fn
```

Then, to use the [DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader) from pytorch we have to define our custom dataset class and the associated collator : 

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  def __init__(self, ids_to_audiofile, ids_to_encodedsources, sort_by_target_len=True, **kwargs):
    self.ids_to_audiofilefeatures = {i: f for i, f in ids_to_audiofile.items()}
    self.ids_to_encodedsources = ids_to_encodedsources
    self.identities = list(sorted(ids_to_encodedsources.keys()))

    self.process_file_fn = kwargs.get('process_file_fn', Data.read_and_slice_signal)
    kwargs['slice_fn'] = kwargs.get('slice_fn', Data.wav2vec_extraction)
    kwargs['save_features'] = kwargs.get('save_features', True)
    self.process_file_fn_args = kwargs

    self.ids_input_lens = {}
    if kwargs['slice_fn'] == Data.wav2vec_extraction or kwargs.get('use_wav2vec', True):
      for i, f in self.ids_to_audiofilefeatures.items():
        fname = f.replace('.flac', '.wav2vec_shape.pk')
        if os.path.isfile(fname):
          self.ids_input_lens[i] = pk.load(open(fname, 'rb'))[0]

    if sort_by_target_len:
      self.identities = CustomDataset._sort_by_targets_len(self.identities, ids_to_encodedsources)
  
  @staticmethod
  def _sort_by_targets_len(ids, ids2es):
    return list(map(lambda x: x[0], sorted([(i, len(ids2es[i])) for i in ids], key=lambda x: x[1])))
  
  def __len__(self):
    return len(self.identities)
  
  def __getitem__(self, idx):
    signal = self.process_file_fn(self.ids_to_audiofilefeatures[self.identities[idx]], **self.process_file_fn_args)
    input_ = torch.Tensor(signal) if isinstance(signal, np.ndarray) else signal

    target = torch.LongTensor(self.ids_to_encodedsources[self.identities[idx]])
    input_len = self.ids_input_lens[self.identities[idx]] if self.identities[idx] in self.ids_input_lens else len(input_)
    target_len = len(target)
    return input_, target, input_len, target_len

class CustomCollator(object):
  def __init__(self, audio_pad, text_pad):
    self.audio_pad = audio_pad
    self.text_pad = text_pad
  
  def __call__(self, batch):
    inputs, targets, input_lens, target_lens = zip(*batch)
    inputs_batch = pad_sequence(inputs, batch_first=True, padding_value=self.audio_pad).float()
    targets_batch = pad_sequence(targets, batch_first=True, padding_value=self.text_pad)
    input_lens = torch.LongTensor(input_lens)
    target_lens = torch.LongTensor(target_lens)
    return inputs_batch, targets_batch, input_lens, target_lens
```

Finally, we can define a class that will handle the training, evaluation and everything else that we want : 

```python
class CTCTrainer(object):
  def __init__(self):
    pass
  
  def set_metadata(self):
    pass
  
  def set_data_loader(self):
    pass
  
  def instanciate_model(self):
    pass
  
  def train(self):
    pass
  
  def train_pass(self):
    pass
  
  @torch.no_grad():
  def evaluation(self):
    pass
```
