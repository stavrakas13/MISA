import math
import numpy as np, random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from utils import to_gpu
from utils import ReverseLayerF

from mmlatch import Feedback   # MMLatch feedback block

class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        # self.text_size     = config.embedding_size
        self.text_size     = 768 if config.use_bert else config.embedding_size

        self.visual_size   = config.visual_size
        self.acoustic_size = config.acoustic_size

        self.input_sizes  = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size  = config.num_classes
        self.dropout_rate = config.dropout
        self.activation   = self.config.activation()
        self.tanh         = nn.Tanh()

        # choose RNN cell per config
        rnn = nn.LSTM if config.rnncell == 'lstm' else nn.GRU

        # --- Stage I: text encoder ---
        if config.use_bert:
            bertcfg = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertcfg)
        else:
            self.embed = nn.Embedding(len(config.word2id), self.text_size)
            self.trnn1 = rnn(self.text_size, self.hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*self.hidden_sizes[0], self.hidden_sizes[0], bidirectional=True)

        # --- Stage I: visual & acoustic encoders ---
        self.vrnn1 = rnn(self.visual_size,   self.hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*self.hidden_sizes[1], self.hidden_sizes[1], bidirectional=True)
        self.arnn1 = rnn(self.acoustic_size, self.hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*self.hidden_sizes[2], self.hidden_sizes[2], bidirectional=True)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=self.hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=self.hidden_sizes[1]*4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=self.hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        # --- private encoders ---
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())

        # --- shared encoder ---
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        # --- discriminators ---
        if not config.use_cmd_sim:
            self.discriminator = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                self.activation,
                nn.Dropout(self.dropout_rate),
                nn.Linear(config.hidden_size, len(self.hidden_sizes))
            )
        self.sp_discriminator = nn.Linear(config.hidden_size, 4)

        # --- fusion modules ---
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= self.output_size))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # --- layer norms for seq outputs ---
        self.tlayer_norm = nn.LayerNorm(self.hidden_sizes[0]*2)
        self.vlayer_norm = nn.LayerNorm(self.hidden_sizes[1]*2)
        self.alayer_norm = nn.LayerNorm(self.hidden_sizes[2]*2)

        # --- NEW: MMLatch feedback block ---
        hi_dims  = (self.text_size,          # π.χ. 768
            self.acoustic_size*2,    # π.χ. 74*2 = 148
            self.visual_size*2)      # π.χ. 47*2 = 94

        low_dims = (self.text_size,          # 768
                    self.acoustic_size,      # 74
                    self.visual_size)        # 47

        self.feedback = Feedback(
            hi_dims  = hi_dims,
            low_dims = low_dims,
            mask_type = "learnable_static_mask",
            dropout   = 0.1,
            device    = config.device,
        )
    
    # def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
    #     # move lengths to CPU and ensure it's long\
    #     lengths = lengths.clamp(min=1, max=sequence.size(1)) # ← NEW
    #     cpu_lengths = lengths.cpu().long()
    #     packed_sequence = pack_padded_sequence(sequence, cpu_lengths, batch_first=True, enforce_sorted=False)

    #     if self.config.rnncell == "lstm":
    #         packed_h1, (final_h1, _) = rnn1(packed_sequence)
    #     else:
    #         packed_h1, final_h1 = rnn1(packed_sequence)

    #     padded_h1, _ = pad_packed_sequence(packed_h1)
    #     normed_h1 = layer_norm(padded_h1)
    #     packed_normed_h1 = pack_padded_sequence(normed_h1, cpu_lengths, batch_first=True, enforce_sorted=False)


    #     if self.config.rnncell == "lstm":
    #         _, (final_h2, _) = rnn2(packed_normed_h1)
    #     else:
    #         _, final_h2 = rnn2(packed_normed_h1)

    #     return final_h1, final_h2
    # =========================================================
# 1)  extract_features   (επιστρέφει ΤΕΛΙΚΑ hidden states)
# =========================================================
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        # --- ασφαλές lengths -------------------------------------------------
        lengths     = lengths.clamp(min=1, max=sequence.size(1))
        cpu_lengths = lengths.cpu().long()
        # ---------------------------------------------------------------------

        # 1ο BiRNN -------------------------------------------------------------
        packed1 = pack_padded_sequence(sequence, cpu_lengths,
                                    batch_first=True, enforce_sorted=False)
        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed1)     # final_h1: (dirs, B, H)
        else:
            packed_h1, final_h1 = rnn1(packed1)

        # layer-norm στο sequence ---------------------------------------------
        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)  # (B,L,H)
        normed_h1    = layer_norm(padded_h1)

        # 2ο BiRNN -------------------------------------------------------------
        packed2 = pack_padded_sequence(normed_h1, cpu_lengths,
                                    batch_first=True, enforce_sorted=False)
        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed2)
        else:
            _, final_h2 = rnn2(packed2)

        return final_h1, final_h2                          # (dirs,B,H) , (dirs,B,H)
    
    
    def alignment(self, sentences, visual, acoustic, lengths, len_t, len_v, len_a,
                  bert_sent, bert_type, bert_mask):
        
        batch_size = lengths.size(0)

        if self.config.use_bert:
            # 1.a BERT contextual output (με specials)
            bert_ctx = self.bertmodel(
                input_ids=bert_sent,
                attention_mask=bert_mask,
                token_type_ids=bert_type,
                return_dict=True
            ).last_hidden_state                             # (B, 77, 768)

            # 1.b Raw BERT embeddings (WordPiece + position + segment)
            raw_t = self.bertmodel.embeddings(
                input_ids=bert_sent,
                token_type_ids=bert_type
            )                                               # (B, 77, 768)

            # 1.c Κόβουμε [CLS]/[SEP] για raw & ctx
            raw_t = raw_t[:, 1:-1, :]                       # (B, L, 768)
            ctx_t = bert_ctx[:, 1:-1, :]                    # (B, L, 768)
            mask_t = bert_mask[:, 1:-1]                     # (B, L)

            # 1.d Utterance-level RAW = masked mean των raw embed
            raw_t_mean = (raw_t * mask_t.unsqueeze(2)).sum(1) / mask_t.sum(1, keepdim=True)
            # (B, 768)

            # 1.e Utterance-level CONTEXTUAL = masked mean των ctx
            utterance_text = (ctx_t * mask_t.unsqueeze(2)).sum(1) / mask_t.sum(1, keepdim=True)
            # (B, 768)
        else:
            emb_t = self.embed(sentences)                   # (B, L, D)
            seq_t, _ = self.extract_features_seq(
                emb_t, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            raw_t_mean     = emb_t.mean(1)                  # (B, D)
            utterance_text = seq_t.mean(1)                  # (B, 768)
            
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = final_h2v.permute(1,0,2).contiguous().view(batch_size, -1)  # (B, 94)

        # print(seq_v.shape, "Shape of seq visual")

        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = final_h2a.permute(1,0,2).contiguous().view(batch_size, -1)  # (B, 148)

        # print(seq_a.shape, "Shape of seq acoustic")

        L = raw_t.size(1) 
        
        hi_x_seq = utterance_text.unsqueeze(1).expand(-1, L, -1)   # (B, L, 768)
        hi_y_seq = utterance_audio.unsqueeze(1).expand(-1, L, -1)  # (B, L, 148)
        hi_z_seq = utterance_video.unsqueeze(1).expand(-1, L, -1)  # (B, L, 94)

        # seq_t_out = bert_out
        # --- MMLatch feedback on sequences ---
        seq_t, seq_a, seq_v = self.feedback(
            low_x = raw_t,   # (B,75,768)
            low_y = acoustic,          # (B,75,74)
            low_z = visual,            # (B,75,35)
            hi_x  = hi_x_seq,         # (B,75,768)
            hi_y  = hi_y_seq,             # (B,75,148)
            hi_z  = hi_z_seq,             # (B,75,70)
            lengths = len_t            # len_t == 75
        )


        pad_cls = raw_t[:, :1, :]      # (B,1,768)
        pad_sep = raw_t[:, -1:, :]
        seq_t_full = torch.cat([pad_cls, seq_t, pad_sep], dim=1)   # (B,77,768)

        B = lengths.size(0)

        # Stage II: re-encode masked sequences
        if self.config.use_bert:
            bert_out2 = self.bertmodel(
                inputs_embeds=seq_t_full,
                attention_mask=bert_mask,
                return_dict=True
            ).last_hidden_state
            mask_len2 = bert_mask.sum(1, keepdim=True)
            fused_t = (bert_out2 * bert_mask.unsqueeze(2)).sum(1) / mask_len2
            utt_t = self.project_t(fused_t)
        else:
            f_h1_t, f_h2_t = self.extract_features(
                seq_t, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            utt_t = self.project_t(torch.cat((f_h1_t, f_h2_t), dim=1))

                # ----------------- VISUAL -----------------
        f_h1_v, f_h2_v = self.extract_features(
                seq_v, len_v, self.vrnn1, self.vrnn2, self.vlayer_norm)

        # reshape 2×35 -> 70
        B = lengths.size(0)
        f_h1_v = f_h1_v.permute(1,0,2).contiguous().view(B, -1)
        f_h2_v = f_h2_v.permute(1,0,2).contiguous().view(B, -1)
        utt_v  = self.project_v(torch.cat((f_h1_v, f_h2_v), dim=1))

        # ----------------- ACOUSTIC  --------------  (χρησιμοποίησε len_a!)
        f_h1_a, f_h2_a = self.extract_features(
                seq_a, len_a, self.arnn1, self.arnn2, self.alayer_norm)

        f_h1_a = f_h1_a.permute(1,0,2).contiguous().view(B, -1)
        f_h2_a = f_h2_a.permute(1,0,2).contiguous().view(B, -1)
        utt_a  = self.project_a(torch.cat((f_h1_a, f_h2_a), dim=1))

        # Continue with original MISA pipeline
        self.shared_private(utt_t, utt_v, utt_a)
        if not self.config.use_cmd_sim:
            rt = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            rv = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            ra = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)
            self.domain_label_t = self.discriminator(rt)
            self.domain_label_v = self.discriminator(rv)
            self.domain_label_a = self.discriminator(ra)
        else:
            self.domain_label_t = self.domain_label_v = self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s   = self.sp_discriminator(
            (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a) / 3.0)
        
        self.reconstruct()
        
        h_stack = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a,
                               self.utt_shared_t,  self.utt_shared_v,  self.utt_shared_a), dim=0)
        h_fused = self.transformer_encoder(h_stack)
        h_cat   = torch.cat([h_fused[i] for i in range(6)], dim=1)
        return self.fusion(h_cat)

    def reconstruct(self):
        self.utt_t = self.utt_private_t + self.utt_shared_t
        self.utt_v = self.utt_private_v + self.utt_shared_v
        self.utt_a = self.utt_private_a + self.utt_shared_a
        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, ut, uv, ua):
        self.utt_t_orig = ut; self.utt_v_orig = uv; self.utt_a_orig = ua
        self.utt_private_t = self.private_t(ut)
        self.utt_private_v = self.private_v(uv)
        self.utt_private_a = self.private_a(ua)
        self.utt_shared_t  = self.shared(ut)
        self.utt_shared_v  = self.shared(uv)
        self.utt_shared_a  = self.shared(ua)

    def forward(self, sentences, video, acoustic, lengths, len_t, len_v, len_a,
                bert_sent, bert_type, bert_mask):
        return self.alignment(sentences, video, acoustic,
                              lengths, len_t, len_v, len_a, bert_sent, bert_type, bert_mask)
