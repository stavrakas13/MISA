import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from memory import MemoryModule

class PadPackedSequence(nn.Module):
    """Some Information about PadPackedSequence"""

    def __init__(self, batch_first=True):
        super(PadPackedSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        max_length = lengths.max().item()
        x, _ = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=max_length
        )
        return x


class PackSequence(nn.Module):
    def __init__(self, batch_first=True):
        super(PackSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        x = pack_padded_sequence(
            x, lengths, batch_first=self.batch_first, enforce_sorted=False
        )
        lengths = lengths[x.sorted_indices]
        return x, lengths


class RNN_latch(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat",
        dropout=0,
        rnn_type="lstm",
        packed_sequence=True,
        device="cpu",
        memory_augmented=False,
    ):
        # hidden_size=148 #changed...
        super(RNN_latch, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        # self.hidden_size = 148
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.merge_bi = merge_bi
        self.rnn_type = rnn_type.lower()

        self.out_size = hidden_size
        # self.out_size = 148

        if bidirectional and merge_bi == "cat":
            self.out_size = 2 * hidden_size

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            batch_first=batch_first,
            num_layers=layers,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)
        self.packed_sequence = packed_sequence

        if packed_sequence:
            self.pack = PackSequence(batch_first=batch_first)
            self.unpack = PadPackedSequence(batch_first=batch_first)

        self.memory_augmented = memory_augmented
        if memory_augmented:
            self.memory_module = MemoryModule(
                memory_slots=32,
                memory_dim=self.out_size,  # match RNN output dim
                controller_dim=self.out_size
            )
            self.memory_gate = nn.Linear(self.out_size, self.out_size)

    def _merge_bi(self, forward, backward):
        if self.merge_bi == "sum":
            return forward + backward

        return torch.cat((forward, backward), dim=-1)

    def _select_last_unpadded(self, out, lengths):
        gather_dim = 1 if self.batch_first else 0
        gather_idx = (
            (lengths - 1)  # -1 to convert to indices
            .unsqueeze(1)  # (B) -> (B, 1)
            .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
            # (B, 1, H) if batch_first else (1, B, H)
            .unsqueeze(gather_dim)
        )
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)

        return last_out

    def _final_output(self, out, lengths):
        # Collect last hidden state
        # Code adapted from https://stackoverflow.com/a/50950188

        if not self.bidirectional:
            return self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])
        # Last backward corresponds to first token
        last_backward_out = backward[:, 0, :] if self.batch_first else backward[0, ...]
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)

        return self._merge_bi(last_forward_out, last_backward_out)

    def merge_hidden_bi(self, out):
        if not self.bidirectional:
            return out

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])

        return self._merge_bi(forward, backward)

    def forward(self, x, lengths, initial_hidden=None):
        self.rnn.flatten_parameters()

        if self.packed_sequence:
            lengths = lengths.to("cpu")
            x, lengths = self.pack(x, lengths)
            lengths = lengths.to(self.device)

        if initial_hidden is not None:
            out, hidden = self.rnn(x, initial_hidden)
        else:
            out, hidden = self.rnn(x)

        if self.packed_sequence:
            out = self.unpack(out, lengths)

        out = self.drop(out)
        last_timestep = self._final_output(out, lengths)
        out = self.merge_hidden_bi(out)

        if self.memory_augmented:
            mem_out = self.memory_module(out)  # shape: [B, L, D]
            gate = torch.sigmoid(self.memory_gate(out))  # shape: [B, L, D]
            out = gate * out + (1 - gate) * mem_out

        return out, last_timestep, hidden

class FeedbackUnit(nn.Module):
    """
    Δημιουργεί δύο μάσκες (από y & z), τις συνδυάζει και
    τις εφαρμόζει στο x.

    in_dim  : διαστάσεις των hi-vectors  (y, z)
    out_dim : διαστάσεις του low-vector  (x)   – πρέπει να ταιριάζει!
    """
    def __init__(self,
                 hi_y_dim,
                 hi_z_dim,
                 out_dim,
                 mask_type="learnable_sequence_mask",
                 dropout=0.1,
                 device="cpu",
                 memory_augmented=False):
        super().__init__()
        self.mask_type = mask_type
        self.dropout   = dropout

        self.mask_type = mask_type    # <-- FIX: χρειάζεται για το dict lookup

        if mask_type == "learnable_sequence_mask":
            self.mask_y = RNN_latch(hi_y_dim, out_dim, dropout=dropout, device=device, memory_augmented=memory_augmented)
            self.mask_z = RNN_latch(hi_z_dim, out_dim, dropout=dropout, device=device, memory_augmented=memory_augmented)
        else:  # "learnable_static_mask"
            self.mask_y = nn.Linear(hi_y_dim, out_dim)
            self.mask_z = nn.Linear(hi_z_dim, out_dim)
            
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()

        self._mask_fn = {
            "learnable_static_mask":   self._learnable_static_mask,
            "learnable_sequence_mask": self._learnable_sequence_mask,
        }[mask_type]

    # ------------------  internal helpers ------------------

    def _learnable_sequence_mask(self, y, z, lengths):
        """
        y, z : (B, L, in_dim)
        lengths : (B,) LongTensor
        """
        oy, _, _ = self.mask_y(y, lengths)   # (B, L, out_dim)
        oz, _, _ = self.mask_z(z, lengths)   # (B, L, out_dim)
        return 0.5 * (self.sigmoid(oy) + self.sigmoid(oz))

    def _learnable_static_mask(self, y, z, _):
        """
        static → δεν χρειάζονται lengths
        """
        oy = self.mask_y(y)                  # (B, L, out_dim)
        oz = self.mask_z(z)                  # (B, L, out_dim)
        return 0.5 * (self.sigmoid(oy) + self.sigmoid(oz))

    # --------------------------------------------------
    def forward(self, x, y, z, lengths):
        """
        x : (B, L, out_dim)  — low-level sequence που θα φιλτραριστεί
        y : (B, L, in_dim)   — hi-vector από 1η modality
        z : (B, L, in_dim)   — hi-vector από 2η modality
        """
        mask = F.dropout(
            self._mask_fn(y, z, lengths), p=self.dropout, training=self.training
        )                                     # (B, L, out_dim)
        return x * mask

# ============================================================
#                 F E E D B A C K    B L O C K
# ============================================================
class Feedback(nn.Module):
    """
    hi_dims  : (d_t, d_a, d_v)  ← πλάτη των contextualised seq   (seq_t, seq_a, seq_v)
    low_dims : (r_t, r_a, r_v)  ← πλάτη των raw seq             (raw_t, acoustic, visual)
    """
    def __init__(self,
                 hi_dims,
                 low_dims,
                 mask_type="learnable_sequence_mask",
                 dropout=0.1,
                 device="cpu",
                 memory_augmented=memory_augmented):
        super().__init__()

        dt, da, dv = hi_dims     # 768, 148, 94        (π.χ.)
        rt, ra, rv = low_dims    # 768,  74, 47        (π.χ.)

        # Για να φιλτράρω TEXT χρησιμοποιώ hi-AUDIO + hi-VISION → διαστάσεις da, dv
        self.f_t = FeedbackUnit(hi_y_dim=da, hi_z_dim=dv, out_dim=rt,
                                mask_type=mask_type, dropout=dropout, device=device,
                                memory_augmented=memory_augmented)

        # Για να φιλτράρω AUDIO χρησιμοποιώ hi-TEXT  + hi-VISION → διαστάσεις dt, dv
        self.f_a = FeedbackUnit(hi_y_dim=dt, hi_z_dim=dv, out_dim=ra,
                                mask_type=mask_type, dropout=dropout, device=device,
                                memory_augmented=memory_augmented)

        # Για να φιλτράρω VISION χρησιμοποιώ hi-TEXT  + hi-AUDIO  → διαστάσεις dt, da
        self.f_v = FeedbackUnit(hi_y_dim=dt, hi_z_dim=da, out_dim=rv,
                                mask_type=mask_type, dropout=dropout, device=device,
                                memory_augmented=memory_augmented)

    # ------------------------------------------------------------
    def forward(self,
                low_x, low_y, low_z,      # raw  seq  (x)
                hi_x,  hi_y,  hi_z,       # high seq  (y,z)
                lengths):
        """
        Επιστρέφει τα low-seq αφού εφαρμοστούν οι αντίστοιχες μάσκες.
        Όλες οι ακολουθίες υποθέτουμε ότι έχουν κοινά μήκη `lengths` (B,).
        """
        x = self.f_t(low_x, hi_y, hi_z, lengths)   # text  ← (audio, vision)
        y = self.f_a(low_y, hi_x, hi_z, lengths)   # audio ← (text , vision)
        z = self.f_v(low_z, hi_x, hi_y, lengths)   # vision← (text , audio)
        return x, y, z
