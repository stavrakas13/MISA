import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PadPackedSequence(nn.Module):
    """
    Μετατρέπει ένα PackedSequence πίσω σε padded tensor
    εξασφαλίζοντας ότι τα lengths είναι έγκυρα.
    """
    def __init__(self, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first

    def forward(self, packed, lengths):
        # lengths: CPU int64, clamp ώστε να μην ξεφύγει κανένα
        lengths = lengths.detach().to("cpu").long()
        max_seq_len = packed.data.size(0)
        lengths = lengths.clamp(min=1, max=max_seq_len)

        max_len = lengths.max().item()
        padded, _ = pad_packed_sequence(
            packed,
            batch_first=self.batch_first,
            total_length=max_len,
        )
        return padded


class PackSequence(nn.Module):
    """
    Πακετάρει (pack) ένα padded tensor σε PackedSequence
    με ασφάλειες για τύπο, συσκευή και εύρος των lengths.
    """
    def __init__(self, batch_first: bool = True):
        super().__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        # 1) CPU int64
        lengths = lengths.detach().to("cpu").long()

        # 2) Clamp σε [1, real_len]
        real_len = x.size(1) if self.batch_first else x.size(0)
        lengths = lengths.clamp(min=1, max=real_len)

        # 3) Pack (δεν χρειάζεται sort)
        packed = pack_padded_sequence(
            x,
            lengths,
            batch_first=self.batch_first,
            enforce_sorted=False,
        )

        # 4) Επιβολή της ίδιας αναδιάταξης στα lengths
        lengths = lengths[packed.sorted_indices]
        return packed, lengths


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
    ):

        super(RNN_latch, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.merge_bi = merge_bi
        self.rnn_type = rnn_type.lower()

        self.out_size = hidden_size

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

        return out, last_timestep, hidden

class FeedbackUnit(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mod1_sz,
        hi_y_size,  # νέο όρισμα: actual feature size του hi_y
        hi_z_size,  # νέο όρισμα: actual feature size του hi_z
        mask_type="learnable_sequence_mask",
        dropout=0.1,
        device="cpu",
    ):
        super(FeedbackUnit, self).__init__()
        self.mask_type = mask_type
        self.mod1_sz = mod1_sz
        self.hidden_dim = hidden_dim
        

        # αρχικός κώδικας - δεν έπαιρνε υπόψη το πραγματικό μέγεθος του input
        # if mask_type == "learnable_sequence_mask":
        #     self.mask1 = RNN_latch(hidden_dim, mod1_sz, dropout=dropout, device=device)
        #     self.mask2 = RNN_latch(hidden_dim, mod1_sz, dropout=dropout, device=device)
        # else:
        #     self.mask1 = nn.Linear(hidden_dim, mod1_sz)
        #     self.mask2 = nn.Linear(hidden_dim, mod1_sz)

        # νέος κώδικας: χρησιμοποιεί το σωστό input_size για κάθε modality
        if mask_type == "learnable_sequence_mask":
            self.mask1 = RNN_latch(input_size=hi_y_size, hidden_size=mod1_sz, dropout=dropout, device=device)
            self.mask2 = RNN_latch(input_size=hi_z_size, hidden_size=mod1_sz, dropout=dropout, device=device)
        else:
            self.mask1 = nn.Linear(hi_y_size, mod1_sz)
            self.mask2 = nn.Linear(hi_z_size, mod1_sz)

        mask_fn = {
            "learnable_static_mask": self._learnable_static_mask,
            "learnable_sequence_mask": self._learnable_sequence_mask,
        }

        self.get_mask = mask_fn[self.mask_type]

    def _learnable_sequence_mask(self, y, z, lengths=None):
        oy, _, _ = self.mask1(y, lengths)
        oz, _, _ = self.mask2(z, lengths)
        lg = (torch.sigmoid(oy) + torch.sigmoid(oz)) * 0.5
        mask = lg
        return mask

    def _learnable_static_mask(self, y, z, lengths=None):
        y = self.mask1(y)
        z = self.mask2(z)
        mask1 = torch.sigmoid(y)
        mask2 = torch.sigmoid(z)
        mask = (mask1 + mask2) * 0.5
        return mask

    # def forward(self, x, y, z, lengths=None):
    #     mask = self.get_mask(y, z, lengths=lengths)
    #     mask = F.dropout(mask, p=0.2)
    #     x_new = x * mask
    #     return x_new
    
    def forward(self, x, y, z, lengths=None):
        mask = self.get_mask(y, z, lengths=lengths)
        mask = F.dropout(mask, p=0.2)

        # ──► εναρμόνιση μήκους (B, Lx, D) ↔ (B, Lmask, D)
        if mask.size(1) != x.size(1):
            # (B, L, D) → (B, D, L) για 1-D interpolation
            mask = mask.transpose(1, 2)
            mask = F.interpolate(
                mask, size=x.size(1), mode="nearest"   # ή "linear" αν θες λείανση
            )
            mask = mask.transpose(1, 2)               # πίσω στο (B, L, D)

            x_new = x * mask
            return x_new



class Feedback(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        mask_type="learnable_sequence_mask",
        dropout=0.1,
        device="cpu",
    ):
        super(Feedback, self).__init__()

        # αρχικός κώδικας - δεν περνούσε τις πραγματικές διαστάσεις του hi_y και hi_z
        # self.f1 = FeedbackUnit(
        #     hidden_dim,
        #     mod1_sz,
        #     mask_type=mask_type,
        #     dropout=dropout,
        #     device=device,
        # )

        # νέος κώδικας - περνάμε τις πραγματικές διαστάσεις: hi_y_size = 2 * mod2_sz, hi_z_size = 2 * mod3_sz
        self.f1 = FeedbackUnit(
            hidden_dim,
            mod1_sz,
            hi_y_size=2 * mod2_sz,
            hi_z_size=2 * mod3_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )

        self.f2 = FeedbackUnit(
            hidden_dim,
            mod2_sz,
            hi_y_size=2 * mod1_sz,
            hi_z_size=2 * mod3_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )

        self.f3 = FeedbackUnit(
            hidden_dim,
            mod3_sz,
            hi_y_size=2 * mod1_sz,
            hi_z_size=2 * mod2_sz,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )
        print(f"mod2_sz: {mod2_sz}, 2 * mod2_sz = {2 * mod2_sz}")


    def forward(self, low_x, low_y, low_z, hi_x, hi_y, hi_z, lengths=None):
        x = self.f1(low_x, hi_y, hi_z, lengths=lengths)
        y = self.f2(low_y, hi_x, hi_z, lengths=lengths)
        z = self.f3(low_z, hi_x, hi_y, lengths=lengths)
        return x, y, z
