import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryModule(nn.Module):
    def __init__(
        self,
        memory_slots,
        memory_dim,
        controller_dim,
        dropout=0.3,
        use_gru_style=False,
        use_kv=True,
        num_read_hops=1,
        use_usage_decay=False,
        hard_gate=False
    ):
        super().__init__()
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim
        self.controller_dim = controller_dim
        self.use_gru_style = use_gru_style
        self.use_kv = use_kv
        self.num_read_hops = num_read_hops
        self.use_usage_decay = use_usage_decay
        self.hard_gate = hard_gate

        if self.use_kv:
            self.memory_keys = nn.Parameter(torch.empty(memory_slots, memory_dim))
            self.memory_values = nn.Parameter(torch.empty(memory_slots, memory_dim))
            nn.init.xavier_uniform_(self.memory_keys)
            nn.init.xavier_uniform_(self.memory_values)
        else:
            self.memory = nn.Parameter(torch.empty(memory_slots, memory_dim))
            nn.init.xavier_uniform_(self.memory)

        self.key_layer = nn.Linear(controller_dim, memory_dim)
        self.erase_layer = nn.Linear(controller_dim, memory_dim)
        self.write_layer = nn.Linear(controller_dim, memory_dim)
        self.write_gate = nn.Linear(controller_dim, memory_dim)
        self.gate = nn.Linear(controller_dim, memory_dim)

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(memory_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # learned memory initializer
        self.init_proj = nn.Linear(controller_dim, memory_slots * memory_dim)

        # post-fusion FFN
        self.out_ffn = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(memory_dim),
        )

        if self.use_usage_decay:
            self.register_buffer("usage", torch.zeros(memory_slots))

    def _address_memory(self, key):
        key_norm = F.normalize(key, dim=-1)
        mem = self.memory_keys if self.use_kv else self.memory
        mem_norm = F.normalize(mem, dim=-1)
        sim = torch.matmul(key_norm, mem_norm.T) / self.temperature
        return torch.softmax(sim, dim=-1)

    def read(self, controller_out):
        read_val = controller_out
        for _ in range(self.num_read_hops):
            key = self.key_layer(read_val)
            weights = self._address_memory(key)
            mem = self.memory_values if self.use_kv else self.memory
            read_val = torch.matmul(weights, mem)
        return read_val, weights

    def write(self, controller_out, weights):
        erase = torch.sigmoid(self.erase_layer(controller_out)).unsqueeze(1)
        add = self.write_layer(controller_out).unsqueeze(1)
        gate = torch.sigmoid(self.write_gate(controller_out)).unsqueeze(1)
        add = self.dropout(gate * add)
        weights = weights.unsqueeze(-1)

        erase_matrix = (1 - weights * erase).mean(dim=0)
        add_matrix = (weights * add).mean(dim=0)

        if self.use_usage_decay:
            self.usage = 0.9 * self.usage + weights.squeeze(-1).sum(dim=0).detach()
            erase_matrix *= (1 - self.usage.unsqueeze(-1))

        mem = self.memory_values if self.use_kv else self.memory

        if self.use_gru_style:
            updated = (1 - erase_matrix) * mem.data + erase_matrix * add_matrix
        else:
            updated = mem.data * erase_matrix + add_matrix

        if self.use_kv:
            self.memory_values.data = self.ln(updated)
        else:
            self.memory.data = self.ln(updated)

    def reset(self):
        with torch.no_grad():
            if self.use_kv:
                nn.init.xavier_uniform_(self.memory_keys)
                nn.init.xavier_uniform_(self.memory_values)
            else:
                nn.init.xavier_uniform_(self.memory)
            if self.use_usage_decay:
                self.usage.zero_()

    def begin_sequence(self, controller_init=None):
        with torch.no_grad():
            if controller_init is not None:
                if controller_init.dim() == 3:
                    controller_init = controller_init.mean(dim=1)  # [B, L, D] → [B, D]
                projected = self.init_proj(controller_init)  # [B, S * D]
                # init = projected.view(-1, self.memory_slots, self.memory_dim)  # [B, S, D]
                projected = self.init_proj(controller_init)
                if controller_init.dim() == 2:  # [B, D]
                    init = projected.view(-1, self.memory_slots, self.memory_dim)[0]
                else:  # [D]
                    init = projected.view(self.memory_slots, self.memory_dim)

                if self.use_kv:
                    self.memory_keys.data = F.normalize(init.clone(), dim=-1)
                    self.memory_values.data = init
                else:
                    self.memory = init
            else:
                B = 1  # default fallback batch size
                init = torch.zeros(
                    B, self.memory_slots, self.memory_dim, device=self.device
                )
                if self.use_kv:
                    self.memory_keys.data = F.normalize(init.clone(), dim=-1)
                    self.memory_values.data = init
                else:
                    self.memory = init

            if self.use_usage_decay:
                self.usage.zero_()

    def forward(self, controller_seq, write=True):
        if controller_seq.dim() == 2:
            read_val, weights = self.read(controller_seq)
            if write:
                self.write(controller_seq, weights)
            g = torch.sigmoid(self.gate(controller_seq))
            if self.hard_gate:
                g = (g >= 0.5).float()
            fused = self.ln(g * controller_seq + (1 - g) * read_val)
            return self.out_ffn(fused)

        elif controller_seq.dim() == 3:
            B, T, D = controller_seq.shape

            controller_init = controller_seq.mean(dim=1)
            self.begin_sequence(controller_init=controller_init)
            out = []
            for t in range(T):
                c_t = controller_seq[:, t, :]
                read_val, weights = self.read(c_t)
                if write:
                    self.write(c_t, weights)
                g = torch.sigmoid(self.gate(c_t))
                if self.hard_gate:
                    g = (g >= 0.5).float()
                fused = self.ln(g * c_t + (1 - g) * read_val)
                out_t = self.out_ffn(fused)
                out.append(out_t.unsqueeze(1))
            return torch.cat(out, dim=1)
        else:
            raise ValueError(f"Unsupported input shape: {controller_seq.shape}")
