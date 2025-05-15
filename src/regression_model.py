import torch
import torch.nn as nn
from transformers import BertModel
#from transformers.modeling_utils import no_init_weights, init_empty_weights

class GatedFusion(nn.Module):
    """改进版门控融合模块（支持三特征输入）"""
    def __init__(self, hidden_size):
        super().__init__()
        # 更精细的门控机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.Sigmoid()
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
    def forward(self, feat1, feat2, feat3):
        combined = torch.cat([feat1, feat2, feat3], dim=-1)
        gate_values = self.gate(combined)
        gate1, gate2 = gate_values.chunk(2, dim=-1)
        
        projected = self.projection(combined)
        return gate1 * projected + gate2 * feat1  # 动态残差连接

class HierarchicalCLSExtractor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 更合理的分层权重初始化
        self.weights_low = nn.Parameter(torch.randn(4))
        self.weights_mid = nn.Parameter(torch.randn(4))
        self.weights_high = nn.Parameter(torch.randn(4))
        
        # 修改门控层输入维度
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # 修改为单个hidden_size
            nn.Softmax(dim=-1)
        )
        
    def forward(self, hidden_states):
        # 安全索引处理
        low = self._weighted_sum(hidden_states[1:5], self.weights_low)
        mid = self._weighted_sum(hidden_states[5:9], self.weights_mid)
        high = self._weighted_sum(hidden_states[9:13], self.weights_high)
        
        # 动态融合 - 修改为直接拼接后平均
        combined_features = torch.cat([
            low.unsqueeze(1), 
            mid.unsqueeze(1), 
            high.unsqueeze(1)
        ], dim=1)  # [batch, 3, hidden_size]
        
        # 计算注意力权重
        gate_values = self.gate(combined_features.mean(dim=1))  # [batch, hidden_size]
        
        # 应用注意力
        return (gate_values.unsqueeze(1) * combined_features).sum(dim=1)  # [batch, hidden_size]

    def _weighted_sum(self, layers, weights):
        norm_weights = torch.softmax(weights, dim=0)
        return sum(w * layer[:, 0, :] for w, layer in zip(norm_weights, layers))

class DualBertRegressionModel(nn.Module):
    def __init__(self, pretrained_model_name, dropout_prob=0.1):
        super().__init__()
        
        # 三路BERT编码器
        self.bert_reasoning = BertModel.from_pretrained(pretrained_model_name, 
                                                    output_hidden_states=True)
        self.bert_ans = BertModel.from_pretrained(pretrained_model_name, 
                                                output_hidden_states=True)
        self.bert_content = BertModel.from_pretrained(pretrained_model_name,
                                                    output_hidden_states=True)
        
        # 统一获取hidden_size
        hidden_size = self.bert_reasoning.config.hidden_size
        
        # 特征提取器
        self.extractor = HierarchicalCLSExtractor(hidden_size)
        self.fusion = GatedFusion(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        # 增强回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            ResidualBlock(512, 512, dropout_prob),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            ResidualBlock(256, 256, dropout_prob),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, reasoning_ids, ans_ids, content_ids, 
               reasoning_mask=None, ans_mask=None, content_mask=None, **kwargs):
        # 自动生成mask如果未提供
        reasoning_mask = reasoning_mask if reasoning_mask is not None \
                        else (reasoning_ids != 0).long()
        ans_mask = ans_mask if ans_mask is not None \
                  else (ans_ids != 0).long()
        content_mask = content_mask if content_mask is not None \
                     else (content_ids != 0).long()
        
        # 获取各路径特征
        reasoning_out = self.bert_reasoning(reasoning_ids, 
                                          attention_mask=reasoning_mask,
                                          **kwargs)
        ans_out = self.bert_ans(ans_ids, 
                               attention_mask=ans_mask,
                               **kwargs)
        content_out = self.bert_content(content_ids, 
                                      attention_mask=content_mask,
                                      **kwargs)
        
        # 分层特征提取
        reasoning_feat = self.extractor(reasoning_out.hidden_states)
        ans_feat = self.extractor(ans_out.hidden_states)
        content_feat = self.extractor(content_out.hidden_states)
        
        # 门控融合
        fused = self.fusion(
            self.dropout(reasoning_feat),
            self.dropout(ans_feat),
            self.dropout(content_feat)
        )
        
        return self.regressor(fused)

class ResidualBlock(nn.Module):
    """改进残差块"""
    def __init__(self, in_dim, out_dim, dropout_prob):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        return self.shortcut(x) + self.linear(x)