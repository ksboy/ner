from numpy.core.numeric import NaN
from transformers import BertForTokenClassification
import torch
from torch import nn
from torch.nn import functional as F
from torchcrf import CRF
from transformers import BertModel, BertPreTrainedModel
# from transformers import add_start_docstrings, add_start_docstrings_to_callable

# from loss import FocalLoss, DSCLoss, DiceLoss, LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
import numpy as np

class BertForTokenClassificationMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels_i = config.num_labels_i  # BIO
        self.num_labels_c = config.num_labels_c # 包括 None 

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.identifier = nn.Linear(config.hidden_size, self.num_labels_i)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels_c)
        self.loss = config.loss
        if self.loss == "weightedLoss":
            self.log_var_a = torch.nn.Parameter(torch.zeros((1,), requires_grad=True))
            self.log_var_b = torch.nn.Parameter(torch.zeros((1,), requires_grad=True))

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_c = None,
        labels_i = None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(token_type_ids),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # batch_size, sequence_length, hidden_size = sequence_output.shape

        # identification
        logits_i = self.identifier(sequence_output)
        if labels_i is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_i = logits_i.view(-1, self.num_labels_i)
                active_labels_i = torch.where(
                    active_loss, labels_i.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels_i)
                )
                loss_i = loss_fct(active_logits_i, active_labels_i)
            else:
                loss_i = loss_fct(logits_i.view(-1, self.num_labels_i), labels_i.view(-1))
        
        # classification
        # mask = token_type_ids.unsqueeze(-1).expand_as(sequence_output).bool()
        # entity_embedding = torch.sum(sequence_output * mask, dim=1) / torch.sum(mask, dim=1)
        # # batch_size, hidden_size = entity_embedding.shape

        logits_c = self.classifier(sequence_output)
        if labels_c is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_c = logits_c.view(-1, self.num_labels_c)
                active_labels_c = torch.where(
                    active_loss, labels_c.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels_c)
                )
                loss_c = loss_fct(active_logits_c, active_labels_c)
            else:
                loss_c = loss_fct(logits_c.view(-1, self.num_labels_c), labels_c.view(-1))

        if self.loss == "weightedLoss":
            loss_a = torch.sum(torch.exp(- self.log_var_a) * loss_i + self.log_var_a, -1)
            loss_b = torch.sum(torch.exp(- self.log_var_a) * loss_c + self.log_var_a, -1)
            outputs = ([loss_a + loss_b, loss_a, loss_b],)  + ([logits_i, logits_c],) + outputs
        else:
            outputs = ([loss_i + loss_c, loss_i, loss_c],)  + ([logits_i, logits_c],) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
    

    def predict():
        pass

from seqeval.metrics.sequence_labeling import get_entities
class BertForTokenClassificationJoint(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels_i = config.num_labels_i  # BIO
        self.num_labels_c = config.num_labels_c # 包括 None 

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.identifier = nn.Linear(config.hidden_size, self.num_labels_i)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels_c)
        # self.bilstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=config.hidden_size//2, hidden_size=config.hidden_size, batch_first=True )
        self.pad_token_label_id = config.pad_token_label_id

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_c = None,
        labels_i = None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(token_type_ids),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        batch_size, sequence_length, hidden_size = sequence_output.shape

        # identification
        logits_i = self.identifier(sequence_output)
        if labels_i is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits_i = logits_i.view(-1, self.num_labels_i)
                active_labels_i = torch.where(
                    active_loss, labels_i.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels_i)
                )
                loss_i = loss_fct(active_logits_i, active_labels_i)
            else:
                loss_i = loss_fct(logits_i.view(-1, self.num_labels_i), labels_i.view(-1))
        
        # classification
        mask = token_type_ids.unsqueeze(-1).expand_as(sequence_output).bool()
        entity_embedding = torch.sum(sequence_output * mask, dim=1) / (torch.sum(mask, dim=1)+ 1e-8)
        # batch_size, hidden_size = entity_embedding.shape

        logits_c = self.classifier(entity_embedding.unsqueeze(1))
        logits_c = logits_c.repeat(1, sequence_length, 1)
        if labels_c is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if token_type_ids is not None:
                active_loss = token_type_ids.view(-1) == 1
                active_logits_c = logits_c.view(-1, self.num_labels_c)
                active_labels_c = torch.where(
                    active_loss, labels_c.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels_c)
                )
                loss_c = loss_fct(active_logits_c, active_labels_c)
            else:
                loss_c = loss_fct(logits_c.view(-1, self.num_labels_c), labels_c.view(-1))
        outputs = ([loss_i + loss_c, loss_i, loss_c],) + ([logits_i, logits_c],) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)
    

    def identify(self, sequence_output):
        batch_logits_i = self.identifier(sequence_output)
        batch_preds_i = torch.argmax(batch_logits_i, axis=2).cpu().numpy()
        vf = np.vectorize(lambda x:self.config.id2label_i[str(x)])
        return vf(batch_preds_i).tolist()

    def classify(self, batch_preds_i, labels_i, sequence_output):
        # 去除 无关 token
        # mask = labels_i != self.pad_token_label_id
        batch_size, sequence_length, hidden_size = sequence_output.shape
        
        valid_sequence_output = torch.zeros_like(sequence_output)
        valid_batch_preds_i = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            k = -1
            for j in range(sequence_length):
                if labels_i[i][j].item() != self.pad_token_label_id:
                    k += 1
                    valid_sequence_output[i][k] = sequence_output[i][j]
                    valid_batch_preds_i[i].append(batch_preds_i[i][j])

        batch_entity_list = []
        for preds_i in valid_batch_preds_i:
            entity_list = get_entities(preds_i)
            batch_entity_list.append(entity_list)

        for entity_list, preds_i, context_embedding in zip(batch_entity_list, valid_batch_preds_i, valid_sequence_output):
            for entity in entity_list:
                _, start, end = entity
                entity_embedding = []
                for j in range(start, end+1):
                    entity_embedding.append(context_embedding[j])
                entity_embedding = torch.stack(entity_embedding, dim=0)
                entity_embedding = torch.mean(entity_embedding, dim=0)
                logit_c = self.classifier(entity_embedding.unsqueeze(0))[0]
                pred_c = np.argmax(logit_c.cpu().numpy(), axis=-1)
                entity_type = self.config.id2label_c[str(pred_c)]
                for j in range(start, end+1):
                    preds_i[j]+= "-"+entity_type
        return valid_batch_preds_i

    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_c = None,
        labels_i = None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(token_type_ids),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]

        batch_preds_i = self.identify(sequence_output)
        batch_preds_c = self.classify(batch_preds_i, labels_i, sequence_output)
        return batch_preds_c  # (loss), scores, (hidden_states), (attentions)


class BertForEnrtityClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels_c = config.num_labels_c # 包括 None 

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.identifier = nn.Linear(config.hidden_size, self.num_labels_i)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels_c)
        # self.bilstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=config.hidden_size//2, hidden_size=config.hidden_size, batch_first=True )
        self.pad_token_label_id = config.pad_token_label_id

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_c = None,
        labels_i = None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(token_type_ids),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        batch_size, sequence_length, hidden_size = sequence_output.shape
        
        # classification
        mask = token_type_ids.unsqueeze(-1).expand_as(sequence_output).bool()
        entity_embedding = torch.sum(sequence_output * mask, dim=1) / (torch.sum(mask, dim=1)+ 1e-8)
        # batch_size, hidden_size = entity_embedding.shape

        logits_c = self.classifier(entity_embedding.unsqueeze(1))
        logits_c = logits_c.repeat(1, sequence_length, 1)
        if labels_c is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if token_type_ids is not None:
                active_loss = token_type_ids.view(-1) == 1
                active_logits_c = logits_c.view(-1, self.num_labels_c)
                active_labels_c = torch.where(
                    active_loss, labels_c.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels_c)
                )
                loss_c = loss_fct(active_logits_c, active_labels_c)
            else:
                loss_c = loss_fct(logits_c.view(-1, self.num_labels_c), labels_c.view(-1))
        outputs = ([loss_i + loss_c, loss_i, loss_c],) + ([logits_i, logits_c],) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)
