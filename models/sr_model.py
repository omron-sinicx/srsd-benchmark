import torch
from torch import nn
from torchdistill.losses.registry import get_loss
from torchdistill.losses.util import register_func2extract_org_output

from eq.pred import get_predict_function
from eq.vocabulary import SOS_TOKEN, SymbolVocabulary
from .decoder import build_decoder
from .encoder import DimMaxpool, build_encoder, make_mlp
from .registry import register_model_class


@register_func2extract_org_output
def extract_internal_org_loss(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    org_loss_dict = dict()
    org_loss_dict['loss'] = student_outputs['loss']
    return org_loss_dict


@register_model_class
class SymbolicTransformer(nn.Module):
    def __init__(self, model_config, symbols=None):
        super().__init__()
        vocab_config = model_config['vocabulary']
        self.vocabulary = \
            SymbolVocabulary(symbols=symbols, **vocab_config)
        self.encoder = build_encoder(model_config['encoder'])
        self.pooler = DimMaxpool(dim=1, unsqueezes=False)
        symbol_embedding_config = model_config['symbol_embedding']
        vocabulary_size = self.vocabulary.get_size()
        symbol_embedding_config['num_embeddings'] = vocabulary_size
        self.symbol_embedding = nn.Embedding(**symbol_embedding_config)
        self.decoder = build_decoder(model_config['decoder'])
        self.mlp = make_mlp(model_config['mlp'])
        classifier_config = model_config['classifier']
        classifier_config['out_features'] = vocabulary_size
        self.classifier = nn.Linear(**model_config['classifier'])
        self.predict = get_predict_function(model_config['predict_func'])
        self.default_max_length = model_config['default_max_length']
        criterion_config = model_config['criterion']
        self.criteria = get_loss(criterion_config['type'], criterion_config['params'])
        sampling_config = criterion_config.get('sampling', dict())
        self.sampling_strategy = sampling_config.get('strategy', 'exponential')
        self.sampling_factor = sampling_config.get('factor', 0.99)

    def scheduled_sampling(self, decoded_tokens, target_sequences, max_length, device):
        decoding_indices = torch.arange(0, target_sequences.shape[1], device=device)
        if self.sampling_strategy == 'linear':
            threshold_table = torch.max(self.sampling_factor, 1 - decoding_indices / max_length)
        elif self.sampling_strategy == 'exponential':
            threshold_table = self.sampling_factor ** decoding_indices
        elif self.sampling_strategy == 'sigmoid':
            threshold_table = \
                self.sampling_factor / (self.sampling_factor + torch.exp(decoding_indices / self.sampling_factor))
        else:
            raise ValueError(f'sampling_strategy {self.sampling_strategy} is not expected')

        threshold_table = threshold_table.unsqueeze(0).repeat(target_sequences.shape[0], 1).tril()

        # conduct sampling based on the above thresholds
        random_mat = torch.rand(target_sequences.shape, device=device)
        next_decoded_tokens = torch.where(random_mat < threshold_table, target_sequences, decoded_tokens)
        return next_decoded_tokens

    def free_run_train(self, batch_size, src_pool_enc, target_sequences, max_length, unavailable_var_indices, device):
        sos_index = self.vocabulary.symbol2index(SOS_TOKEN)
        decoded_tokens = torch.ones(batch_size, 1, dtype=torch.int64, device=device)
        decoded_tokens[:] = sos_index
        complete_flags = [False for _ in range(batch_size)]
        loss = None
        for i in range(max_length):
            decoded_embeds = self.symbol_embedding(decoded_tokens)
            decoder_output = self.decoder(decoded_embeds, src_pool_enc)
            # We care logits only for the last token
            mlp_output = self.mlp(decoder_output[:, -1:])
            logits = self.classifier(mlp_output)
            # Expand the decoded tokens
            decoded_tokens, logits, complete_flags = \
                self.predict(logits, decoded_tokens, complete_flags, unavailable_var_indices, self.vocabulary)
            if target_sequences is not None:
                tmp_loss = self.criteria(logits[:, -1], target_sequences[:, i])
                if loss is None:
                    loss = tmp_loss
                else:
                    loss += tmp_loss
            if all(complete_flags):
                break
        return loss

    def forward(self, x, target_sequences=None, max_length=None):
        if max_length is None:
            max_length = self.default_max_length

        if target_sequences is not None:
            max_length = target_sequences.shape[1]

        device = x.device
        x = self.encoder(x)
        src_pool_enc = self.pooler(x).flatten(2)
        # Fix src_pool_enc when testing decoder
        # src_pool_enc = torch.ones(1, 5, 512)
        # <SOS> token
        sos_index = self.vocabulary.symbol2index(SOS_TOKEN)
        batch_size = x.shape[0]
        decoded_tokens = torch.ones(batch_size, 1, dtype=torch.int64, device=device)
        decoded_tokens[:] = sos_index
        max_num_vars_in_batch = x.shape[2] - 1
        unavailable_var_indices = self.vocabulary.get_unavailable_variable_indices(max_num_vars_in_batch)
        complete_flags = [False for _ in range(batch_size)]
        loss = None
        for i in range(max_length):
            if self.training and target_sequences is not None and i > 0:
                decoded_tokens[:, 1:] = \
                    self.scheduled_sampling(decoded_tokens[:, 1:], target_sequences[:, :i],
                                            target_sequences.shape[1], device)

            decoded_embeds = self.symbol_embedding(decoded_tokens)
            decoder_output = self.decoder(decoded_embeds, src_pool_enc)
            # We care logits only for the last token
            mlp_output = self.mlp(decoder_output[:, -1:])
            logits = self.classifier(mlp_output)
            # Expand the decoded tokens
            decoded_tokens, logits, complete_flags = \
                self.predict(logits, decoded_tokens, complete_flags,
                             unavailable_var_indices, self.vocabulary, self.training)
            if target_sequences is not None:
                tmp_loss = self.criteria(logits[:, -1], target_sequences[:, i])
                if loss is None:
                    loss = tmp_loss
                else:
                    loss += tmp_loss
            if not self.training and all(complete_flags):
                break

        # if self.training:
        #     loss += self.free_run_train(batch_size, src_pool_enc, target_sequences, max_length,
        #                                 unavailable_var_indices, device)

        output_dict = {'pred': decoded_tokens, 'loss': loss}
        if not self.training:
            output_dict['pred_symbols'] = \
                [self.vocabulary.convert_indices_to_symbols(seq) for seq in decoded_tokens.cpu().detach().tolist()]
        return output_dict
