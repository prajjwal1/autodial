import torch

def convert(hf_model_path, parlai_model_path):
    hf_model = torch.load(hf_model_path, map_location='cpu')
    parlai_model_original = torch.load(parlai_model_path)
    parlai_model = parlai_model_original['model']
    #  parlai_convention_dict_from_hf = {}

    mapping_to_perform = [
        ['model.', ''],
           ]

    for idx in range(0, 12):
        key_encoder_attn = [f'encoder.layers.{idx}.self_attn_layer_norm', f'encoder.layers.{idx}.norm1']
        mapping_to_perform.append(key_encoder_attn)

    for idx in range(0, 12):
        key_encoder_attn = [f'decoder.layers.{idx}.self_attn_layer_norm', f'decoder.layers.{idx}.norm1']
        mapping_to_perform.append(key_encoder_attn)

    for idx in range(0, 12):
        key_encoder_attn = [f'encoder.layers.{idx}.self_attn', f'encoder.layers.{idx}.attention']
        mapping_to_perform.append(key_encoder_attn)

    all_keys = [
                ['final_layer_norm', 'norm2'],
                ['encoder_attn_layer_norm', 'norm3'],
                ['encoder_attn', 'encoder_attention'],
                ['fc1', 'ffn.lin1'],
                ['fc2', 'ffn.lin2'],
                ['q_proj', 'q_lin'],
                ['k_proj', 'k_lin'],
                ['v_proj', 'v_lin'],
                ['out_proj', 'out_lin'],
                ['self_attn', 'self_attention'],
                ['embed_tokens', 'embeddings'],
                ['embed_positions', 'position_embeddings'],
                ['layernorm_embedding', 'norm_embeddings'],
                ['attn', 'attention'],
            ]
    mapping_to_perform.extend(all_keys)

    for key, val in hf_model.items():
        for u1, u2 in mapping_to_perform:
            key = key.replace(u1, u2)
        if 'position' in key and val.size(0) > 1024:
            val = val[:1024]
        if 'embeddings' in key and val.size(0) > 50264:
            val = val[:50264]
        parlai_model[key] = val

    if 'model.shared.weight' in hf_model:
        parlai_model['embeddings.weight'] = hf_model['model.shared.weight'][:50264]
    if 'shared.weight' in hf_model:
        parlai_model['embeddings.weight'] = hf_model['shared.weight'][:50264]


    print(parlai_model['encoder.embeddings.weight'].shape)
    print(parlai_model['encoder.position_embeddings.weight'].shape)
    print(parlai_model['decoder.embeddings.weight'].shape)
    print(parlai_model['decoder.position_embeddings.weight'].shape)
    print(parlai_model['decoder.norm_embeddings.weight'].shape)

    set1 = parlai_model.keys()
    set2 = parlai_model.keys()
    print(set1^set2)
    print(len(set1^set2))
    print('Done')

    if 'model' in parlai_model:
        parlai_model.pop('model')
    if 'final_logits_bias' in parlai_model:
        parlai_model.pop('final_logits_bias')
    if 'lm_head.weight' in parlai_model:
        parlai_model.pop('lm_head.weight')
    if 'shared.weight' in parlai_model:
        parlai_model.pop('shared.weight')


    parlai_model_original['model'] = parlai_model

    torch.save(parlai_model_original, 'model_from_hf_in_parlai')

if __name__ == '__main__':
    convert('bart_hf_original', 'parlai_model')
