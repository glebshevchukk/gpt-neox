config = DeepSpeedTransformerConfig(batch_size = 64, max_seq_length = 128, hidden_size = 1024, heads = 16, attn_dropout_ratio = 0.1,
                                    hidden_dropout_ratio = 0.1, num_hidden_layers = 24, initializer_range = 0.02, local_rank = 0,
                                    seed = 1234, fp16 = True, pre_layer_norm=True, attn_dropout_checkpoint=False,
                                    normalize_invertible=False, gelu_checkpoint=False)

self.layer = nn.ModuleList([copy.deepcopy(DeepSpeedTransformerLayer(cuda_config)) for _ in range(config.num_hidden_layers)])
