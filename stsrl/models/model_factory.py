from stsrl.models.sts_mlp import SimpleMlpModule


def build_model(num_layers=4, num_cells=256, output_len=1, softmax_head=False):
    return SimpleMlpModule(num_cells=num_cells, num_layers=num_layers, output_len=output_len, softmax_head=softmax_head)