from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, DataSegmentConfig
from zoology.data.associative_recall import MQARConfig

input_seq_len = 64
model_args = dict(
    name="zoology.mixers.attention.MHA",
    kwargs={
        "dropout": 0.1,
        "num_heads": 1
    },
)

config = TrainConfig(
    max_epochs=40,
    data=DataConfig(
        train_configs=[MQARConfig(num_examples=10_240, vocab_size=1024, input_seq_len=input_seq_len)],
        test_configs=[MQARConfig(num_examples=1_024, vocab_size=1024, input_seq_len=input_seq_len)],# , **factory_kwargs)],
    ),
    model=ModelConfig(
        vocab_size=1024,
        sequence_mixer=ModuleConfig(**model_args)
    ),
)

configs = [config]