from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, DataSegmentConfig
from zoology.data.associative_recall import MQARConfig

input_seq_len = 64

config = TrainConfig(
    max_epochs=20,
    data=DataConfig(
        train_configs=[MQARConfig(num_examples=10_000, vocab_size=128, input_seq_len=input_seq_len)],
        test_configs=[MQARConfig(num_examples=1_000, vocab_size=128, input_seq_len=input_seq_len)],# , **factory_kwargs)],
    ),
    model=ModelConfig(
        vocab_size=128,
        sequence_mixer=ModuleConfig(name="zoology.mixers.attention.MHA")
    ),
)

configs = [config]