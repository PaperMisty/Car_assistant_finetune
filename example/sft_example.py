from dataclasses import dataclass


@dataclass
class SFTConfig:
    # 学习率
    lr_max: float = 3e-5
    lr_min: float = 3e-7
    train_data_size: int = 20000
    # 每一个micro batch的大小
    # 总共有10000个batch
    batch_size: int = 2
    # 梯度累积次数
    # 总共有多少个step: 1250
    gradient_accumulation_steps: int = 8
    # 预热比例
    # 预热比例： 1250*0.1 = 125
    warmup_ratio: float = 0.1
    # 评估间隔
    # 每更新100次参数（100个step），执行一次评估
    # 总共会评估13次
    eval_iter: int = 100
    # 测试集样本数量
    test_data_size: int = 500
    # tensorboard日志输出位置
    log_dir: str = "./logs/02_sft_demo"
    # 日志间隔
    # 每100个step，打印一次train/loss和lr，总共会打印13次
    log_iter: int = 100
    # 最终保存目录
    save_dir: str = "./finetuned/02_sft_demo"


def get_data(sft_config: SFTConfig, data_type, tokenizer):
    """
    获取到训练 / 测试集的数据
    输出：数据的token_ids, assistant_answer_masks
    Args:
        sft_config: 训练的配置
        data_type: 数据类型: train,test
    """

    # 1、加载本地数据
    from datasets import load_dataset

    dataset = load_dataset("./data/ultrachat_200k")
    # 2、加载正确的分片
    split_name = "train_sft" if data_type == "train" else "test_sft"
    data = dataset[split_name]

    data = data.shuffle()

    # 3、获取 训练 / 测试 样本数量

    data_size = sft_config.train_data_size if data_type == "train" else sft_config.test_data_size

    # 4、遍历数据，对每条数据，通过tokenzier,使用apply_chat_template方法，返回token_ids和assistant_masks
    all_token_ids = []

    all_assistant_masks = []
    for i in range(data_size):
        message_list = data[i]["messages"]

        result_dict = tokenizer.apply_chat_template(
            message_list,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            max_length=2400,
            truncation=True,
        )
        all_token_ids.append(result_dict["input_ids"])
        all_assistant_masks.append(result_dict["assistant_masks"])

    # 5、最终输出token_ids和assistant_masks

    return all_token_ids, all_assistant_masks


import torch


def compute_loss(output_logits, labels, assistant_answer_mask):
    """
    计算损失：
    Args:
        output_logits: 模型前向传播输出的logits结果，shape: batch_size, seq_len, vocab_size
        labels: 真实的答案, shape: batch_size, seq_len
        assistant_answer_mask: assistant回答的掩码, shape:batch_size, seq_len
    """
    # 1、对labels进行处理，将labels里面，非assistant_answer部分，置为-100
    labels = labels.masked_fill(assistant_answer_mask != 1, -100)

    # output_logits.shape: batch_size, seq_len, vocab_size
    # 拉平： [batch_size * seq_len, vocab_size]
    output_logits = output_logits.reshape(-1, output_logits.shape[-1])
    # labels同样需要拉平：(batch_size * seq_len, )
    labels = labels.reshape(-1)
    loss = torch.nn.functional.cross_entropy(output_logits, labels, ignore_index=-100)
    return loss


from typing import List
import numpy as np


def cosine_decay_with_warmup(current_step, total_step, warmup_ratio, lr_max, lr_min):
    """
    带预热的 余弦衰减函数
    """

    warmup_step = total_step * warmup_ratio

    if current_step <= warmup_step:

        return lr_max * (current_step + 1) / warmup_step
    else:
        progress = (current_step - warmup_step) / (total_step - warmup_step)

        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

        return lr


def compute_eval_loss(model, test_data_token_ids, test_data_assistant_mask, tokenizer, sft_config: SFTConfig):
    """ """

    model.eval()

    total_batch = (len(test_data_token_ids) + sft_config.batch_size - 1) // sft_config.batch_size
    total_batch_loss = []
    for batch in range(total_batch):

        # 1、张量准备
        # train_data_token_ids[0:2],train_data_token_ids[2:4]
        current_batch_token_ids: List[List] = test_data_token_ids[
            batch * sft_config.batch_size : (batch + 1) * sft_config.batch_size
        ]
        current_batch_assistant_mask: List[List] = test_data_assistant_mask[
            batch * sft_config.batch_size : (batch + 1) * sft_config.batch_size
        ]

        # padding
        # 获取到最长的序列的长度
        max_length = max([len(sample) for sample in current_batch_token_ids])

        # 对每个序列做padding

        for sample_token_ids, sample_mask in zip(current_batch_token_ids, current_batch_assistant_mask):

            padding_length = max_length - len(sample_token_ids)

            sample_token_ids.extend([tokenizer.pad_token_id] * padding_length)

            sample_mask.extend([0] * padding_length)

        current_batch_token_ids_tensor = torch.tensor(current_batch_token_ids, dtype=torch.long).to("cuda")
        current_batch_assistant_mask_tensor = torch.tensor(current_batch_assistant_mask, dtype=torch.long).to("cuda")

        input_ids = current_batch_token_ids_tensor[:, :-1]
        labels = current_batch_token_ids_tensor[:, 1:]

        assistant_mask = current_batch_assistant_mask_tensor[:, 1:]

        # 2、模型前向传播，获取logits
        with torch.no_grad():
            output_logits = model(input_ids).logits

        # 3、计算损失
        loss = compute_loss(output_logits, labels, assistant_mask)

        total_batch_loss.append(loss.item())

    return sum(total_batch_loss) / len(total_batch_loss)


def train(sft_config: SFTConfig):
    # 1、初始化模型，tokenizer，获取数据，构造optimizer,初始化SummaryWriter, tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.optim.adamw import AdamW
    import tqdm
    from torch.utils.tensorboard import SummaryWriter

    model = AutoModelForCausalLM.from_pretrained("./model/Qwen3-0.6B-Base/")
    tokenizer = AutoTokenizer.from_pretrained("./model/Qwen3-0.6B-Base/")
    with open("./new_chat_template.jinja", mode="r") as f:
        tokenizer.chat_template = f.read()
    model.to("cuda")
    model.gradient_checkpointing_enable()
    model.train()
    model.config.use_cache = False

    train_data_token_ids, train_data_assistant_mask = get_data(
        sft_config=sft_config, data_type="train", tokenizer=tokenizer
    )
    test_data_token_ids, test_data_assistant_mask = get_data(
        sft_config=sft_config, data_type="test", tokenizer=tokenizer
    )

    optimizer = AdamW(model.parameters(), lr=sft_config.lr_max)
    optimizer.zero_grad()

    writer = SummaryWriter(log_dir=sft_config.log_dir)

    # 2、遍历数据集，
    total_batch = (len(train_data_token_ids) + sft_config.batch_size - 1) // sft_config.batch_size
    total_step = (total_batch + sft_config.gradient_accumulation_steps - 1) // sft_config.gradient_accumulation_steps

    progress_bar = tqdm.tqdm(total=total_step)

    total_loss = []
    for batch in range(total_batch):

        # 1、张量准备
        # train_data_token_ids[0:2],train_data_token_ids[2:4]
        current_batch_token_ids: List[List] = train_data_token_ids[
            batch * sft_config.batch_size : (batch + 1) * sft_config.batch_size
        ]
        current_batch_assistant_mask: List[List] = train_data_assistant_mask[
            batch * sft_config.batch_size : (batch + 1) * sft_config.batch_size
        ]

        # padding , 按mini_batch做padding
        # 获取到最长的序列的长度
        max_length = max([len(sample) for sample in current_batch_token_ids])

        # 对每个序列做padding

        for sample_token_ids, sample_mask in zip(current_batch_token_ids, current_batch_assistant_mask):

            padding_length = max_length - len(sample_token_ids)

            sample_token_ids.extend([tokenizer.pad_token_id] * padding_length)

            sample_mask.extend([0] * padding_length)

        current_batch_token_ids_tensor = torch.tensor(current_batch_token_ids, dtype=torch.long).to("cuda")
        current_batch_assistant_mask_tensor = torch.tensor(current_batch_assistant_mask, dtype=torch.long).to("cuda")

        input_ids = current_batch_token_ids_tensor[:, :-1]
        labels = current_batch_token_ids_tensor[:, 1:]

        assistant_mask = current_batch_assistant_mask_tensor[:, 1:]

        # 2、模型前向传播，获取logits

        output_logits = model(input_ids).logits

        # 3、计算损失

        loss = compute_loss(output_logits, labels, assistant_mask)

        # 4、反向传播
        total_loss.append(loss.item())
        loss = loss / sft_config.gradient_accumulation_steps

        loss.backward()

        # 5、参数更新

        is_gradient_accumulation_steps = (batch + 1) % sft_config.gradient_accumulation_steps == 0
        last_batch = (batch + 1) == total_batch
        # 如果达到了梯度累积的步数，或者是最后一个批次，才做参数更新
        if is_gradient_accumulation_steps or last_batch:

            current_step = batch // sft_config.gradient_accumulation_steps
            current_step_lr = cosine_decay_with_warmup(
                current_step, total_step, sft_config.warmup_ratio, sft_config.lr_max, sft_config.lr_min
            )
            # optimizer可以传递多组不同的参数的，可以对每组参数，使用不同的lr和权重衰减系数
            # 但是我们这里，构造optimizer时，只传递了一组参数，所以在param_groups（这个属性是一个列表）仅存在一个值，这个值是dict，这个dict里面存储了lr，
            # 需要对这个学习率更新
            optimizer.param_groups[0]["lr"] = current_step_lr

            optimizer.step()

            optimizer.zero_grad()

            # 判断当前是否需要跑验证
            should_eval = (current_step + 1) % sft_config.eval_iter == 0
            is_last_step = (current_step + 1) == total_step
            if should_eval or is_last_step:

                current_loss = compute_eval_loss(
                    model, test_data_token_ids, test_data_assistant_mask, tokenizer, sft_config
                )
                writer.add_scalar("eval/loss", scalar_value=current_loss, global_step=current_step)
                model.train()

            should_log = (current_step + 1) % sft_config.log_iter == 0

            if should_log or is_last_step:
                # 打印训练损失和学习率
                writer.add_scalar("train/lr", scalar_value=current_step_lr, global_step=current_step)
                # total_loss里面存的是微批次的loss,前100个step,对应多少个batch? 100 * gradient_accumulation_steps(8)
                # total_loss = [12,23,34] total_loss[:-1]
                last_step_iter_loss = total_loss[-sft_config.log_iter * sft_config.gradient_accumulation_steps :]

                average_loss = sum(last_step_iter_loss) / len(last_step_iter_loss)

                writer.add_scalar("train/loss", scalar_value=average_loss, global_step=current_step)

            progress_bar.update(1)
    model.config.eos_token_id = 151645  # 151645: <|im_end|>
    model.generation_config.eos_token_id = 151645  #
    tokenizer.eos_token_id = 151645
    model.save_pretrained(sft_config.save_dir)
    tokenizer.save_pretrained(sft_config.save_dir)


if __name__ == "__main__":
    sft_config = SFTConfig()
    train(sft_config)


'''
# 当前情况: CPU和GPU串行:
# CPU: [准备batch1] ----等待GPU---- [准备batch2] ----等待GPU---- [准备batch3] ...
# GPU: ---等待CPU--- [前向+反向1] ---等待CPU--- [前向+反向2] ---等待CPU--- ...

# 理想情况: 
# CPU: [准备batch1] [准备batch2] [准备batch3] [准备batch4] ...
# GPU:             [前向+反向1] [前向+反向2] [前向+反向3] ...

from torch.utils.data import Dataset, DataLoader

class SFTDataset(Dataset):
    def __init__(self, token_ids, assistant_masks, pad_token_id):
        self.token_ids = token_ids
        self.assistant_masks = assistant_masks
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.token_ids[idx], dtype=torch.long),
            "mask": torch.tensor(self.assistant_masks[idx], dtype=torch.long),
        }

def collate_fn(batch):
    """动态 padding，在 DataLoader worker 进程中执行"""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    masks = torch.nn.utils.rnn.pad_sequence(
        [item["mask"] for item in batch], batch_first=True, padding_value=0
    )
    return input_ids, masks

train_loader = DataLoader(
    SFTDataset(train_data_token_ids, train_data_assistant_mask, tokenizer.pad_token_id),
    batch_size=sft_config.batch_size,
    shuffle=False,
    num_workers=4,          # 多进程预取
    pin_memory=True,        # 锁页内存，加速 CPU→GPU 传输
    collate_fn=collate_fn,
)


'''
