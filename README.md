# MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models

MiniGPT-4 拥有许多与 GPT-4 类似的能力，例如生成详细的图像描述以及从手写草稿创建网站。MiniGPT-4 展现出其他一些新兴能力，包括根据给定的图像创作故事和诗歌、为图像中显示的问题提供解决方案、根据食物照片教用户如何烹饪等等。在我的 Project 中，我使用 MiniGPT-4 生成对于图像的描述。


激活环境
```bash
conda env create -f environment.yml
conda activate minigptv
```

下载权重 MiniGPT-4 (Vicuna 7B) 为例
`https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view`
`https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main`

修改配置
使用 `eval_configs/minigpt4_benchmark_iconvqa.yaml` 中的配置。
把 TODO 中模型的路径修改。

评估
使用 iconqa 数据集。
下载地址：`https://iconqa2021.s3.us-west-1.amazonaws.com/iconqa_data.zip`
只使用其 choose_txt 部分，使用 `eval_scripts/prepare_iconqa.py` 处理成评估的格式。
```bash
python3 eval_scripts/prepare_iconqa.py \
  --problems /path/to/iconqa_data/problems.json \
  --img-root /path/to/iconqa_data/iconqa/val/choose_txt \
  --out /path/to/iconqa_data/choose_text_val.json \
  --split val \
  --keep-types choose_txt \
  --append-hint \
  --check-images
```
修改配置文件中 datasets 的部分，把 TODO 改为数据集的实际路径。

执行
运行示例：`python sample.py --cfg-path eval_configs/minigpt4_benchmark_iconvqa.yaml --image /path/to/your/image.png`
运行评估：`torchrun --master-port 12345 --nproc_per_node 1 eval_scripts/eval_vqa.py --cfg-path eval_configs/minigpt4_benchmark_iconvqa.yaml  --dataset iconvqa`