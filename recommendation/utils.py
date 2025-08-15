import importlib
import os
import re
import sys
import yaml
import html
import hashlib
import datetime
import requests
import tiktoken
from typing import Union, Optional
import logging
from logging import getLogger
from accelerate.utils import set_seed

from dataset import InterDataset 
from model import AbstractModel
from dataset import AbstractDataset

def get_dataset(dataset_name: str):
    """
    数据集工厂函数 (Dataset Factory)。

    根据传入的数据集名称，返回对应的数据集处理【类】。
    pipeline 会用返回的类和 config 来创建实例。
    """
    # 在这里，我们进行一个简单的判断。
    # 无论命令行传入 --dataset AmazonReviews2014 还是 --dataset Baby
    # 只要它们的数据格式都遵循 .inter 规范，我们都返回 InterDataset 这个类来处理。
    
    # 您可以将所有遵循该格式的数据集名称都列在这里
    if dataset_name in ["AmazonReviews2014", "Beauty", "Baby"]: 
        print(f"INFO: Routing dataset '{dataset_name}' to InterDataset loader.")
        return InterDataset
    else:
        # 如果将来有其他类型的数据集，可以在这里添加 elif 分支
        raise ValueError(f"在 utils.py 的 get_dataset 中, 未知或未注册的数据集名称: '{dataset_name}'")


def init_seed(seed, reproducibility):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M")
    return cur


def get_command_line_args_str():
    return '_'.join(sys.argv).replace('/', '|')


# --- MODIFICATION START: get_file_name 函数的完整替换 ---
def get_file_name(config: dict, suffix: str = ''):
    # 1. 生成一个基于完整配置的短哈希，确保唯一性
    #    这个哈希是用来区分不同参数组合的最终标识符
    config_hash = hashlib.md5(str(config).encode(encoding="utf-8")).hexdigest()[:8] # 使用8位哈希

    # 2. 构建一个简洁、可读的文件名前缀
    #    包含模型、数据集、类别以及主要的Sweep参数，以便一眼识别
    parts = []
    if 'model' in config:
        parts.append(config['model'])
    if 'dataset' in config:
        parts.append(config['dataset'])
    if 'category' in config: # 数据集类别通常是一个重要的区分符
        parts.append(config['category'])
    
    # 添加主要 Sweep 参数，确保它们的值在文件名中简洁表示 (替换小数点等)
    # 检查这些键是否存在，以避免KeyError
    if 'lr' in config:
        parts.append(f"lr{str(config['lr']).replace('.', 'p')}") # 例如: lr0p0003
    if 'n_codebook' in config:
        parts.append(f"ncb{config['n_codebook']}") # 例如: ncb32
    if 'temperature' in config:
        parts.append(f"temp{str(config['temperature']).replace('.', 'p')}") # 例如: temp0p07
    if 'n_edges' in config:
        parts.append(f"ne{config['n_edges']}") # 例如: ne100
    if 'propagation_steps' in config:
        parts.append(f"ps{config['propagation_steps']}") # 例如: ps3

    # 如果没有特定的参数来构建前缀，则使用run_id作为基础
    if not parts:
        base_filename = config.get('run_id', 'run')
    else:
        base_filename = "_".join(parts)
    
    # 3. 组合文件名：[基本前缀]-[本地时间]-[配置哈希][后缀]
    # 这样既有可读性，又有时间戳和哈希保证唯一性
    logfilename = "{}-{}-{}{}".format(
        base_filename,
        config['run_local_time'], # 例如: Jun-21-2025_01-50
        config_hash,
        suffix
    )

    # 4. 对最终文件名进行长度检查和截断，确保它远低于文件系统限制 (例如 255 字符)
    # MAX_SAFE_FILENAME_LEN 是一个保守的安全长度限制，可以根据你的文件系统进行调整
    MAX_SAFE_FILENAME_LEN = 150 
    if len(logfilename) > MAX_SAFE_FILENAME_LEN:
        # 截断策略：保留足够的前缀，然后连接时间戳、哈希和后缀，确保唯一性不丢失
        
        # 计算需要保留的唯一部分（哈希 + 时间戳 + 后缀 + 连接符）的长度
        required_unique_part_len = len(config_hash) + len(config['run_local_time']) + len(suffix) + 2 # +2 for hyphens
        
        # 计算前缀可以有多长
        prefix_len = MAX_SAFE_FILENAME_LEN - required_unique_part_len
        
        if prefix_len > 0:
            base_filename_short = base_filename[:prefix_len]
        else:
            base_filename_short = "" # 如果前缀都放不下，就空着

        logfilename = "{}-{}-{}{}".format(
            base_filename_short,
            config['run_local_time'],
            config_hash,
            suffix
        )
        # 再次确认截断后是否仍然超长，如果还超长，则最终强制截断到安全长度
        if len(logfilename) > MAX_SAFE_FILENAME_LEN:
             logfilename = logfilename[:MAX_SAFE_FILENAME_LEN] 

    return logfilename


def init_logger(config: dict):
    LOGROOT = config['log_dir']
    os.makedirs(LOGROOT, exist_ok=True)
    dataset_name = os.path.join(LOGROOT, config["dataset"])
    os.makedirs(dataset_name, exist_ok=True)
    model_name = os.path.join(dataset_name, config["model"])
    os.makedirs(model_name, exist_ok=True)

    logfilename = get_file_name(config, suffix='.log')
    if len(logfilename) > 255:
        logfilename = logfilename[:245] + logfilename[-10:]
    logfilepath = os.path.join(LOGROOT, config["dataset"], config["model"], logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[sh, fh])

    if not config['accelerator'].is_main_process:
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()


def log(message, accelerator, logger, level='info'):
    if accelerator.is_main_process:
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)
        else:
            raise ValueError(f'Invalid log level: {level}')


def get_tokenizer(model_name: str):
    """
    Retrieves the tokenizer for a given model name.

    Args:
        model_name (str): The model name.

    Returns:
        AbstractTokenizer: The tokenizer for the given model name.

    Raises:
        ValueError: If the tokenizer is not found.
    """
    try:
        tokenizer_class = getattr(
            importlib.import_module(f'models.tokenizer'),
            f'{model_name}Tokenizer'
        )
    except:
        raise ValueError(f'Tokenizer for model "{model_name}" not found.')
    return tokenizer_class


def get_model(model_name: str) -> AbstractModel:
    """
    更智能的模型工厂函数。
    它会自动从 models/{model_name}/model.py 路径加载模型类。
    """
    try:
        # 1. 动态地构建模型所在的模块路径
        #    例如，如果 model_name 是 'encoder_retrieve'，路径就是 'models.encoder_retrieve.model'
        module_path = f'models.{model_name}.model'
        
        # 2. 导入这个具体的模块
        model_module = importlib.import_module(module_path)
        
        # 3. 从导入的模块中，根据模型名称获取模型【类】
        #    这里我们约定，类名和模型名(文件夹名)是一样的
        model_class = getattr(model_module, model_name)
        
    except (ImportError, AttributeError) as e:
        # 如果找不到模块或类，给出清晰的错误提示
        print(f"ERROR: 尝试加载模型 '{model_name}' 时失败。 异常: {e}")
        raise ValueError(
            f'Model "{model_name}" not found. '
            f'请检查以下几点：\n'
            f'1. 是否存在名为 "models/{model_name}/" 的文件夹。\n'
            f'2. 该文件夹中是否存在一个 "model.py" 文件。\n'
            f'3. 在 "model.py" 文件中，是否定义了一个与文件夹同名的类，即 class {model_name}(...):'
        )
        
    return model_class

def get_trainer(model_name: Union[str, AbstractModel]):
    """
    Returns the trainer class based on the given model name.

    Parameters:
        model_name (Union[str, AbstractModel]): The name of the model or an instance of the AbstractModel class.

    Returns:
        trainer_class: The trainer class corresponding to the given model name. If the model name is not found, the default Trainer class is returned.
    """
    from trainer import Trainer
    if isinstance(model_name, str):
        try:
            trainer_class = getattr(
                importlib.import_module(f'models.trainer'),
                f'{model_name}Trainer'
            )
            return trainer_class
        except:
            return Trainer
    else:
        return Trainer


def get_pipeline(model_name: Union[str, AbstractModel]):
    """
    Returns the pipeline class based on the given model name.

    Parameters:
        model_name (Union[str, AbstractModel]): The name of the model or an instance of the AbstractModel class.

    Returns:
        pipeline_class: The pipeline class corresponding to the given model name. If the model name is not found, the default Pipeline class is returned.
    """
    from pipeline import Pipeline
    if isinstance(model_name, str):
        try:
            pipeline_class = getattr(
                importlib.import_module(f'models.{model_name}.pipeline'),
                f'{model_name}Pipeline'
            )
            return pipeline_class
        except:
            return Pipeline
    else:
        return Pipeline

def get_total_steps(config, train_dataloader):
    """
    Calculate the total number of steps for training based on the given configuration and dataloader.

    Args:
        config (dict): The configuration dictionary containing the training parameters.
        train_dataloader (DataLoader): The dataloader for the training dataset.

    Returns:
        int: The total number of steps for training.

    """
    if config['steps'] is not None:
        return config['steps']
    else:
        return len(train_dataloader) * config['epochs']


def convert_config_dict(config: dict) -> dict:
    """
    Convert the values in a dictionary to their appropriate types.

    Args:
        config (dict): The dictionary containing the configuration values.

    Returns:
        dict: The dictionary with the converted values.

    """
    for key in config:
        v = config[key]
        if not isinstance(v, str):
            continue
        try:
            new_v = eval(v)
            if new_v is not None and not isinstance(
                new_v, (str, int, float, bool, list, dict, tuple)
            ):
                new_v = v
        except (NameError, SyntaxError, TypeError):
            if isinstance(v, str) and v.lower() in ['true', 'false']:
                new_v = (v.lower() == 'true')
            else:
                new_v = v
        config[key] = new_v
    return config


# /recommendation/utils.py

def get_config(
    model_name: Union[str, AbstractModel],
    dataset_name: Union[str, AbstractDataset],
    config_file: Union[str, list[str], None],
    config_dict: Optional[dict]
) -> dict:
    """
    Get the configuration for a model and dataset.
    Overwrite rule: config_dict > config_file > model config.yaml > default.yaml
    
    # 移除加载数据集配置的逻辑
    """
    final_config = {}
    logger = getLogger()

    # Load default configs
    current_path = os.path.dirname(os.path.realpath(__file__))
    config_file_list = [os.path.join(current_path, 'default.yaml')]

    if isinstance(dataset_name, str):
        final_config['dataset'] = dataset_name
    else:
        logger.info(
            'Custom dataset, '
            'whose config should be manually loaded and passed '
            'via "config_file" or "config_dict".'
        )
        final_config['dataset'] = dataset_name.__class__.__name__

    if isinstance(model_name, str):
        # 你的想法是只加载 models/config.yaml，而不是每个模型的独立配置
        config_file_list.append(
            os.path.join(current_path, f'models/config.yaml')
        )
        final_config['model'] = model_name
    else:
        logger.info(
            'Custom model, '
            'whose config should be manually loaded and passed '
            'via "config_file" or "config_dict".'
        )
        final_config['model'] = model_name.__class__.__name__

    if config_file:
        if isinstance(config_file, str):
            config_file = [config_file]
        config_file_list.extend(config_file)

    for file in config_file_list:
        cur_config = yaml.safe_load(open(file, 'r'))
        if cur_config is not None:
            final_config.update(cur_config)

    if config_dict:
        final_config.update(config_dict)

    final_config['run_local_time'] = get_local_time()
    final_config = convert_config_dict(final_config)
    return final_config


def parse_command_line_args(unparsed: list[str]) -> dict:
    """
    Parses command line arguments and returns a dictionary of key-value pairs.

    Args:
        unparsed (list[str]): A list of command line arguments in the format '--key=value'.

    Returns:
        dict: A dictionary containing the parsed key-value pairs.

    Example:
        >>> parse_command_line_args(['--name=John', '--age=25', '--is_student=True'])
        {'name': 'John', 'age': 25, 'is_student': True}
    """
    args = {}
    for text_arg in unparsed:
        if '=' not in text_arg:
            raise ValueError(f"Invalid command line argument: {text_arg}, please add '=' to separate key and value.")
        key, value = text_arg.split('=')
        key = key[len('--'):]
        try:
            value = eval(value)
        except:
            pass
        args[key] = value
    return args


def download_file(url: str, path: str) -> None:
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        path (str): The path where the downloaded file will be saved.
    """
    logger = getLogger()
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded {os.path.basename(path)}")
    else:
        logger.error(f"Failed to download {os.path.basename(path)}")


def list_to_str(l: Union[list, str], remove_blank=False) -> str:
    """
    Converts a list or a string to a string representation.

    Args:
        l (Union[list, str]): The input list or string.

    Returns:
        str: The string representation of the input.
    """
    ret = ''
    if isinstance(l, list):
        ret = ', '.join(map(str, l))
    else:
        ret = l
    if remove_blank:
        ret = ret.replace(' ', '')
    return ret


def clean_text(raw_text: str) -> str:
    """
    Cleans the raw text by removing HTML tags, special characters, and extra spaces.

    Args:
        raw_text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text

def init_device():
    """
    Set the visible devices for training. Supports multiple GPUs.

    Returns:
        torch.device: The device to use for training.

    """
    import torch
    use_ddp = True if os.environ.get("WORLD_SIZE") else False # Check if DDP is enabled
    if torch.cuda.is_available():
        return torch.device('cuda'), use_ddp
    else:
        return torch.device('cpu'), use_ddp

def config_for_log(config: dict) -> dict:
    config = config.copy()
    config.pop('device', None)
    config.pop('accelerator', None)
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = str(v)
    return config


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
